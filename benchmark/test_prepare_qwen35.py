#!/usr/bin/env python3
"""Tests for the dense converter's System One path: merging a Kev LoRA and
carrying its pointer head (Emone-lmw.1).

What these pin, each of which would fail silently in the field:

- the merge arithmetic. `W + alpha/r * B @ A`, applied a row chunk at a time,
  must equal the whole-matrix merge, or quantized rows near a chunk boundary
  get another row's delta and the model answers slightly wrong forever.
- the refusals. DoRA, rsLoRA, trained token rows, a head fitted on another
  base revision -- each merges or loads without error and scores garbage, so
  the converter must refuse them rather than approximate.
- the head reader. `head.pt` is a torch pickle downloaded from the Hub, and
  this file reads it without torch. The reader resolves three globals and
  nothing else; a pickle naming any other callable must be refused before it
  can run.

    cd benchmark && python3 -m unittest test_prepare_qwen35 -v

It imports the converter, which imports numpy and ml_dtypes, so it skips
where those are absent. The full-model parity it rests on is in
benchmark/kev/ (merged vs unmerged probabilities agree to 2e-6).
"""

from __future__ import annotations

import json
import pathlib
import pickle
import shutil
import struct
import sys
import tempfile
import unittest
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

try:
    import numpy as np

    import prepare_qwen35 as prepare

    IMPORT_ERROR = ""
except (SystemExit, ImportError) as exc:  # the module exits when a dependency is missing
    prepare = None
    IMPORT_ERROR = str(exc)


# --- a torch checkpoint, written without torch ---------------------------------
#
# The opcodes torch.save emits for a dict of float32 tensors (protocol 2): each
# tensor is `torch._utils._rebuild_tensor_v2(storage, offset, size, stride,
# requires_grad, hooks)`, and each storage a persistent id naming a file in the
# zip's data/ directory.


def _str(s: str) -> bytes:
    raw = s.encode()
    return b"X" + struct.pack("<I", len(raw)) + raw


def _int(i: int) -> bytes:
    return b"J" + struct.pack("<i", i)


def _global(module: str, name: str) -> bytes:
    return b"c" + module.encode() + b"\n" + name.encode() + b"\n"


def _tuple(*items: bytes) -> bytes:
    return b"(" + b"".join(items) + b"t"


def _ordered_dict(pairs: list[tuple[bytes, bytes]]) -> bytes:
    body = _global("collections", "OrderedDict") + b")R"
    if pairs:
        body += b"(" + b"".join(k + v for k, v in pairs) + b"u"
    return body


def _tensor(key: str, shape: tuple[int, ...], stride: tuple[int, ...] | None = None) -> bytes:
    count = 1
    for d in shape:
        count *= d
    if stride is None:
        stride = tuple(int(np.prod(shape[i + 1 :])) for i in range(len(shape)))
    storage = (
        _tuple(
            _str("storage"), _global("torch", "FloatStorage"), _str(key), _str("cpu"), _int(count)
        )
        + b"Q"
    )
    return (
        _global("torch._utils", "_rebuild_tensor_v2")
        + _tuple(
            storage,
            _int(0),
            _tuple(*map(_int, shape)),
            _tuple(*map(_int, stride)),
            b"\x89",
            _ordered_dict([]),
        )
        + b"R"
    )


def write_head_pt(
    path: pathlib.Path, tensors: dict[str, np.ndarray], scalars: dict, stride=None
) -> None:
    items = []
    entries = []
    for key, (name, value) in enumerate(tensors.items()):
        entries.append((_str(name), _tensor(str(key), value.shape, stride)))
    items.append((_str("head"), _ordered_dict(entries)))
    for name, value in scalars.items():
        if isinstance(value, float):
            encoded = b"G" + struct.pack(">d", value)
        elif isinstance(value, bool):
            encoded = b"\x88" if value else b"\x89"
        elif isinstance(value, int):
            encoded = _int(value)
        else:
            encoded = _str(value)
        items.append((_str(name), encoded))
    data = b"\x80\x02}(" + b"".join(k + v for k, v in items) + b"u."
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("head/data.pkl", data)
        archive.writestr("head/byteorder", "little")
        for key, value in enumerate(tensors.values()):
            archive.writestr(f"head/data/{key}", np.ascontiguousarray(value, dtype="<f4").tobytes())


@unittest.skipIf(prepare is None, f"prepare_qwen35 unavailable: {IMPORT_ERROR}")
class KevTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = pathlib.Path(tempfile.mkdtemp(prefix="kev-convert-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.size = prepare.SIZES["kev-0.8b"]
        self.rng = np.random.default_rng(0)


class HeadReaderTests(KevTestCase):
    def head(self, dim: int = 8, hidden: int = 16) -> dict[str, np.ndarray]:
        return {
            "q.weight": self.rng.standard_normal((dim, hidden)).astype(np.float32),
            "q.bias": self.rng.standard_normal(dim).astype(np.float32),
            "k.weight": self.rng.standard_normal((dim, hidden)).astype(np.float32),
            "k.bias": self.rng.standard_normal(dim).astype(np.float32),
        }

    def scalars(self, **over) -> dict:
        base = {
            "temperature": 2.25,
            "head_dim": 8,
            "base_revision": self.size.commit,
            "option_isolation": False,
        }
        return {**base, **over}

    def test_reads_tensors_and_scalars_exactly(self) -> None:
        head = self.head()
        write_head_pt(self.root / "head.pt", head, self.scalars())
        meta = prepare.read_kev_head(self.root / "head.pt")
        self.assertEqual(meta["temperature"], 2.25)
        self.assertEqual(meta["base_revision"], self.size.commit)
        for name, value in head.items():
            np.testing.assert_array_equal(meta["head"][name], value)
        checked = prepare.check_kev_head(meta, self.size, hidden=16)
        self.assertEqual(sorted(checked), sorted(head))

    def test_refuses_any_other_global_before_calling_it(self) -> None:
        marker = self.root / "ran"
        # GLOBAL os.system, then call it: what a hostile checkpoint looks like.
        data = b"\x80\x02" + _global("os", "system") + _tuple(_str(f"touch {marker}")) + b"R."
        with zipfile.ZipFile(self.root / "head.pt", "w") as archive:
            archive.writestr("head/data.pkl", data)
        with self.assertRaises(pickle.UnpicklingError):
            prepare.read_kev_head(self.root / "head.pt")
        self.assertFalse(marker.exists())

    def test_refuses_a_strided_tensor(self) -> None:
        write_head_pt(self.root / "head.pt", {"q.bias": np.zeros(4, np.float32)}, {}, stride=(2,))
        with self.assertRaises(pickle.UnpicklingError):
            prepare.read_kev_head(self.root / "head.pt")

    def test_refuses_a_head_fitted_on_another_revision_or_width(self) -> None:
        write_head_pt(self.root / "head.pt", self.head(), self.scalars(base_revision="0" * 40))
        with self.assertRaises(SystemExit):
            prepare.check_kev_head(
                prepare.read_kev_head(self.root / "head.pt"), self.size, hidden=16
            )
        write_head_pt(self.root / "wide.pt", self.head(), self.scalars())
        with self.assertRaises(SystemExit):
            prepare.check_kev_head(
                prepare.read_kev_head(self.root / "wide.pt"), self.size, hidden=32
            )

    def test_refuses_a_missing_or_zero_temperature(self) -> None:
        write_head_pt(self.root / "head.pt", self.head(), self.scalars(temperature=0.0))
        with self.assertRaises(SystemExit):
            prepare.check_kev_head(
                prepare.read_kev_head(self.root / "head.pt"), self.size, hidden=16
            )

    def test_written_head_carries_what_the_scorer_needs(self) -> None:
        head = self.head()
        prepare.write_decision_head(
            self.root, head, {"temperature": 2.25, "head_dim": 8}, self.size
        )
        raw = (self.root / prepare.DECISION_HEAD_FILE).read_bytes()
        (length,) = struct.unpack("<Q", raw[:8])
        header = json.loads(raw[8 : 8 + length])
        meta = header.pop("__metadata__")
        self.assertEqual(meta["format"], "kev-pointer-v1")
        self.assertEqual(float(meta["temperature"]), 2.25)
        self.assertEqual(json.loads(meta["delimiters"]), list(prepare.KEV_DELIMITERS))
        self.assertEqual(meta["adapter_commit"], self.size.adapter.commit)
        self.assertEqual((8 + length) % 8, 0)
        payload = raw[8 + length :]
        for name, value in head.items():
            start, stop = header[name]["data_offsets"]
            got = np.frombuffer(payload[start:stop], dtype="<f4").reshape(header[name]["shape"])
            np.testing.assert_array_equal(got, value)


class AdapterTests(KevTestCase):
    def config(self, **over) -> dict:
        base = {
            "peft_type": "LORA",
            "base_model_name_or_path": self.size.repo,
            "r": 16,
            "lora_alpha": 32,
            "bias": "none",
            "use_dora": False,
            "use_rslora": False,
            "fan_in_fan_out": False,
            "modules_to_save": None,
            "trainable_token_indices": None,
            "rank_pattern": {},
            "alpha_pattern": {},
        }
        return {**base, **over}

    def test_accepts_kev_as_published(self) -> None:
        prepare.check_adapter_config(self.config(), self.size)

    def test_refuses_what_a_plain_merge_would_get_wrong(self) -> None:
        for over in (
            {"use_dora": True},
            {"use_rslora": True},
            {"r": 8},
            {"base_model_name_or_path": "Qwen/Qwen3.5-0.8B"},
            {"trainable_token_indices": {"embed_tokens": [1]}},
            {"modules_to_save": ["head"]},
            {"rank_pattern": {"q_proj": 4}},
            {"bias": "all"},
        ):
            with self.subTest(over=over), self.assertRaises(SystemExit):
                prepare.check_adapter_config(self.config(**over), self.size)

    def test_names_map_onto_the_checkpoint(self) -> None:
        self.assertEqual(
            prepare.lora_source_name(
                "base_model.model.layers.3.linear_attn.in_proj_qkv.lora_B.weight"
            ),
            ("model.language_model.layers.3.linear_attn.in_proj_qkv.weight", "B"),
        )
        for bad in (
            "base_model.model.embed_tokens.lora_A.weight",
            "layers.0.mlp.up_proj.lora_A.weight",
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                prepare.lora_source_name(bad)

    def write_adapter(self, tensors: dict[str, np.ndarray]) -> pathlib.Path:
        header, blobs, cursor = {}, [], 0
        for name, value in tensors.items():
            blob = np.ascontiguousarray(value, dtype="<f4").tobytes()
            header[name] = {
                "dtype": "F32",
                "shape": list(value.shape),
                "data_offsets": [cursor, cursor + len(blob)],
            }
            blobs.append(blob)
            cursor += len(blob)
        raw = json.dumps(header).encode()
        raw += b" " * (-len(raw) % 8)
        path = self.root / "adapter_model.safetensors"
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))
        return path

    def test_loads_pairs_and_refuses_an_orphan(self) -> None:
        stem = "base_model.model.layers.0.mlp.up_proj"
        a = self.rng.standard_normal((16, 64)).astype(np.float32)
        b = self.rng.standard_normal((32, 16)).astype(np.float32)
        pairs = prepare.load_lora(
            self.write_adapter({f"{stem}.lora_A.weight": a, f"{stem}.lora_B.weight": b}), self.size
        )
        got_a, got_b = pairs["model.language_model.layers.0.mlp.up_proj.weight"]
        np.testing.assert_array_equal(got_a, a)
        np.testing.assert_array_equal(got_b, b)
        with self.assertRaises(SystemExit):
            prepare.load_lora(self.write_adapter({f"{stem}.lora_A.weight": a}), self.size)


class MergeTests(KevTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.name = "model.language_model.layers.0.mlp.down_proj.weight"
        self.a = self.rng.standard_normal((16, 64)).astype(np.float32)
        self.b = self.rng.standard_normal((40, 16)).astype(np.float32)
        self.w = self.rng.standard_normal((40, 64)).astype(np.float32)
        saved = (prepare.LORA, prepare.LORA_SCALE)
        self.addCleanup(
            lambda: setattr(prepare, "LORA", saved[0]) or setattr(prepare, "LORA_SCALE", saved[1])
        )
        prepare.LORA = {self.name: (self.a, self.b)}
        prepare.LORA_SCALE = 32 / 16

    def test_chunked_merge_equals_the_whole_matrix(self) -> None:
        whole = self.w + 2.0 * (self.b @ self.a)
        for step in (1, 7, 40):
            with self.subTest(step=step):
                rows = [
                    prepare.merged(self.name, i, self.w[i : i + step]) for i in range(0, 40, step)
                ]
                np.testing.assert_allclose(np.concatenate(rows), whole, rtol=0, atol=1e-5)

    def test_unadapted_tensors_are_untouched(self) -> None:
        piece = self.w[:5]
        self.assertIs(prepare.merged("model.language_model.norm.weight", 0, piece), piece)

    def test_a_chunk_past_the_adapter_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            prepare.merged(self.name, 38, self.w[:5])


class SizeTests(KevTestCase):
    def test_kev_is_pinned_to_the_base_its_adapter_names(self) -> None:
        self.assertEqual(self.size.repo, "Qwen/Qwen3.5-0.8B-Base")
        self.assertEqual(len(self.size.commit), 40)
        self.assertEqual(len(self.size.adapter.commit), 40)
        self.assertEqual(
            (self.size.layers, self.size.hidden, self.size.tied_output), (24, 1024, True)
        )

    def test_chat_sizes_carry_no_adapter(self) -> None:
        for key, size in prepare.SIZES.items():
            if key not in ("kev-0.8b", "rev-0.8b"):
                self.assertIsNone(size.adapter, key)

    def test_rev_shares_the_post_trained_base_of_the_side_model(self) -> None:
        rev, side = prepare.SIZES["rev-0.8b"], prepare.SIZES["0.8b"]
        self.assertEqual((rev.repo, rev.commit), (side.repo, side.commit))
        self.assertEqual((rev.layers, rev.hidden, rev.tied_output), (24, 1024, True))
        pinned = dict(rev.adapter.sha256)
        self.assertEqual(sorted(pinned), sorted(prepare.ADAPTER_FILES))
        self.assertTrue(all(len(d) == 64 for d in pinned.values()))
        self.assertEqual(rev.adapter.commit, pinned["adapter_model.safetensors"])


class AdapterPackageTests(KevTestCase):
    def test_package_carries_every_pair_and_pins_its_files(self) -> None:
        rev = prepare.SIZES["rev-0.8b"]
        lora = {
            "model.language_model.layers.0.mlp.up_proj.weight": (
                self.rng.standard_normal((16, 8)).astype(np.float32),
                self.rng.standard_normal((12, 16)).astype(np.float32),
            ),
            "model.language_model.layers.3.linear_attn.in_proj_a.weight": (
                self.rng.standard_normal((16, 8)).astype(np.float32),
                self.rng.standard_normal((4, 16)).astype(np.float32),
            ),
        }
        head = {
            "q.weight": np.ones((4, 8), np.float32),
            "q.bias": np.zeros(4, np.float32),
            "k.weight": np.ones((4, 8), np.float32),
            "k.bias": np.zeros(4, np.float32),
        }
        prepare.write_adapter_package(
            self.root, lora, head, {"temperature": 2.5, "head_dim": 4}, rev
        )
        raw = (self.root / prepare.ADAPTER_PACKAGE_FILE).read_bytes()
        (length,) = struct.unpack("<Q", raw[:8])
        header = json.loads(raw[8 : 8 + length])
        self.assertEqual(header.pop("__metadata__")["format"], "emone-lora-v1")
        self.assertEqual(
            sorted(header),
            [
                "layers.0.mlp.up_proj.lora_A",
                "layers.0.mlp.up_proj.lora_B",
                "layers.3.linear_attn.in_proj_a.lora_A",
                "layers.3.linear_attn.in_proj_a.lora_B",
            ],
        )
        entry = header["layers.0.mlp.up_proj.lora_B"]
        start, stop = entry["data_offsets"]
        body = raw[8 + length :]
        np.testing.assert_array_equal(
            np.frombuffer(body[start:stop], "<f4").reshape(entry["shape"]),
            lora["model.language_model.layers.0.mlp.up_proj.weight"][1],
        )
        manifest = json.loads((self.root / prepare.ADAPTER_MANIFEST_FILE).read_text())
        self.assertEqual(manifest["base"]["model_id"], prepare.SIZES["0.8b"].model_id_stem)
        self.assertEqual((manifest["rank"], manifest["alpha"]), (16, 32))
        for name, digest in manifest["files"].items():
            self.assertEqual(prepare.file_sha256(self.root / name), digest)

    def test_kev_has_no_chat_base_to_lay_over(self) -> None:
        with self.assertRaises(SystemExit):
            prepare.base_model_id(prepare.SIZES["kev-0.8b"])


class LocalAdapterTests(KevTestCase):
    def adapter(self, files: dict[str, bytes]) -> prepare.Adapter:
        import hashlib

        return prepare.Adapter(
            "local/test",
            "x" * 64,
            16,
            32,
            tuple((n, hashlib.sha256(b).hexdigest()) for n, b in files.items()),
        )

    def files(self) -> dict[str, bytes]:
        return {name: name.encode() * 3 for name in prepare.ADAPTER_FILES}

    def test_copies_files_whose_digests_match(self) -> None:
        source, target = self.root / "src", self.root / "dst"
        source.mkdir(), target.mkdir()
        files = self.files()
        for name, blob in files.items():
            (source / name).write_bytes(blob)
        prepare.copy_local_adapter(source, target, self.adapter(files))
        for name, blob in files.items():
            self.assertEqual((target / name).read_bytes(), blob)

    def test_refuses_a_file_that_is_not_the_pinned_one(self) -> None:
        source, target = self.root / "src", self.root / "dst"
        source.mkdir(), target.mkdir()
        files = self.files()
        adapter = self.adapter(files)
        for name, blob in files.items():
            (source / name).write_bytes(blob)
        (source / "head.pt").write_bytes(b"retrained")
        with self.assertRaises(SystemExit):
            prepare.copy_local_adapter(source, target, adapter)
        self.assertFalse((target / "head.pt").exists())

    def test_local_adapter_config_may_name_the_trainers_path(self) -> None:
        size = self.size._replace(adapter=self.adapter(self.files()))
        config = {
            "peft_type": "LORA",
            "base_model_name_or_path": "/some/trainer/path/qwen35-0.8b",
            "r": 16,
            "lora_alpha": 32,
        }
        prepare.check_adapter_config(config, size)
        with self.assertRaises(SystemExit):
            prepare.check_adapter_config({**config, "r": 8}, size)


if __name__ == "__main__":
    unittest.main()
