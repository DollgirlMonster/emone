import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, readdirSync, writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  CHAT_NOISE_ROWS,
  COMPACTION_BACKEND,
  defaultPreset,
  ensureCompactionPreset,
  repointCompactionRow,
  setDefaultPreset,
  withoutRows,
} from "../src/setup.js";

const STANDARD = fileURLToPath(new URL("./fixtures/standard-preset.yml", import.meta.url));

function fixture() {
  return readFileSync(STANDARD, "utf8");
}

function home(settingsText) {
  const dshHome = mkdtempSync(join(tmpdir(), "dsh-emone-test-"));
  if (settingsText !== undefined) writeFileSync(join(dshHome, "settings.yaml"), settingsText);
  return dshHome;
}

function presetAt(dshHome, id) {
  return join(dshHome, ".agent-presets", id, "agent.cordis.yml");
}

function writeUserPreset(dshHome, id, text) {
  mkdirSync(join(dshHome, ".agent-presets", id), { recursive: true });
  writeFileSync(presetAt(dshHome, id), text);
}

function backups(path) {
  const directory = join(path, "..");
  const name = path.split("/").pop();
  return readdirSync(directory).filter((entry) => entry.startsWith(`${name}.bak-`));
}

test("the stock compaction row is replaced, and nothing else is", () => {
  const { text, changed } = repointCompactionRow(fixture());
  assert.equal(changed, true);
  assert.match(
    text,
    /- id: compaction-basic\n\s+name: dsh-emone\/backend\n\s+config:\n\s+maxTokens: 32768/,
  );
  assert.doesNotMatch(text, /@deepseek-ai\/dsh-compaction-basic/);
  for (const kept of [
    "- id: command-compact",
    "- id: tool-result-pruner",
    "- id: compaction\n  name: cordis:group",
    "- id: present",
    "thresholdChars: 8192",
  ]) {
    assert.ok(text.includes(kept), `lost ${kept}`);
  }
});

test("re-pointing is idempotent and tolerates a composition without the row", () => {
  const once = repointCompactionRow(fixture());
  const twice = repointCompactionRow(once.text);
  assert.equal(twice.changed, false);
  assert.equal(twice.text, once.text);

  const bare = "- id: present\n  name: '@deepseek-ai/dsh-tool-present'\n";
  const untouched = repointCompactionRow(bare);
  assert.equal(untouched.changed, false);
  assert.equal(untouched.text, bare);
});

test("the default preset is read and only set when absent", () => {
  const settings = "ui-theme:\n  preference: dark\nagent-presets:\n  default: qwen38\n";
  assert.equal(defaultPreset(settings), "qwen38");
  assert.equal(setDefaultPreset(settings, "emone").changed, false);
  assert.equal(defaultPreset("ui-theme:\n  preference: dark\n"), null);

  const inserted = setDefaultPreset("agent-presets:\nother: 1\n", "emone");
  assert.equal(inserted.changed, true);
  assert.match(inserted.text, /agent-presets:\n {2}default: emone\nother: 1\n/);

  const appended = setDefaultPreset("ui-theme:\n  preference: dark\n", "emone");
  assert.equal(appended.changed, true);
  assert.equal(defaultPreset(appended.text), "emone");
});

test("the plugin's own preset is generated and the default is set when unset", () => {
  const dshHome = home("ui-theme:\n  preference: dark\n");
  const messages = [];
  const result = ensureCompactionPreset({
    dshHome,
    presetId: "emone",
    standardPath: STANDARD,
    log: (message) => messages.push(message),
  });
  assert.equal(result.generated, true);
  assert.equal(result.defaultSet, true);
  const generated = readFileSync(presetAt(dshHome, "emone"), "utf8");
  assert.ok(generated.includes(COMPACTION_BACKEND));
  assert.equal(defaultPreset(readFileSync(join(dshHome, "settings.yaml"), "utf8")), "emone");
  assert.equal(backups(join(dshHome, "settings.yaml")).length, 1);
  assert.ok(messages.some((message) => message.includes("wrote the emone agent preset")));
});

test("a user's own default preset is adopted, not replaced", () => {
  const dshHome = home("agent-presets:\n  default: qwen38\n");
  writeUserPreset(dshHome, "qwen38", fixture());
  const result = ensureCompactionPreset({
    dshHome,
    presetId: "emone",
    standardPath: STANDARD,
    log: () => {},
  });
  assert.equal(result.adopted, "qwen38");
  const adopted = readFileSync(presetAt(dshHome, "qwen38"), "utf8");
  assert.ok(adopted.includes(COMPACTION_BACKEND));
  assert.ok(adopted.includes("- id: tool-result-pruner"));
  assert.equal(backups(presetAt(dshHome, "qwen38")).length, 1);
  // The person's choice of preset is left alone.
  assert.equal(defaultPreset(readFileSync(join(dshHome, "settings.yaml"), "utf8")), "qwen38");
});

test("adoption can be turned off, and a shipped default is never edited", () => {
  const off = home("agent-presets:\n  default: qwen38\n");
  writeUserPreset(off, "qwen38", fixture());
  const result = ensureCompactionPreset({
    dshHome: off,
    presetId: "emone",
    standardPath: STANDARD,
    adopt: false,
    log: () => {},
  });
  assert.equal(result.adopted, null);
  assert.ok(
    readFileSync(presetAt(off, "qwen38"), "utf8").includes("@deepseek-ai/dsh-compaction-basic"),
  );

  const shipped = home("agent-presets:\n  default: standard\n");
  const untouched = ensureCompactionPreset({
    dshHome: shipped,
    presetId: "emone",
    standardPath: STANDARD,
    log: () => {},
  });
  assert.equal(untouched.adopted, null);
  assert.equal(untouched.generated, true);
  assert.equal(defaultPreset(readFileSync(join(shipped, "settings.yaml"), "utf8")), "standard");
});

test("a second run changes nothing", () => {
  const dshHome = home("agent-presets:\n  default: qwen38\n");
  writeUserPreset(dshHome, "qwen38", fixture());
  ensureCompactionPreset({ dshHome, presetId: "emone", standardPath: STANDARD, log: () => {} });
  const before = readFileSync(presetAt(dshHome, "qwen38"), "utf8");
  const second = ensureCompactionPreset({
    dshHome,
    presetId: "emone",
    standardPath: STANDARD,
    log: () => {},
  });
  assert.equal(second.generated, false);
  assert.equal(second.adopted, "qwen38");
  assert.equal(readFileSync(presetAt(dshHome, "qwen38"), "utf8"), before);
  assert.equal(backups(presetAt(dshHome, "qwen38")).length, 1);
});

test("a missing standard preset is reported, not fatal", () => {
  const dshHome = home("agent-presets:\n  default: qwen38\n");
  const messages = [];
  const result = ensureCompactionPreset({
    dshHome,
    presetId: "emone",
    standardPath: join(dshHome, "absent.yml"),
    log: (message) => messages.push(message),
  });
  assert.equal(result.preset, null);
  assert.ok(messages.some((message) => message.includes("no standard preset")));
});

test("withoutRows drops a row with its body and keeps the rest", () => {
  // The rows it targets are not tool definitions, so the engine's CLI-strip
  // cannot drop their prompt cost; the tool rows around them must survive,
  // because that strip is what handles those.
  const text = [
    "- id: persona",
    "  name: '@deepseek-ai/dsh-persona'",
    "- id: agent-instructions",
    "  name: '@deepseek-ai/dsh-agent-instructions'",
    "  config:",
    "    maxBytes: 65536",
    "- id: tool-bash",
    "  name: '@deepseek-ai/dsh-tool-bash'",
    "- id: tool-skill",
    "  name: '@deepseek-ai/dsh-tool-skill'",
    "  config:",
    "    catalog: true",
    "- id: tool-fs",
    "  name: '@deepseek-ai/dsh-tool-fs'",
  ].join("\n");
  const { text: stripped, changed } = withoutRows(text);
  assert.equal(changed, true);
  assert.equal(stripped.includes("agent-instructions"), false);
  assert.equal(stripped.includes("maxBytes"), false, "the row's whole body goes");
  assert.equal(stripped.includes("tool-skill"), false);
  assert.equal(stripped.includes("catalog: true"), false);
  assert.ok(stripped.includes("persona"));
  assert.ok(stripped.includes("dsh-tool-bash"));
  assert.ok(stripped.includes("dsh-tool-fs"));
  // Idempotent: nothing left to remove the second time.
  assert.equal(withoutRows(stripped).changed, false);
});

test("the generated preset is chat-shaped, not agent-shaped", () => {
  const dshHome = home("agent-presets:\n  default: qwen38\n");
  const standardPath = join(dshHome, "standard.yml");
  writeFileSync(
    standardPath,
    [
      "- id: agent-instructions",
      "  name: '@deepseek-ai/dsh-agent-instructions'",
      "- id: compaction",
      "  name: cordis:group",
      "  config:",
      "    - id: compaction-basic",
      "      name: '@deepseek-ai/dsh-compaction-basic'",
      "- id: tool-skill",
      "  name: '@deepseek-ai/dsh-tool-skill'",
    ].join("\n"),
  );
  ensureCompactionPreset({
    dshHome,
    presetId: "emone",
    standardPath,
    adopt: false,
    log: () => {},
  });
  const written = readFileSync(presetAt(dshHome, "emone"), "utf8");
  for (const row of CHAT_NOISE_ROWS) {
    assert.equal(written.includes(`- id: ${row}`), false, `${row} should be dropped`);
  }
  // The compaction mount still happens: that is this module's other job.
  assert.ok(written.includes(COMPACTION_BACKEND));
});
