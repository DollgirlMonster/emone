/**
 * The compaction backend the generated preset mounts.
 *
 * All the behaviour is in `./compaction.js`; this file is only the binding to
 * the real engine, so the tested unit needs no harness packages on its import
 * path.
 *
 * @module dsh-emone/backend
 */
import BasicCompactionEngine from "@deepseek-ai/dsh-compaction-basic";
import { createAuxiliaryQuietCompaction } from "./compaction.js";

/** `dsh-compaction-basic` with auxiliary calls forced to think off. */
export class EmoneCompaction extends createAuxiliaryQuietCompaction(BasicCompactionEngine) {}

export default EmoneCompaction;
