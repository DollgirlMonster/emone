/**
 * The models watcher: one refresh per burst, and never a reason the harness
 * cannot exit.
 *
 * The filesystem is injected, so these tests are about the debounce and the
 * lifecycle rather than about `fs.watch` timing.
 */
import assert from "node:assert/strict";
import test from "node:test";

import { watchModels } from "../src/models-watch.js";

/** A watcher whose listener and close() the test drives by hand. */
function fakeWatch() {
  const state = { listener: null, closed: 0, unref: 0, options: null };
  const watch = (_dir, options, listener) => {
    state.options = options;
    state.listener = listener;
    return {
      close: () => {
        state.closed += 1;
      },
      unref: () => {
        state.unref += 1;
      },
    };
  };
  return { state, watch };
}

/** Timers that only run when the test says so. */
function fakeTimers() {
  const pending = new Map();
  let next = 1;
  return {
    setTimer: (callback, ms) => {
      const id = next++;
      pending.set(id, { callback, ms });
      return {
        id,
        unref() {
          this.unreffed = true;
        },
      };
    },
    clearTimer: (handle) => {
      pending.delete(handle?.id ?? handle);
    },
    runAll: () => {
      const entries = [...pending.values()];
      pending.clear();
      for (const entry of entries) entry.callback();
    },
    get size() {
      return pending.size;
    },
  };
}

test("a burst of changes refreshes once, after the folder goes quiet", () => {
  const { state, watch } = fakeWatch();
  const timers = fakeTimers();
  let refreshes = 0;
  const handle = watchModels({
    modelsDir: "/models",
    refresh: () => {
      refreshes += 1;
    },
    watch,
    setTimer: timers.setTimer,
    clearTimer: timers.clearTimer,
    debounceMs: 250,
  });

  assert.equal(handle.watching, true);
  assert.deepEqual(state.options, { persistent: false });
  assert.equal(state.unref, 1, "the watcher must not hold the process open");

  // An install lands as thousands of events: still one refresh, one timer.
  state.listener("rename", "qwen3.5_2B_4Bit");
  state.listener("change", "manifest.json");
  state.listener("rename", "qwen3.5_2B_4Bit");
  assert.equal(timers.size, 1, "the quiet period restarts rather than stacking");
  assert.equal(refreshes, 0, "nothing runs before the folder is quiet");

  timers.runAll();
  assert.equal(refreshes, 1);

  // A later change is a new burst, not a continuation of the first.
  state.listener("rename", "qwen3.5_4B_8Bit");
  timers.runAll();
  assert.equal(refreshes, 2);
});

test("close stops refreshes and closes the watcher, and is idempotent", () => {
  const { state, watch } = fakeWatch();
  const timers = fakeTimers();
  let refreshes = 0;
  const handle = watchModels({
    modelsDir: "/models",
    refresh: () => {
      refreshes += 1;
    },
    watch,
    setTimer: timers.setTimer,
    clearTimer: timers.clearTimer,
  });

  state.listener("rename", "x");
  handle.close();
  assert.equal(timers.size, 0, "the pending refresh is cancelled");
  assert.equal(state.closed, 1);

  timers.runAll();
  state.listener("rename", "y");
  timers.runAll();
  assert.equal(refreshes, 0, "a closed watcher never refreshes again");

  handle.close();
  assert.equal(state.closed, 2, "close is idempotent, not guarded by state");
});

test("a refresh that throws is logged, not raised", () => {
  const { state, watch } = fakeWatch();
  const timers = fakeTimers();
  const messages = [];
  watchModels({
    modelsDir: "/models",
    refresh: () => {
      throw new Error("settings are read-only");
    },
    log: (message) => messages.push(message),
    watch,
    setTimer: timers.setTimer,
    clearTimer: timers.clearTimer,
  });

  state.listener("rename", "x");
  timers.runAll();
  assert.ok(
    messages.some((message) =>
      /refresh after a models\/ change failed: settings are read-only/.test(message),
    ),
    `the failure must be logged: ${JSON.stringify(messages)}`,
  );
});

test("no models directory, or an unwatchable one, is reported and not fatal", () => {
  const messages = [];
  const none = watchModels({ modelsDir: null, refresh: () => {}, log: (m) => messages.push(m) });
  assert.equal(none.watching, false);
  none.close();

  const failed = watchModels({
    modelsDir: "/models",
    refresh: () => {},
    log: (m) => messages.push(m),
    watch: () => {
      throw new Error("EMFILE");
    },
  });
  assert.equal(failed.watching, false);
  assert.match(messages.at(-1), /cannot watch \/models: EMFILE/);
  assert.match(messages.at(-1), /only refresh at boot/);
});

test("a missing refresh callback is a programming error, not a silent no-op", () => {
  assert.throws(() => watchModels({ modelsDir: "/models" }), /needs a refresh callback/);
});
