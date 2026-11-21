/**
 * Keep the route in step with `models/` while the harness is running.
 *
 * The route is written at boot, which is enough for a folder that does not
 * change during a session. Installing a model is a long download the person
 * starts and then wants to use, though, and deleting one is the other direction:
 * without this the picker keeps showing the folder as it was at boot, offering a
 * model that is gone and hiding one that is now there.
 *
 * A refresh is cheap (one process, one file write, and `settings.yaml` is
 * hot-reloaded) but an install is not one event -- thousands of files land in
 * the directory -- so changes are coalesced into one refresh per quiet period.
 *
 * The watcher is deliberately unref'd and non-persistent: it must never be the
 * reason the harness cannot exit. Nothing here throws outward; a platform where
 * watching fails falls back to the boot-time refresh with a log line.
 *
 * @module dsh-emone/models-watch
 */
import { watch as watchFileSystem } from "node:fs";

/** How long the folder has to be quiet before the route is rebuilt. */
export const DEFAULT_DEBOUNCE_MS = 2000;

function describe(error) {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Watch a models directory and refresh once per burst of changes.
 *
 * @param options - `modelsDir`, `refresh` (a zero-argument callback), plus
 *   injectable `log`, `debounceMs`, `watch` and timers for tests.
 * @returns `{ watching, close() }` — `close` is idempotent and is what a
 *   disposal hook calls.
 */
export function watchModels({
  modelsDir,
  refresh,
  log = () => {},
  debounceMs = DEFAULT_DEBOUNCE_MS,
  watch = watchFileSystem,
  setTimer = setTimeout,
  clearTimer = clearTimeout,
} = {}) {
  if (typeof modelsDir !== "string" || modelsDir === "") {
    return { watching: false, close() {} };
  }
  if (typeof refresh !== "function") {
    throw new Error("dsh-emone: watchModels needs a refresh callback");
  }

  let timer = null;
  let closed = false;

  const fire = () => {
    timer = null;
    if (closed) return;
    try {
      refresh();
    } catch (error) {
      log(`dsh-emone: refresh after a models/ change failed: ${describe(error)}`);
    }
  };

  const schedule = () => {
    if (closed) return;
    // Restart the quiet period: an install emits events for minutes.
    if (timer !== null) clearTimer(timer);
    timer = setTimer(fire, debounceMs);
    if (typeof timer?.unref === "function") timer.unref();
  };

  let watcher;
  try {
    watcher = watch(modelsDir, { persistent: false }, schedule);
  } catch (error) {
    log(
      `dsh-emone: cannot watch ${modelsDir}: ${describe(error)}; ` +
        "the route will only refresh at boot",
    );
    return { watching: false, close() {} };
  }
  if (typeof watcher?.unref === "function") watcher.unref();
  log(`dsh-emone: watching ${modelsDir} so installs and removals reach the picker`);

  return {
    watching: true,
    close() {
      closed = true;
      if (timer !== null) {
        clearTimer(timer);
        timer = null;
      }
      try {
        watcher.close();
      } catch {
        // Already closed, or the platform closed it for us.
      }
    },
  };
}
