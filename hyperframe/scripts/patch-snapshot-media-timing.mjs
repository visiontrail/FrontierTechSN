const marker = '// FrontierTechSN: snapshot media uses composition mount offsets'

// HyperFrames 0.6.99's snapshot bundler preserves scene-local video starts.
// The renderer adds composition offsets separately, but snapshot's selector
// compares those local starts directly with the requested global timestamp.
const before = `            return Array.from(document.querySelectorAll("video[data-start]")).map((el) => {
              const v2 = el;
              const start = parseFloat(v2.dataset.start ?? "0") || 0;
              const rawRate = v2.defaultPlaybackRate;`

const after = `            return Array.from(document.querySelectorAll("video[data-start]")).map((el) => {
              const v2 = el;
              const numericTiming = (element, key, fallback) => {
                const raw = element.dataset[key] ?? fallback;
                const value = Number(raw);
                if (!Number.isFinite(value) || String(raw).trim() === "") {
                  throw new Error("Snapshot media requires resolved numeric timing: " + key + "=" + raw);
                }
                return value;
              };
              let start = numericTiming(v2, "start", "0");
              let visibleStart = start;
              let visibleEnd = Infinity;
              // Video timing is local to its composition, including nested mounts.
              // Bundled inner roots have their composition/timing attrs stripped.
              for (let parent = v2.parentElement; parent; parent = parent.parentElement) {
                if (parent.hasAttribute("data-composition-id") || parent.hasAttribute("data-composition-file") || parent.hasAttribute("data-composition-src")) {
                  const mountStart = numericTiming(parent, "start", "0");
                  start += mountStart;
                  visibleStart = Math.max(mountStart, visibleStart + mountStart);
                  visibleEnd += mountStart;
                  if (parent.dataset.end !== undefined) {
                    visibleEnd = Math.min(visibleEnd, numericTiming(parent, "end"));
                  } else if (parent.dataset.duration !== undefined) {
                    visibleEnd = Math.min(visibleEnd, mountStart + numericTiming(parent, "duration"));
                  }
                }
              }
              const rawRate = v2.defaultPlaybackRate;`

const beforeActive = '              const activeNow = t2 >= start && t2 < start + duration2 && relTime >= 0 && !!v2.id;'
const afterActive = '              const activeNow = t2 >= visibleStart && t2 < Math.min(start + duration2, visibleEnd) && relTime >= 0 && !!v2.id;'

export function patchSnapshotMediaTiming(code) {
  const markerCount = code.split(marker).length - 1
  const beforeCount = code.split(before).length - 1
  const afterCount = code.split(after).length - 1
  const beforeActiveCount = code.split(beforeActive).length - 1
  const afterActiveCount = code.split(afterActive).length - 1
  if (markerCount === 1 && beforeCount === 0 && afterCount === 1 && beforeActiveCount === 0 && afterActiveCount === 1) return code
  if (markerCount !== 0 || beforeCount !== 1 || afterCount !== 0 || beforeActiveCount !== 1 || afterActiveCount !== 0) {
    throw new Error(`HyperFrames snapshot timing patch expected 1 original anchor or 1 complete patch, found original=${beforeCount}, patched=${afterCount}, markers=${markerCount}; review the installed version before snapshotting`)
  }
  return code.replace(before, after).replace(beforeActive, afterActive)
    .replace(/^(#![^\n]*\n)?/, (prefix) => `${prefix}${marker}\n`)
}
