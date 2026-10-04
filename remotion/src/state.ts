import type {Actor, SceneSpec, Vec3} from './spec.ts'

export const clamp = (v: number) => Math.max(0, Math.min(1, v))
export const smooth = (v: number) => {const p = clamp(v); return p * p * (3 - 2 * p)}
export const mix = (a: number, b: number, p: number) => a + (b - a) * p
export const mix3 = (a: Vec3, b: Vec3, p: number): Vec3 => a.map((v, i) => mix(v, b[i], p)) as Vec3
export function actorState(actor: Actor, frame: number, durationFrames: number) {
  const t = clamp(frame / (durationFrames - 1))
  const start = actor.motion_start ?? .12; const end = actor.motion_end ?? .72
  const progress = actor.action === 'hold' ? 0 : smooth((t - start) / (end - start))
  const rotation = actor.rotation ?? [0, 0, 0]
  // Explicit reveals stage later subjects inside the declared motion window.
  // An omitted action retains the recognizable entrance used by older plans.
  const reveal = actor.action === 'reveal' ? progress
    : actor.action === undefined ? mix(.86, 1, smooth(t / Math.min(end, Math.max(.08, start)))) : 1
  return {
    position: mix3(actor.position, actor.end_position ?? actor.position, progress),
    rotation: mix3(rotation, actor.end_rotation ?? rotation, progress),
    scale: (actor.scale ?? [1, 1, 1]).map((v) => v * reveal) as Vec3,
    visible: reveal > 0,
    explode: mix(actor.explode ?? 0, actor.end_explode ?? actor.explode ?? 0, progress),
    progress,
  }
}
export const linkVisible = (from: ReturnType<typeof actorState>, to: ReturnType<typeof actorState>) => from.visible && to.visible
export function sceneState(spec: SceneSpec, frame: number) {
  const f = Math.min(spec.duration_frames - 1, Math.max(0, frame))
  const settle = Math.max(...spec.actors.map((a) => a.motion_end ?? .72))
  const progress = smooth(clamp(f / (spec.duration_frames - 1)) / settle)
  return {
    frame: f,
    seconds: f / spec.fps,
    camera: {
      position: mix3(spec.camera.position, spec.camera.end_position ?? spec.camera.position, progress),
      target: mix3(spec.camera.target, spec.camera.end_target ?? spec.camera.target, progress),
      fov: spec.camera.fov ?? 40,
    },
    actors: spec.actors.map((actor) => ({id: actor.id, ...actorState(actor, f, spec.duration_frames)})),
    // The planner records this separate 3D settle cue for preflight and review.
    flowPhase: Math.min(f / spec.fps, (spec.duration_frames - 1) / spec.fps * settle),
  }
}
