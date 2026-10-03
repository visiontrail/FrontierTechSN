export type Vec3 = [number, number, number]
export type ActorKind = 'chip' | 'rack' | 'satellite' | 'robot' | 'data' | 'object'
export type Actor = {
  id: string
  kind: ActorKind
  primitive?: 'box' | 'sphere' | 'cylinder' | 'torus'
  position: Vec3
  end_position?: Vec3
  rotation?: Vec3
  end_rotation?: Vec3
  scale?: Vec3
  color?: string
  explode?: number
  end_explode?: number
  motion_start?: number
  motion_end?: number
  action?: 'reveal' | 'translate' | 'rotate' | 'explode' | 'hold'
}
export type SceneSpec = {
  contract: 'three-d-explainer-v1'
  id: string
  width: number
  height: number
  fps: number
  duration_frames: number
  background: string
  accent: string
  camera: {position: Vec3; target: Vec3; end_position?: Vec3; end_target?: Vec3; fov?: number}
  actors: Actor[]
  links?: {from: string; to: string; color?: string; particles?: number}[]
  copy?: string[]
  narration_excerpt?: string
  semantic_intent?: string
  representation: 'illustration'
}

const object = (value: unknown, at: string): Record<string, unknown> => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${at}: expected object`)
  return value as Record<string, unknown>
}
const keys = (value: Record<string, unknown>, allowed: string[], at: string) => {
  for (const key of Object.keys(value)) if (!allowed.includes(key)) throw new Error(`${at}.${key}: unsupported field`)
}
const number = (v: unknown, lo: number, hi: number, at: string, integer = false): number => {
  if (typeof v !== 'number' || !Number.isFinite(v) || v < lo || v > hi || (integer && !Number.isInteger(v))) throw new Error(`${at}: expected ${integer ? 'integer' : 'finite number'} in [${lo}, ${hi}]`)
  return v
}
const vector = (v: unknown, lo: number, hi: number, at: string): Vec3 => {
  if (!Array.isArray(v) || v.length !== 3) throw new Error(`${at}: expected three coordinates`)
  return v.map((n, i) => number(n, lo, hi, `${at}[${i}]`)) as Vec3
}
const color = (v: unknown, at: string) => {
  if (typeof v !== 'string' || !/^#[0-9a-fA-F]{6}$/.test(v)) throw new Error(`${at}: expected #RRGGBB`)
}
const id = (v: unknown, at: string) => {
  if (typeof v !== 'string' || !/^[a-zA-Z][a-zA-Z0-9_-]{0,47}$/.test(v)) throw new Error(`${at}: invalid identifier`)
}
const text = (v: unknown, max: number, at: string) => {
  if (typeof v !== 'string' || v.length > max) throw new Error(`${at}: expected text up to ${max} characters`)
}

// This is also the CLI boundary. Model output never becomes executable code or an asset URL.
export function validateSpec(input: unknown): SceneSpec {
  const s = object(input, 'scene')
  keys(s, ['contract', 'id', 'width', 'height', 'fps', 'duration_frames', 'background', 'accent', 'camera', 'actors', 'links', 'copy', 'narration_excerpt', 'semantic_intent', 'representation'], 'scene')
  if (s.contract !== 'three-d-explainer-v1' || s.representation !== 'illustration') throw new Error('scene: unsupported contract or representation')
  id(s.id, 'scene.id')
  for (const dim of ['width', 'height']) if (number(s[dim], 320, 3840, dim, true) % 2) throw new Error(`${dim}: must be even`)
  const fps = number(s.fps, 24, 60, 'fps', true)
  const frames = number(s.duration_frames, 2, 3600, 'duration_frames', true)
  if (frames / fps > 60) throw new Error('duration: maximum 60 seconds')
  color(s.background, 'background'); color(s.accent, 'accent')
  const c = object(s.camera, 'camera')
  keys(c, ['position', 'target', 'end_position', 'end_target', 'fov'], 'camera')
  const cp = vector(c.position, -30, 30, 'camera.position')
  const ct = vector(c.target, -12, 12, 'camera.target')
  const ep = c.end_position === undefined ? cp : vector(c.end_position, -30, 30, 'camera.end_position')
  const et = c.end_target === undefined ? ct : vector(c.end_target, -12, 12, 'camera.end_target')
  if (c.fov !== undefined) number(c.fov, 20, 75, 'camera.fov')
  // Camera and target interpolate together: reject any path passing through its target.
  const r = cp.map((v, i) => v - ct[i]); const d = ep.map((v, i) => v - et[i] - r[i])
  const dd = d.reduce((a, v) => a + v * v, 0)
  const p = dd ? Math.min(1, Math.max(0, -r.reduce((a, v, i) => a + v * d[i], 0) / dd)) : 0
  if (Math.hypot(...r.map((v, i) => v + p * d[i])) < 2) throw new Error('camera: path must remain at least 2 units from target')
  if (!Array.isArray(s.actors) || s.actors.length < 1 || s.actors.length > 12) throw new Error('actors: expected 1–12 actors')
  const ids = new Set<string>()
  for (const [i, raw] of s.actors.entries()) {
    const a = object(raw, `actors[${i}]`); const at = `actors[${i}]`
    keys(a, ['id', 'kind', 'primitive', 'position', 'end_position', 'rotation', 'end_rotation', 'scale', 'color', 'explode', 'end_explode', 'motion_start', 'motion_end', 'action'], at)
    id(a.id, `${at}.id`)
    if (ids.has(a.id as string)) throw new Error(`${at}: duplicate id`)
    ids.add(a.id as string)
    if (!['chip', 'rack', 'satellite', 'robot', 'data', 'object'].includes(a.kind as string)) throw new Error(`${at}: invalid kind`)
    vector(a.position, -12, 12, `${at}.position`)
    if (a.end_position !== undefined) vector(a.end_position, -12, 12, `${at}.end_position`)
    if (a.scale !== undefined) vector(a.scale, .1, 4, `${at}.scale`)
    for (const k of ['rotation', 'end_rotation']) if (a[k] !== undefined) vector(a[k], -4 * Math.PI, 4 * Math.PI, `${at}.${k}`)
    if (a.color !== undefined) color(a.color, `${at}.color`)
    if (a.primitive !== undefined && (a.kind !== 'object' || !['box', 'sphere', 'cylinder', 'torus'].includes(a.primitive as string))) throw new Error(`${at}: invalid primitive`)
    for (const k of ['explode', 'end_explode']) if (a[k] !== undefined) {
      number(a[k], 0, 1, `${at}.${k}`)
      if (!['chip', 'rack', 'satellite'].includes(a.kind as string)) throw new Error(`${at}: this kind has no exploded view`)
    }
    const start = a.motion_start === undefined ? .12 : number(a.motion_start, 0, 1, `${at}.motion_start`)
    const end = a.motion_end === undefined ? .72 : number(a.motion_end, 0, 1, `${at}.motion_end`)
    if (start >= end) throw new Error(`${at}: motion_start must precede motion_end`)
    if (a.action !== undefined && !['reveal', 'translate', 'rotate', 'explode', 'hold'].includes(a.action as string)) throw new Error(`${at}: invalid action`)
  }
  if (s.links !== undefined) {
    if (!Array.isArray(s.links) || s.links.length > 24) throw new Error('links: expected at most 24 links')
    for (const [i, raw] of s.links.entries()) {
      const l = object(raw, `links[${i}]`)
      keys(l, ['from', 'to', 'color', 'particles'], `links[${i}]`)
      if (!ids.has(l.from as string) || !ids.has(l.to as string) || l.from === l.to) throw new Error(`links[${i}]: invalid actor references`)
      if (l.color !== undefined) color(l.color, `links[${i}].color`)
      if (l.particles !== undefined) number(l.particles, 0, 12, `links[${i}].particles`, true)
    }
  }
  if (s.copy !== undefined) {
    if (!Array.isArray(s.copy) || s.copy.length > 12) throw new Error('copy: expected at most 12 strings')
    s.copy.forEach((v, i) => text(v, 500, `copy[${i}]`))
  }
  if (s.narration_excerpt !== undefined) text(s.narration_excerpt, 10000, 'narration_excerpt')
  if (s.semantic_intent !== undefined) text(s.semantic_intent, 2000, 'semantic_intent')
  return structuredClone(input) as SceneSpec
}
