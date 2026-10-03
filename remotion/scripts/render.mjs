import {bundle} from '@remotion/bundler'
import {selectComposition, renderMedia, renderStill} from '@remotion/renderer'
import {createHash, randomUUID} from 'node:crypto'
import {execFileSync} from 'node:child_process'
import fs from 'node:fs/promises'
import path from 'node:path'
import {fileURLToPath} from 'node:url'
import {validateSpec} from '../src/spec.ts'

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)))
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex')
export function parseArgs(argv) {
  const args = {}
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i]
    if (key === '--verify-determinism') {args.verify = true; continue}
    if (!['--spec', '--output', '--stills', '--browser', '--concurrency'].includes(key) || !argv[i + 1] || argv[i + 1].startsWith('--')) throw new Error(`Invalid argument: ${key}`)
    args[key.slice(2)] = argv[++i]
  }
  if (!args.spec || !args.output) throw new Error('Usage: node remotion/scripts/render.mjs --spec scene.json --output clip.mp4 [--stills 0,45,89] [--browser PATH] [--verify-determinism]')
  return args
}
function inspectImage(filename) {
  const gray = execFileSync('ffmpeg', ['-v', 'error', '-i', filename, '-vf', 'scale=64:64,format=gray', '-frames:v', '1', '-f', 'rawvideo', '-'], {maxBuffer: 1024 * 1024})
  const values = new Set(gray)
  const range = Math.max(...gray) - Math.min(...gray)
  if (gray.length !== 4096 || values.size < 16 || range < 20) throw new Error(`Blank/low-detail WebGL frame: ${filename}`)
  return {luminance_range: range, distinct_luminance: values.size}
}
export async function renderScene(args) {
  const started = performance.now()
  const raw = await fs.readFile(path.resolve(args.spec))
  const spec = validateSpec(JSON.parse(raw))
  const output = path.resolve(args.output)
  if (path.extname(output).toLowerCase() !== '.mp4') throw new Error('output must end in .mp4')
  const requested = args.stills === undefined ? [0, Math.floor((spec.duration_frames - 1) / 2), spec.duration_frames - 1] : args.stills.split(',').map(Number)
  if (!requested.length || requested.length > 12 || requested.some((n) => !Number.isInteger(n) || n < 0 || n >= spec.duration_frames)) throw new Error('stills: expected 1–12 frame indices within the composition')
  const frames = [...new Set(requested)]
  const concurrency = Number(args.concurrency ?? 2)
  if (!Number.isInteger(concurrency) || concurrency < 1 || concurrency > 4) throw new Error('concurrency: expected 1–4')
  const explicitBrowser = args.browser || process.env.REMOTION_BROWSER_EXECUTABLE
  const browserExecutable = explicitBrowser ? path.resolve(explicitBrowser) : JSON.parse(execFileSync(process.execPath,
    [path.join(root, 'scripts/ensure-browser.mjs')], {cwd: root, encoding: 'utf8', maxBuffer: 1024 * 1024, timeout: 600000})).path
  const chromiumOptions = {gl: 'angle'}
  const inputProps = {spec}
  const reportPath = `${output}.render.json`
  const stillDir = path.join(path.dirname(output), `${path.basename(output, '.mp4')}.stills`)
  await fs.mkdir(path.dirname(output), {recursive: true})
  await fs.mkdir(stillDir, {recursive: true})
  await fs.mkdir(path.join(root, '.bundle'), {recursive: true})
  const bundleDir = await fs.mkdtemp(path.join(root, '.bundle', 'render-'))
  const candidate = path.join(path.dirname(output), `.${path.basename(output, '.mp4')}.${randomUUID()}.mp4`)
  const report = {contract: 'three-d-render-v1', passed: false, input_sha256: hash(raw), output,
    engine: {remotion: '4.0.532', three: '0.186.1', react_three_fiber: '9.8.1', gl: 'angle'}, stills: []}
  try {
    const serveUrl = await bundle({entryPoint: path.join(root, 'src/index.tsx'), outDir: bundleDir})
    const options = {serveUrl, inputProps, browserExecutable, chromiumOptions}
    const composition = await selectComposition({...options, id: 'Explainer3D'})
    let last = -1
    await renderMedia({...options, composition, codec: 'h264', crf: 19, pixelFormat: 'yuv420p', outputLocation: candidate, concurrency,
      onProgress: ({progress}) => {const p = Math.floor(progress * 10); if (p !== last) {last = p; process.stderr.write(`3D render ${p * 10}%\n`)}}})
    const probe = JSON.parse(execFileSync('ffprobe', ['-v', 'error', '-count_frames', '-show_streams', '-of', 'json', candidate], {encoding: 'utf8', maxBuffer: 1024 * 1024}))
    const videos = probe.streams.filter((s) => s.codec_type === 'video')
    const video = videos[0]
    const [num, den] = String(video?.avg_frame_rate ?? '').split('/').map(Number)
    if (videos.length !== 1 || probe.streams.some((s) => s.codec_type === 'audio') || video.width !== spec.width || video.height !== spec.height || Number(video.nb_read_frames) !== spec.duration_frames || num / den !== spec.fps) throw new Error('Rendered video does not match dimensions, cadence, frame count, or silent contract')
    execFileSync('ffmpeg', ['-v', 'error', '-i', candidate, '-f', 'null', '-'], {maxBuffer: 1024 * 1024})
    for (const frame of frames) {
      const filename = path.join(stillDir, `frame-${String(frame).padStart(6, '0')}.png`)
      await renderStill({...options, composition, frame, imageFormat: 'png', output: filename})
      report.stills.push({frame, path: filename, sha256: hash(await fs.readFile(filename)), ...inspectImage(filename)})
    }
    if (args.verify) {
      const reference = report.stills[Math.floor(report.stills.length / 2)]
      const repeated = path.join(stillDir, '.repeat.png')
      // The last requested frame was rendered first; this deliberately seeks backward.
      await renderStill({...options, composition, frame: reference.frame, imageFormat: 'png', output: repeated})
      const repeatedHash = hash(await fs.readFile(repeated))
      if (repeatedHash !== reference.sha256) throw new Error(`Non-deterministic frame ${reference.frame}`)
      report.determinism = {passed: true, frame: reference.frame, sha256: repeatedHash, order: [...frames, reference.frame]}
      await fs.unlink(repeated)
    }
    await fs.rename(candidate, output)
    Object.assign(report, {passed: true, frame_count: spec.duration_frames, width: spec.width, height: spec.height, fps: spec.fps,
      duration_seconds: spec.duration_frames / spec.fps, output_sha256: hash(await fs.readFile(output)), render_seconds: (performance.now() - started) / 1000})
    await fs.writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`)
    return report
  } catch (error) {
    report.error = error instanceof Error ? error.message : String(error)
    report.render_seconds = (performance.now() - started) / 1000
    await fs.writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`)
    throw error
  } finally {
    await fs.rm(bundleDir, {recursive: true, force: true})
    await fs.rm(candidate, {force: true})
  }
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {const report = await renderScene(parseArgs(process.argv.slice(2))); process.stdout.write(`${JSON.stringify(report)}\n`)}
  catch (error) {process.stderr.write(`${error.stack ?? error}\n`); process.exitCode = 1}
}
