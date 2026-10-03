import path from 'node:path'
import {fileURLToPath} from 'node:url'
import {renderScene} from './render.mjs'
const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)))
const output = path.resolve(process.argv[2] ?? path.join(root, '../outputs/three-d-benchmark/sample.mp4'))
const report = await renderScene({spec: path.join(root, 'examples/compute-flow.json'), output, verify: true, stills: '0,45,89'})
process.stdout.write(`${JSON.stringify(report, null, 2)}\n`)
