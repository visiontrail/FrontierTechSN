import {Composition, registerRoot} from 'remotion'
import {Scene} from './Scene'
import {validateSpec} from './spec.ts'

const defaultSpec = validateSpec({
  contract: 'three-d-explainer-v1', id: 'preview', width: 1280, height: 720, fps: 30,
  duration_frames: 120, background: '#FDFBF7', accent: '#1A4A3A', representation: 'illustration',
  camera: {position: [5, 4, 7], target: [0, 0, 0], end_position: [3.8, 3.2, 7.5]},
  actors: [{id: 'chip', kind: 'chip', position: [0, 0, 0], end_explode: .7}],
})
registerRoot(() => <Composition id="Explainer3D" component={Scene} defaultProps={{spec: defaultSpec}}
  durationInFrames={120} fps={30} width={1280} height={720}
  calculateMetadata={({props}) => {
    const spec = validateSpec(props.spec)
    return {width: spec.width, height: spec.height, fps: spec.fps, durationInFrames: spec.duration_frames, props: {spec}}
  }}/>)
