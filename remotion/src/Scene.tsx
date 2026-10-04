import {useLayoutEffect} from 'react'
import {AbsoluteFill, useCurrentFrame} from 'remotion'
import {ThreeCanvas} from '@remotion/three'
import {useThree} from '@react-three/fiber'
import * as THREE from 'three'
import {Curve, Model} from './Models'
import {linkVisible, sceneState} from './state.ts'
import type {SceneSpec, Vec3} from './spec.ts'

function Camera({state}: {state: ReturnType<typeof sceneState>['camera']}) {
  const {camera, invalidate} = useThree()
  useLayoutEffect(() => {
    camera.position.set(...state.position)
    camera.lookAt(...state.target)
    if (camera instanceof THREE.PerspectiveCamera) {camera.fov = state.fov; camera.updateProjectionMatrix()}
    camera.updateMatrixWorld()
    invalidate()
  }, [camera, invalidate, ...state.position, ...state.target, state.fov])
  return null
}
function Link({from, to, color, particles, phase}: {from: Vec3; to: Vec3; color: string; particles: number; phase: number}) {
  const mid: Vec3 = [(from[0] + to[0]) / 2, Math.max(from[1], to[1]) + .45, (from[2] + to[2]) / 2]
  const curve = new THREE.CatmullRomCurve3([new THREE.Vector3(...from), new THREE.Vector3(...mid), new THREE.Vector3(...to)])
  const end = curve.getPoint(.94); const tangent = curve.getTangent(.94)
  const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), tangent)
  return <group>
    <Curve points={[from, mid, to]} color={color} radius={.022}/>
    <mesh position={end} quaternion={q}><coneGeometry args={[.09, .23, 16]}/><meshStandardMaterial color={color}/></mesh>
    {Array.from({length: particles}, (_, i) => {
      const p = curve.getPoint((i / particles + phase * .22) % 1)
      return <mesh key={i} position={p}><sphereGeometry args={[.045, 12, 10]}/><meshStandardMaterial color={'#c1a35d'} emissive={'#c1a35d'} emissiveIntensity={.15}/></mesh>
    })}
  </group>
}
export function Scene({spec}: {spec: SceneSpec}) {
  const frame = useCurrentFrame()
  const state = sceneState(spec, frame)
  return <AbsoluteFill style={{backgroundColor: spec.background}}>
    <ThreeCanvas width={spec.width} height={spec.height} dpr={1} shadows={{type: THREE.PCFShadowMap}}
      camera={{position: state.camera.position, fov: state.camera.fov, near: .1, far: 150}}
      gl={{antialias: true, alpha: false, preserveDrawingBuffer: true}}>
      <color attach="background" args={[spec.background]}/>
      <Camera state={state.camera}/>
      <ambientLight intensity={1.2}/>
      <hemisphereLight args={['#ffffff', '#858779', 1.1]}/>
      <directionalLight position={[5, 8, 6]} intensity={2.7} castShadow shadow-mapSize={[1024, 1024]} shadow-camera-left={-12} shadow-camera-right={12} shadow-camera-top={12} shadow-camera-bottom={-12} shadow-bias={-.0005}/>
      <directionalLight position={[-6, 4, -5]} intensity={1.8} color={'#f6e8c8'}/>
      {spec.actors.map((actor, i) => {
        const a = state.actors[i]
        return <group key={actor.id} position={a.position} rotation={a.rotation} scale={a.scale} visible={a.visible}>
          <Model actor={actor} color={actor.color ?? spec.accent} explode={a.explode} progress={a.progress} phase={state.flowPhase}/>
        </group>
      })}
      {(spec.links ?? []).map((link, i) => {
        const from = state.actors.find((a) => a.id === link.from)!
        const to = state.actors.find((a) => a.id === link.to)!
        return linkVisible(from, to) ? <Link key={i} from={from.position} to={to.position} color={link.color ?? spec.accent} particles={link.particles ?? 5} phase={state.flowPhase}/> : null
      })}
      <mesh position={[0, -2, 0]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow><planeGeometry args={[200, 200]}/><meshStandardMaterial color={spec.background} roughness={1}/></mesh>
    </ThreeCanvas>
  </AbsoluteFill>
}
