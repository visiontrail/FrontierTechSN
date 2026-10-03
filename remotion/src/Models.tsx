import {useMemo} from 'react'
import * as THREE from 'three'
import type {Actor, Vec3} from './spec.ts'

const metal = '#c8c7b8'
const dark = '#263d37'
const gold = '#b59a55'
type BoxProps = {p?: Vec3; s: Vec3; color?: string; rotation?: Vec3; metalness?: number}
function Box({p = [0, 0, 0], s, color = metal, rotation, metalness = .25}: BoxProps) {
  return <mesh position={p} rotation={rotation} castShadow receiveShadow>
    <boxGeometry args={s}/><meshStandardMaterial color={color} metalness={metalness} roughness={.48}/>
  </mesh>
}
function Ball({p, radius, color}: {p: Vec3; radius: number; color: string}) {
  return <mesh position={p} castShadow><sphereGeometry args={[radius, 20, 16]}/><meshStandardMaterial color={color} metalness={.2} roughness={.4}/></mesh>
}
function Cylinder({p, radius, length, color = metal, rotation}: {p: Vec3; radius: number; length: number; color?: string; rotation?: Vec3}) {
  return <mesh position={p} rotation={rotation} castShadow><cylinderGeometry args={[radius, radius, length, 24]}/><meshStandardMaterial color={color} metalness={.35} roughness={.45}/></mesh>
}
export function Curve({points, color, radius = .025}: {points: Vec3[]; color: string; radius?: number}) {
  const key = JSON.stringify(points)
  const curve = useMemo(() => new THREE.CatmullRomCurve3(points.map((v) => new THREE.Vector3(...v))), [key])
  return <mesh><tubeGeometry args={[curve, 32, radius, 8, false]}/><meshStandardMaterial color={color} roughness={.5}/></mesh>
}

function Chip({color, explode}: {color: string; explode: number}) {
  return <group>
    <Box p={[0, -.18, 0]} s={[2.3, .17, 1.8]} color={color}/>
    <Box p={[0, .02 + explode * .5, 0]} s={[1.4, .18, 1.2]} color={dark}/>
    <Box p={[0, .16 + explode * 1.05, 0]} s={[1.04, .07, .86]} color={gold} metalness={.65}/>
    <group position={[0, .31 + explode * 1.6, 0]}>
      <Box s={[1.58, .12, 1.32]}/>
      {Array.from({length: 9}, (_, i) => <Box key={i} p={[-.65 + i * .163, .2, 0]} s={[.06, .35, 1.23]} color={metal}/>)}
    </group>
    {[-1, 1].flatMap((side) => Array.from({length: 10}, (_, i) => <Box key={`${side}-${i}`} p={[-.98 + i * .217, -.24, side * .99]} s={[.08, .07, .34]} color={gold}/>))}
    {[-1, 1].flatMap((side) => Array.from({length: 7}, (_, i) => <Box key={`s${side}-${i}`} p={[side * 1.23, -.24, -.65 + i * .217]} s={[.3, .07, .08]} color={gold}/>))}
    {Array.from({length: 5}, (_, i) => <Box key={`c${i}`} p={[-.8 + .4 * i, -.08, -.73]} s={[.12, .09, .12]} color={dark}/>)}
  </group>
}

function Rack({color, explode}: {color: string; explode: number}) {
  return <group>
    <Box p={[0, -1.48, 0]} s={[1.8, .18, 1.4]} color={dark}/>
    <Box p={[0, 1.48, 0]} s={[1.8, .18, 1.4]} color={dark}/>
    {[-.82, .82].flatMap((x) => [-.61, .61].map((z) => <Box key={`${x},${z}`} p={[x, 0, z]} s={[.12, 2.9, .12]} color={metal}/>))}
    <Box p={[0, 0, -.67]} s={[1.6, 2.8, .06]} color={color}/>
    {Array.from({length: 6}, (_, i) => <group key={i} position={[0, -1.13 + i * .45, explode * (i % 2 ? .8 : .3)]}>
      <Box s={[1.54, .32, 1.15]} color={i % 2 ? color : metal}/>
      <Box p={[0, 0, .585]} s={[1.44, .26, .045]} color={dark}/>
      {[-.44, -.05, .34].map((x) => <group key={x} position={[x, 0, .62]}>
        <Cylinder p={[0, 0, 0]} radius={.1} length={.025} color={metal} rotation={[Math.PI / 2, 0, 0]}/>
        <Cylinder p={[0, 0, .015]} radius={.055} length={.03} color={dark} rotation={[Math.PI / 2, 0, 0]}/>
      </group>)}
      <Ball p={[.64, .04, .63]} radius={.025} color={'#d4b356'}/>
    </group>)}
  </group>
}

function SolarWing({side, color, explode}: {side: number; color: string; explode: number}) {
  return <group position={[side * (1.78 + explode * .6), 0, 0]}>
    <Box s={[2.25, .09, 1.38]} color={metal}/>
    {Array.from({length: 6}, (_, x) => Array.from({length: 3}, (_, z) => <Box key={`${x}-${z}`} p={[-.93 + x * .374, .052, -.44 + z * .44]} s={[.34, .035, .4]} color={color}/>))}
    <Cylinder p={[-side * 1.15, 0, 0]} radius={.045} length={.75 + explode * 1.2} rotation={[0, 0, Math.PI / 2]}/>
  </group>
}
function Satellite({color, explode}: {color: string; explode: number}) {
  return <group>
    <Box s={[1.05, 1.05, 1.1]} color={gold}/>
    <Box p={[0, .58 + explode * .5, 0]} s={[1.1, .09, 1.15]} color={metal}/>
    <Box p={[0, -.58 - explode * .5, 0]} s={[1.1, .09, 1.15]} color={metal}/>
    <SolarWing side={-1} color={color} explode={explode}/><SolarWing side={1} color={color} explode={explode}/>
    <Cylinder p={[0, .88, 0]} radius={.075} length={.62}/>
    <group position={[0, 1.28, 0]} rotation={[.35, 0, .35]}>
      <mesh rotation={[-Math.PI / 2, 0, 0]} castShadow><sphereGeometry args={[.5, 32, 16, 0, Math.PI * 2, 0, .95]}/><meshStandardMaterial color={metal} metalness={.6} roughness={.3} side={THREE.DoubleSide}/></mesh>
      <Cylinder p={[0, .15, 0]} radius={.025} length={.45} color={dark}/>
      <Ball p={[0, .4, 0]} radius={.065} color={gold}/>
    </group>
    {Array.from({length: 5}, (_, i) => <Box key={i} p={[-.42 + i * .21, 0, .57]} s={[.11, .7, .07]} color={metal}/>)}
  </group>
}

function Robot({color, progress}: {color: string; progress: number}) {
  return <group>
    <Box p={[0, .4, 0]} s={[.85, .94, .52]} color={color}/>
    <Cylinder p={[0, 1, 0]} radius={.13} length={.2}/>
    <group position={[0, 1.26, 0]} rotation={[0, progress * -.25, 0]}>
      <Box s={[.66, .45, .48]} color={metal}/>
      <Box p={[0, .03, .25]} s={[.51, .15, .035]} color={dark}/>
      {[-.15, .15].map((x) => <Ball key={x} p={[x, .035, .285]} radius={.045} color={'#d4b356'}/>)}
    </group>
    {[-1, 1].map((side) => <group key={side}>
      <Ball p={[side * .57, .72, 0]} radius={.16} color={metal}/>
      <group position={[side * .57, .72, 0]} rotation={[side > 0 ? -.6 * progress : .1, 0, side * (.12 + (side > 0 ? progress * .45 : 0))]}>
        <Cylinder p={[0, -.32, 0]} radius={.105} length={.58} color={color}/>
        <Ball p={[0, -.64, 0]} radius={.14} color={metal}/>
        <group position={[0, -.64, 0]} rotation={[-.3 - progress * .9, 0, 0]}>
          <Cylinder p={[0, -.25, 0]} radius={.09} length={.46} color={color}/>
          <Box p={[0, -.52, 0]} s={[.19, .21, .15]} color={metal}/>
        </group>
      </group>
      <Ball p={[side * .25, -.14, 0]} radius={.16} color={metal}/>
      <Cylinder p={[side * .25, -.48, 0]} radius={.13} length={.56} color={color}/>
      <Ball p={[side * .25, -.8, 0]} radius={.145} color={metal}/>
      <Cylinder p={[side * .25, -1.1, 0]} radius={.11} length={.52} color={color}/>
      <Box p={[side * .25, -1.4, .13]} s={[.31, .16, .5]} color={dark}/>
    </group>)}
  </group>
}

function Data({color, phase}: {color: string; phase: number}) {
  return <group>
    <mesh rotation={[Math.PI / 2, 0, 0]}><torusGeometry args={[.85, .025, 8, 64]}/><meshStandardMaterial color={metal}/></mesh>
    <mesh><icosahedronGeometry args={[.34, 1]}/><meshStandardMaterial color={color} metalness={.35} roughness={.35}/></mesh>
    {Array.from({length: 7}, (_, i) => {
      const a = i / 7 * Math.PI * 2 + phase * .65
      return <Ball key={i} p={[Math.cos(a) * .85, Math.sin(a * 2) * .22, Math.sin(a) * .85]} radius={.085} color={i % 2 ? color : gold}/>
    })}
  </group>
}
export function Model({actor, color, explode, progress, phase}: {actor: Actor; color: string; explode: number; progress: number; phase: number}) {
  if (actor.kind === 'chip') return <Chip color={color} explode={explode}/>
  if (actor.kind === 'rack') return <Rack color={color} explode={explode}/>
  if (actor.kind === 'satellite') return <Satellite color={color} explode={explode}/>
  if (actor.kind === 'robot') return <Robot color={color} progress={progress}/>
  if (actor.kind === 'data') return <Data color={color} phase={phase}/>
  return <mesh castShadow receiveShadow>
    {actor.primitive === 'sphere' ? <sphereGeometry args={[.7, 32, 24]}/> : actor.primitive === 'cylinder' ? <cylinderGeometry args={[.65, .65, 1.2, 32]}/> : actor.primitive === 'torus' ? <torusGeometry args={[.65, .2, 16, 40]}/> : <boxGeometry args={[1.2, 1.2, 1.2]}/>}
    <meshStandardMaterial color={color} metalness={.25} roughness={.4}/>
  </mesh>
}
