import type { ExplainerProject, ExplainerStyle } from './explainerApi'

const COLOR_STYLE_FIELDS = ['background_color', 'ink_color', 'accent_color'] as const

export interface ExplainerStylePreviewCandidate {
  imageUrl: string
  projectId: string
  projectTitle: string
  sceneId: string
  isCurrentProject: boolean
}

export function sameExplainerVisualStyle(left: ExplainerStyle, right: ExplainerStyle): boolean {
  return COLOR_STYLE_FIELDS.every((field) => left[field].toLowerCase() === right[field].toLowerCase())
    && left.character_description === right.character_description
    && left.visual_direction === right.visual_direction
}

function imageUrl(projectId: string, relative: string | undefined): string | null {
  if (!relative || /^[a-z][a-z\d+.-]*:/i.test(relative) || relative.includes('\\')) return null
  if (Array.from(relative).some((char) => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127)) return null
  const parts = relative.split('/')
  if (parts.some((part) => !part || part === '.' || part === '..') || !/\.(png|jpe?g|webp)$/i.test(relative)) return null
  return `/api/explainers/${encodeURIComponent(projectId)}/assets/${parts.map(encodeURIComponent).join('/')}`
}

export function selectStylePreviews(
  style: ExplainerStyle,
  currentProject?: ExplainerProject,
  library: ExplainerProject[] = [],
  currentProjectId = currentProject?.id,
): ExplainerStylePreviewCandidate[] {
  // A refreshed project is authoritative over an older copy in the library.
  const projects = [
    ...(currentProject ? [currentProject] : []),
    ...library.filter((project) => project.id !== currentProjectId && project.id !== currentProject?.id),
  ]
  const candidates: ExplainerStylePreviewCandidate[] = []
  const seen = new Set<string>()
  for (const project of projects) {
    if (project.status !== 'completed' || !sameExplainerVisualStyle(style, project.style)) continue
    // Top-level scene_artifacts may survive an edit. Only completed delivery
    // artifacts belong to the saved style shown by this project.
    for (const scene of project.artifacts.scene_artifacts ?? []) {
      const url = imageUrl(project.id, scene.image_path)
      if (!url || seen.has(url)) continue
      seen.add(url)
      candidates.push({
        imageUrl: url,
        projectId: project.id,
        projectTitle: project.title,
        sceneId: scene.id,
        isCurrentProject: project.id === currentProject?.id,
      })
    }
  }
  return candidates
}
