export type Artwork = {
  id: string
  title: string
  theme: string
  palette: string
  composition: string
  style: string
  prompt: string
  seed: number
  model: string
  date: string
  approved_at: string
  steps: number
  width: number
  height: number
  image: string
  thumb: string
  mock?: boolean
}

export const asset = (file: string) => import.meta.env.BASE_URL + file
const validId = /^[a-f0-9]{32}-(0[1-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-6])$/
const fields = ['title', 'theme', 'palette', 'composition', 'style', 'prompt', 'model', 'approved_at'] as const
const numeric = ['seed', 'steps', 'width', 'height'] as const

export async function loadGallery(signal: AbortSignal): Promise<Artwork[]> {
  const response = await fetch(asset('gallery/manifest.json'), { signal, cache: 'no-cache' })
  if (!response.ok) throw new Error('Gallery manifest unavailable')
  const value = await response.json()
  if (value?.version !== 1 || !Array.isArray(value?.artworks)) throw new Error('Invalid gallery manifest')
  const ids = new Set<string>()
  for (const item of value.artworks) {
    if (!item || typeof item.id !== 'string' || !validId.test(item.id) || ids.has(item.id)
        || !fields.every((field) => typeof item[field] === 'string' && item[field].length)
        || !numeric.every((field) => Number.isSafeInteger(item[field]) && item[field] >= 0)
        || typeof item.date !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(item.date)
        || item.image !== `gallery/art/${item.id}.png` || item.thumb !== `gallery/thumbs/${item.id}.webp`
        || (import.meta.env.PROD && (item.mock || item.model.startsWith('mock/')))) {
      throw new Error('Invalid artwork in gallery manifest')
    }
    ids.add(item.id)
  }
  return value.artworks
}

export function readableDate(date: string, monthOnly = false) {
  return new Intl.DateTimeFormat('en', {
    year: 'numeric', month: 'long', ...(monthOnly ? {} : { day: 'numeric' }), timeZone: 'UTC',
  }).format(new Date(monthOnly ? `${date}-01T00:00:00Z` : `${date}T00:00:00Z`))
}

export function readFavorites(): string[] {
  try {
    const list: unknown = JSON.parse(localStorage.getItem('museum-favorites-v1') ?? '[]')
    return Array.isArray(list) ? list.filter((id): id is string => typeof id === 'string' && validId.test(id)) : []
  } catch {
    return []
  }
}
