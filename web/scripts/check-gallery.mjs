import { readFile, readdir, lstat } from 'node:fs/promises'
import { join } from 'node:path'

const root = process.argv[2] ?? 'public'
const directory = join(root, 'gallery')
const manifest = JSON.parse(await readFile(join(directory, 'manifest.json'), 'utf8'))
const images = new Set()
const fields = ['id', 'title', 'theme', 'palette', 'composition', 'style', 'prompt', 'seed',
  'model', 'date', 'generated_at', 'approved_at', 'steps', 'width', 'height', 'image', 'thumb']
if (manifest.version !== 1 || !Array.isArray(manifest.artworks)) throw Error('Invalid gallery format')
for (const entry of manifest.artworks) {
  if (!fields.every((key) => Object.hasOwn(entry, key))
      || !/^[a-f0-9]{32}-(0[1-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-6])$/.test(entry.id)
      || !/^\d{4}-\d{2}-\d{2}$/.test(entry.date)
      || Number.isNaN(Date.parse(entry.approved_at)) || typeof entry.model !== 'string'
      || entry.mock || entry.model.startsWith('mock/') || images.has(entry.image)
      || entry.image !== `gallery/art/${entry.id}.png`
      || entry.thumb !== `gallery/thumbs/${entry.id}.webp`) throw Error('Invalid published artwork')
  images.add(entry.image)
  images.add(entry.thumb)
}
for (const file of images) {
  const stat = await lstat(join(root, file))
  if (!stat.isFile() || stat.size === 0) throw Error(`Invalid gallery file ${file}`)
}
for (const item of await readdir(directory, { withFileTypes: true })) {
  if (item.name === 'manifest.json' && item.isFile()) continue
  if (!item.isDirectory() || !['art', 'thumbs'].includes(item.name)) throw Error(`Unexpected gallery file ${item.name}`)
  for (const file of await readdir(join(directory, item.name))) {
    if (!images.has(`gallery/${item.name}/${file}`)) throw Error(`Unreferenced gallery file ${file}`)
  }
}
if (root === 'dist') {
  for (const item of await readdir(root, { withFileTypes: true })) {
    if (!['gallery', 'index.html', 'assets'].includes(item.name)) throw Error(`Unexpected build output ${item.name}`)
  }
}
console.log(`Gallery verified (${manifest.artworks.length} approved works)`)
