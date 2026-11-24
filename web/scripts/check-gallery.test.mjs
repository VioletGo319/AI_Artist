import assert from 'node:assert/strict'
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'
import test from 'node:test'

const script = fileURLToPath(new URL('./check-gallery.mjs', import.meta.url))
const template = {
  title: 'Approved artwork', theme: 'Abstract Geometry', palette: 'blue and gold',
  composition: 'open space', style: 'painterly', prompt: 'A luminous sculpture',
  seed: 123, model: 'Tongyi-MAI/Z-Image-Turbo', date: '2026-11-23',
  generated_at: '2026-11-23T18:00:00Z', approved_at: '2026-11-23T19:00:00Z',
  steps: 9, width: 1024, height: 1024,
}

async function check(slots, mutate = () => {}) {
  const root = await mkdtemp(join(tmpdir(), 'curator-gallery-'))
  try {
    await mkdir(join(root, 'gallery/art'), { recursive: true })
    await mkdir(join(root, 'gallery/thumbs'), { recursive: true })
    const artworks = []
    for (const slot of slots) {
      const id = `${'a'.repeat(32)}-${slot}`
      const entry = { ...template, id, image: `gallery/art/${id}.png`, thumb: `gallery/thumbs/${id}.webp` }
      await writeFile(join(root, entry.image), 'image fixture')
      await writeFile(join(root, entry.thumb), 'thumbnail fixture')
      mutate(entry)
      artworks.push(entry)
    }
    await writeFile(join(root, 'gallery/manifest.json'), JSON.stringify({ version: 1, artworks }))
    return spawnSync(process.execPath, [script, root], { encoding: 'utf8' })
  } finally {
    await rm(root, { recursive: true, force: true })
  }
}

test('publication accepts all 256 canonical batch slots, including the approved 46, 96 and 123', async () => {
  const result = await check(Array.from({ length: 256 }, (_, i) => String(i + 1).padStart(2, '0')))
  assert.equal(result.status, 0, result.stderr)
  assert.match(result.stdout, /256 approved works/)
})

test('publication rejects out-of-range and noncanonical slots', async () => {
  for (const slot of ['00', '0', '1', '001', '257', '999', '-1']) {
    assert.notEqual((await check([slot])).status, 0, slot)
  }
})

test('larger batches retain publication safeguards', async () => {
  for (const mutate of [
    entry => { delete entry.approved_at },
    entry => { entry.mock = true },
    entry => { entry.model = 'mock/example' },
    entry => { entry.image = '../private/candidate.png' },
  ]) {
    assert.notEqual((await check(['123'], mutate)).status, 0)
  }
  assert.notEqual((await check(['123', '123'])).status, 0)
})
