import { expect, test } from '@playwright/test'

test('production build displays the empty approved manifest', async ({ page }) => {
  await page.goto('http://127.0.0.1:4174/')
  await expect(page.getByRole('heading', { name: 'AI Art Museum' })).toBeVisible()
  await expect(page.getByText('The collection opens with its first approved work.')).toBeVisible()
  await expect(page.locator('.tile')).toHaveCount(0)
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(page.viewportSize()!.width)
})

test('mock approval feeds the masonry gallery, filters, details, download and favorites', async ({ page }, testInfo) => {
  await page.goto('/#collection')
  const collection = page.locator('#collection')
  const cards = collection.locator('.tile')
  await expect(cards).toHaveCount(4)
  await expect(page.locator('#monthly .tile')).toHaveCount(3)
  await expect(page.getByText('Private demo · simulated artworks')).toBeVisible()
  if (process.env.ART_SCREENSHOT_DIR) {
    await page.screenshot({ path: `${process.env.ART_SCREENSHOT_DIR}/museum-${testInfo.project.name}.png`, fullPage: true })
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(page.viewportSize()!.width)
  const first = cards.first()
  const title = await first.locator('h3').innerText()
  const theme = await first.locator('.tile-caption span').innerText()
  await collection.getByLabel('Theme').selectOption({ label: theme })
  await expect(cards).toHaveCount(1)
  await collection.getByLabel('Date').selectOption({ index: 1 })
  await expect(cards).toHaveCount(1)
  await first.getByRole('button', { name: 'Add to favorites' }).click()
  await page.getByRole('link', { name: 'Favorites' }).click()
  await expect(page.locator('#favorites .tile')).toHaveCount(1)
  await page.reload()
  await expect(page.locator('#favorites .tile')).toHaveCount(1)
  await page.getByRole('link', { name: `View ${title}` }).click()
  await expect(page.getByRole('heading', { name: title })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Prompt' })).toBeVisible()
  await expect(page.getByText('mock/procedural')).toBeVisible()
  const downloadEvent = page.waitForEvent('download')
  await page.getByRole('link', { name: 'Download image' }).click()
  const download = await downloadEvent
  expect(download.suggestedFilename()).toMatch(/\.png$/)
  expect(await download.failure()).toBeNull()
})
