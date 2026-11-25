import { chromium } from 'playwright-core'
import assert from 'node:assert/strict'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

const browser = await chromium.launch({
  executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  headless: true,
  args: ['--allow-file-access-from-files'],
})

const errors = []
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, colorScheme: 'light' })
page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()) })
page.on('pageerror', (error) => errors.push(error.message))
const url = pathToFileURL(path.resolve('../../assets/loopfill-interactive-guide.html')).href
await page.goto(url)
await page.waitForSelector('.hero')

const initial = await page.evaluate(() => ({
  theme: document.documentElement.className,
  bodyBackground: getComputedStyle(document.body).backgroundColor,
  bodyColor: getComputedStyle(document.body).color,
  scrollWidth: document.documentElement.scrollWidth,
  innerWidth: window.innerWidth,
  title: document.title,
}))
console.log('initial', initial)

const pipelineSlider = page.locator('.pipeline-demo input[type="range"]')
await pipelineSlider.focus()
await page.keyboard.press('Home')
await page.keyboard.press('ArrowRight')
await page.keyboard.press('ArrowRight')
await page.keyboard.press('ArrowRight')
await page.waitForFunction(() => document.querySelector('.step-narration')?.textContent?.includes('STEP 3/5'))

await page.getByRole('tab', { name: '+ 浅层草稿', exact: true }).click()
assert.equal(await page.locator('.path-metric strong').textContent(), '1.73')
await page.getByRole('tab', { name: '+ 二次检索', exact: true }).click()
assert.equal(await page.locator('.path-metric strong').textContent(), '2.28')

assert.equal(await page.locator('.gamma-primary strong').textContent(), '5.06')
await page.getByRole('button', { name: 'Huginn-3.5B', exact: true }).click()
await page.getByRole('button', { name: '采样 · T=1', exact: true }).click()
assert.equal(await page.locator('.bar-row').filter({ hasText: 'LoopFill' }).locator('strong').textContent(), '2.60×')
await page.getByRole('button', { name: 'B=8', exact: true }).click()
assert.equal(await page.locator('.throughput-number strong').textContent(), '94.08')

await page.locator('.quiz-item').nth(1).getByRole('button').click()
assert.equal(await page.locator('.quiz-item').nth(1).locator('p').count(), 1)

const downloadPromise = page.waitForEvent('download')
await page.getByRole('button', { name: '下载 Markdown', exact: true }).click()
const download = await downloadPromise
assert.equal(download.suggestedFilename(), 'loopfill-reading-guide.md')

await page.locator('.pipeline-demo').screenshot({ path: '../../qa/loopfill-guide/pipeline-validated.png' })
await page.locator('.tree-demo').screenshot({ path: '../../qa/loopfill-guide/tree-validated.png' })
await page.locator('.results-demo').screenshot({ path: '../../qa/loopfill-guide/results-validated.png' })
await page.evaluate(() => window.scrollTo(0, 0))
await page.screenshot({ path: '../../qa/loopfill-guide/desktop-light.png', fullPage: false })

await page.getByRole('button', { name: '切换深浅色', exact: true }).click()
await page.waitForTimeout(350)
console.log('dark-theme', await page.evaluate(() => ({
  theme: document.documentElement.className,
  bodyBackground: getComputedStyle(document.body).backgroundColor,
  bodyColor: getComputedStyle(document.body).color,
  leadColor: getComputedStyle(document.querySelector('.hero-lead')).color,
})))

await page.evaluate(() => window.scrollTo(0, 0))
await page.screenshot({ path: '../../qa/loopfill-guide/desktop-validated.png', fullPage: false })

const mobile = await browser.newPage({ viewport: { width: 390, height: 844 }, colorScheme: 'light', isMobile: true })
await mobile.goto(url)
await mobile.waitForSelector('.hero')
const mobileLayout = await mobile.evaluate(() => {
  const overflow = [...document.querySelectorAll('body *')].flatMap((element) => {
    const rect = element.getBoundingClientRect()
    const style = getComputedStyle(element)
    if (style.display === 'none' || style.visibility === 'hidden' || rect.width === 0) return []
    if (rect.right > window.innerWidth + 1 || rect.left < -1) {
      return [{ tag: element.tagName, className: element.className, left: Math.round(rect.left), right: Math.round(rect.right), width: Math.round(rect.width) }]
    }
    return []
  })
  return { scrollWidth: document.documentElement.scrollWidth, innerWidth: window.innerWidth, overflow: overflow.slice(0, 30) }
})
console.log('mobile-layout', JSON.stringify(mobileLayout, null, 2))
await mobile.screenshot({ path: '../../qa/loopfill-guide/mobile-validated.png', fullPage: false })

console.log('runtime-errors', errors)
assert.equal(errors.length, 0)
await browser.close()
console.log('All interaction checks passed.')
