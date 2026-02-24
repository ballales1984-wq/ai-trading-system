const { chromium } = require('playwright');

const TARGET_URL = 'http://localhost:8000';

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  const results = [];

  const viewports = [
    { name: 'desktop', width: 1366, height: 768 },
    { name: 'mobile', width: 390, height: 844 },
  ];

  for (const vp of viewports) {
    await page.setViewportSize({ width: vp.width, height: vp.height });
    await page.goto(TARGET_URL, { waitUntil: 'networkidle', timeout: 30000 });
    const title = await page.title();
    const hasDemo = (await page.locator('#demo').count()) > 0;
    const hasContact = (await page.locator('#contact').count()) > 0;
    await page.screenshot({ path: `./scripts/${vp.name}-landing.png`, fullPage: true });
    results.push({ page: 'landing', viewport: vp.name, title, hasDemo, hasContact });
  }

  await page.setViewportSize({ width: 1366, height: 768 });
  await page.goto(`${TARGET_URL}/dashboard`, { waitUntil: 'networkidle', timeout: 30000 });
  const dashboardHeading = (await page.locator('h1:has-text("Dashboard")').count()) > 0;
  const marketTable = (await page.locator('h2:has-text("Market Overview")').count()) > 0;
  const newsPanel = (await page.locator('h2:has-text("News & Sentiment")').count()) > 0;
  await page.keyboard.press('Tab');
  await page.keyboard.press('Tab');
  const activeTag = await page.evaluate(() => document.activeElement?.tagName || '');
  await page.screenshot({ path: './scripts/desktop-dashboard.png', fullPage: true });
  results.push({ page: 'dashboard', hasHeading: dashboardHeading, hasMarket: marketTable, hasNews: newsPanel, activeTag });

  await page.goto(`${TARGET_URL}/orders`, { waitUntil: 'networkidle', timeout: 30000 });
  const ordersHeading = (await page.locator('h1:has-text("Orders")').count()) > 0;
  const filterInput = (await page.locator('input[placeholder*="Filter symbol"]').count()) > 0;
  const refreshButton = (await page.locator('button:has-text("Refresh orders")').count()) > 0;
  await page.screenshot({ path: './scripts/desktop-orders.png', fullPage: true });
  results.push({ page: 'orders', hasHeading: ordersHeading, hasFilter: filterInput, hasRefresh: refreshButton });

  console.log(JSON.stringify(results, null, 2));
  await browser.close();
})();
