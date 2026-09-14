import { pathToFileURL } from "node:url";
import puppeteer from "puppeteer-core";

const chrome =
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const htmlPath = process.argv[2];
const pdfPath = process.argv[3];

const browser = await puppeteer.launch({
  executablePath: chrome,
  headless: true,
  args: ["--disable-gpu", "--no-first-run", "--no-default-browser-check"],
});
const page = await browser.newPage();
await page.goto(pathToFileURL(htmlPath).href, {
  waitUntil: "load",
  timeout: 60000,
});
const needsMjx = await page.evaluate(
  () => !!document.querySelector("script[src*='mathjax'], script[src*='tex-chtml']"),
);
if (needsMjx) {
  try {
    await page.waitForFunction(
      () => document.documentElement.getAttribute("data-mjx") === "done",
      { timeout: 20000 },
    );
  } catch {
    /* CDN down: print without rendered math */
  }
}
await page.evaluate("document.fonts.ready");
await page.pdf({
  path: pdfPath,
  printBackground: true,
  preferCSSPageSize: true,
  displayHeaderFooter: false,
});
await browser.close();
