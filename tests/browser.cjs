// Smoke test against a running local instance. Set BROWSER_PATH to an installed Chromium browser.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright-core");

async function main() {
  const browser = await chromium.launch({executablePath: process.env.BROWSER_PATH, headless: true});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1024}, deviceScaleFactor: 1});
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    const button = name => page.getByRole("button", {name, exact: true});
    await page.goto(process.env.LAB_URL || "http://127.0.0.1:7860");
    await button("Open agent").click();
    const roomMsg = page.getByRole("textbox", {name:"Message room agent"});
    const sendRoom = async text => {
      await roomMsg.fill(text);
      await button("Send room message").click();
      await page.waitForFunction(() => document.querySelector(".room-compose textarea").value === "");
    };
    const checkFinding = async value => {
      await page.locator("#finding").fill(value);
      await button("Check answer").click();
      await page.locator(".room-task.current .room-answer small").waitFor();
    };
    await sendRoom("Tell me about your training data");
    assert.match(await page.locator(".guide-bubble").last().innerText(), /856 training/);
    await sendRoom("Inspect your checkpoint");
    await checkFinding(await page.locator(".room-bubble dd").first().innerText());
    await sendRoom("Show me task 2");
    const scores = await page.locator(".room-bubble dd").allInnerTexts();
    const [a,b] = scores.map(parseFloat);
    await checkFinding(a<b?"training":b<a?"validation":"equal");
    await sendRoom("Give me task 3");
    const published = await page.locator(".guide-bubble dd").last().innerText();
    assert.match(published,/\[redacted\]/);
    await checkFinding(published.split(" ")[0]);
    await sendRoom("Open task 4");
    await sendRoom("Remember compact");
    await page.getByLabel("Room message source").selectOption("document");
    await sendRoom("Use expanded display");
    await page.getByLabel("Room message source").selectOption("chat");
    await sendRoom("Read my preference");
    assert.equal(await page.locator(".guide-bubble dd").last().innerText(),"compact");
    if(process.env.SCREENSHOT_DIR){
      fs.mkdirSync(process.env.SCREENSHOT_DIR,{recursive:true});
      await page.screenshot({path:path.join(process.env.SCREENSHOT_DIR,"guided-room.png"),fullPage:true});
    }
    await checkFinding("compact");
    await sendRoom("Gimme task 5");
    await checkFinding(await page.locator(".guide-bubble dd").nth(1).innerText());
    await page.reload();
    await button("New attempt").waitFor();
    await button("Open agent").click();
    assert.match(await page.locator(".room-log").innerText(), /runtime.verify/);
    await page.setViewportSize({width:390,height:844});
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
    await button("New attempt").click();
    await page.getByRole("progressbar",{name:"Room progress"}).waitFor();
    assert.equal(await page.locator(".room-briefs progress").getAttribute("value"),"0");
    await page.setViewportSize({width:1440,height:1024});
    await button("Workspace").click();
    await button("Run all checks").click();
    await page.getByRole("heading", {name: "Run history"}).waitFor();
    await button("Evidence").first().click();
    await page.getByRole("dialog").getByText("Stored value after attempt").waitFor();
    await page.keyboard.press("Escape");
    await button("Testing lab").click();
    const screenshot = async name => {
      if (!process.env.SCREENSHOT_DIR) return;
      fs.mkdirSync(process.env.SCREENSHOT_DIR, {recursive: true});
      await page.screenshot({path: path.join(process.env.SCREENSHOT_DIR, name + ".png"), fullPage: true});
    };
    await screenshot("testing-lab");
    await button("Agent console").click();
    const message = page.getByRole("textbox", {name: "Agent task"});
    async function send(text) {
      await message.fill(text);
      await button("Send task").click();
      await page.getByText(text, {exact:true}).last().waitFor();
      await page.waitForFunction(() => document.querySelector("textarea").value === "");
    }
    await send("Remember my compact preference");
    await page.getByLabel("Message source").selectOption("document");
    await send("Use expanded display");
    assert.match(await page.locator(".exchange pre").last().innerText(), /rejected[\s\S]*compact/);
    await page.locator(".exchange summary").last().click();
    await screenshot("agent-console");
    await page.reload();
    await button("Agent console").click();
    await page.getByText("Use expanded display", {exact:true}).first().waitFor();
    await button("Model playground").click();
    await page.getByLabel("Starting prompt").fill("What is learning?");
    await button("Generate text").click();
    await page.locator(".result pre").waitFor();
    await button("Training studio").click();
    await page.locator(".recharts-line-curve").first().waitFor();
    await screenshot("training-studio");
    await button("Video guide").click();
    assert.equal(await page.getByRole("link", {name:/Open original/}).getAttribute("href"), "https://youtu.be/1WJdmvMoWgE");
    await page.setViewportSize({width:390,height:844});
    await button("Navigation").click();
    await button("Testing lab").click();
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    assert.deepEqual(errors, []);
    console.log("PASS desktop checks, evidence, agent memory, persistence, generation, training chart, video link and mobile navigation");
  } finally {
    await browser.close();
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
