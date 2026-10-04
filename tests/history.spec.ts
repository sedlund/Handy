import { expect, test } from "@playwright/test";
import type { HistoryEntry, HistoryUpdatePayload } from "../src/bindings";

declare global {
  interface Window {
    historyFixture: {
      update: (payload: HistoryUpdatePayload) => Promise<void>;
      entry: (id: number) => HistoryEntry;
      pageLoads: () => number;
    };
  }
}

test("retention removes the oldest row without refreshing or moving the visible entry", async ({
  page,
}) => {
  await page.goto("/tests/fixtures/history.html");
  // Leave enough content below the anchor: deleting the bottom row can
  // legitimately clamp scrolling when there is no room to keep its offset.
  const anchor = page.getByText("History recording 4", { exact: true });
  await anchor.scrollIntoViewIfNeeded();
  await anchor.evaluate((element) => {
    const scroll = document.getElementById("history-scroll")!;
    scroll.scrollTop +=
      element.getBoundingClientRect().top -
      scroll.getBoundingClientRect().top -
      80;
  });
  expect(
    await page
      .locator("#history-scroll")
      .evaluate((element) => element.scrollTop),
  ).toBeGreaterThan(0);
  const before = (await anchor.boundingBox())!.y;
  await page.evaluate(async () => {
    await window.historyFixture.update({ action: "deleted", id: 1 });
    await window.historyFixture.update({
      action: "added",
      entry: window.historyFixture.entry(6),
    });
  });
  await expect(
    page.getByText("History recording 1", { exact: true }),
  ).toHaveCount(0);
  await expect(
    page.getByText("History recording 6", { exact: true }),
  ).toHaveCount(1);
  await expect
    .poll(async () => (await anchor.boundingBox())!.y)
    .toBeCloseTo(before, 0);
  expect(await page.evaluate(() => window.historyFixture.pageLoads())).toBe(1);
});

test("duplicate deletion notifications preserve loaded pages and scroll anchoring", async ({
  page,
}) => {
  await page.goto("/tests/fixtures/history.html?count=60");
  await expect(
    page.getByText("History recording 31", { exact: true }),
  ).toHaveCount(1);
  await page.locator("#history-scroll").evaluate((element) => {
    element.scrollTop = element.scrollHeight;
  });
  await expect(
    page.getByText("History recording 1", { exact: true }),
  ).toHaveCount(1);
  const anchor = page.getByText("History recording 40", { exact: true });
  await anchor.scrollIntoViewIfNeeded();
  const before = (await anchor.boundingBox())!.y;
  await page.evaluate(async () => {
    await window.historyFixture.update({ action: "deleted", id: 60 });
    await window.historyFixture.update({ action: "deleted", id: 60 });
  });
  await expect(
    page.getByText("History recording 60", { exact: true }),
  ).toHaveCount(0);
  await expect(
    page.getByText("History recording 1", { exact: true }),
  ).toHaveCount(1);
  await expect
    .poll(async () => (await anchor.boundingBox())!.y)
    .toBeCloseTo(before, 0);
  expect(await page.evaluate(() => window.historyFixture.pageLoads())).toBe(2);
});
