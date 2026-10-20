import { test, expect } from "@playwright/test";

// Use the installed browser on Windows; elsewhere use Playwright's Chromium.
test.use(process.platform === "win32" ? { channel: "msedge" } : {});

test("fallback status is amber, preserves preference, and disappears on recovery", async ({
  page,
}) => {
  await page.addInitScript(() => {
    Object.assign(window, {
      accelerationTest: {
        settings: {
          transcribe_accelerator: "gpu",
          transcribe_gpu_device: "nvidia",
        },
        writes: [],
        listeners: {},
        available: {
          transcribe: ["auto", "gpu", "cpu"],
          gpu_devices: [
            {
              id: "nvidia",
              name: "NVIDIA test GPU",
              total_vram_mb: 6144,
              available: true,
            },
            {
              id: "amd",
              name: "AMD test GPU",
              total_vram_mb: 2048,
              available: true,
            },
          ],
          active_device: {
            name: "NVIDIA test GPU",
            on_gpu: true,
            is_fallback: false,
          },
        },
      },
    });
  });
  // Mock only the native boundary; render the real setting and translations.
  await page.route("**/src/hooks/useSettings.ts*", (route) =>
    route.fulfill({
      contentType: "application/javascript",
      body: `export function useSettings() { const s = window.accelerationTest; return {
      getSetting: key => s.settings[key], isUpdating: () => false,
      updateSetting: async (key, value) => { s.writes.push([key, value]); s.settings[key] = value; }
    }; }`,
    }),
  );
  await page.route("**/src/bindings.ts*", (route) =>
    route.fulfill({
      contentType: "application/javascript",
      body: "export const commands = { getAvailableAccelerators: async () => window.accelerationTest.available };",
    }),
  );
  await page.route("**/@tauri-apps_api_event.js*", (route) =>
    route.fulfill({
      contentType: "application/javascript",
      body: `export async function listen(name, cb) {
      window.accelerationTest.listeners[name] = cb;
      return () => delete window.accelerationTest.listeners[name];
    }`,
    }),
  );
  await page.goto("/tests/fixtures/acceleration.html");
  const selected = page.getByRole("button", { name: /NVIDIA test GPU/ });
  await expect(selected).toBeVisible();
  const status = page.locator('p[aria-live="polite"]');
  await expect(status).toHaveCount(0);

  const updateDevice = async (
    name: string,
    on_gpu: boolean,
    is_fallback: boolean,
    event: string,
  ) => {
    await page.evaluate(
      ({ name, on_gpu, is_fallback, event }) => {
        const state = (
          window as unknown as {
            accelerationTest: {
              available: {
                active_device: {
                  name: string;
                  on_gpu: boolean;
                  is_fallback: boolean;
                };
              };
              listeners: Record<
                string,
                (event: { payload: { event_type: string } }) => void
              >;
            };
          }
        ).accelerationTest;
        state.available.active_device = { name, on_gpu, is_fallback };
        state.listeners[event]({
          payload: { event_type: "loading_completed" },
        });
      },
      { name, on_gpu, is_fallback, event },
    );
  };
  await updateDevice(
    "AMD test GPU",
    true,
    true,
    "transcription-device-fallback",
  );
  await expect(status).toContainText("AMD test GPU (fallback)");
  await expect(status).toHaveClass(/text-amber-600/);
  await expect(selected).toBeVisible();

  await updateDevice("CPU", false, true, "transcription-device-fallback");
  await expect(status).toContainText("your computer's CPU (fallback)");
  await expect(selected).toBeVisible();

  await updateDevice("NVIDIA test GPU", true, false, "model-state-changed");
  await expect(status).toHaveCount(0);
  await expect(selected).toBeVisible();
  const writes = await page.evaluate(
    () =>
      (window as unknown as { accelerationTest: { writes: unknown[] } })
        .accelerationTest.writes,
  );
  expect(writes).toEqual([]);
});
