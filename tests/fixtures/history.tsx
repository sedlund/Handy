import React from "react";
import { createRoot } from "react-dom/client";
import i18next from "i18next";
import { initReactI18next } from "react-i18next";
import { emit } from "@tauri-apps/api/event";
import { mockIPC, mockWindows } from "@tauri-apps/api/mocks";
import { HistorySettings } from "../../src/components/settings/history/HistorySettings";
import type { HistoryEntry, HistoryUpdatePayload } from "../../src/bindings";
import translation from "../../src/i18n/locales/en/translation.json";
import "../../src/App.css";

const count = Number(new URLSearchParams(location.search).get("count") ?? 5);
const entry = (id: number): HistoryEntry => ({
  id,
  file_name: `recording-${id}.wav`,
  timestamp: 1700000000 + id,
  saved: false,
  title: `Recording ${id}`,
  transcription_text: `History recording ${id}`,
  post_processed_text: null,
  post_process_prompt: null,
  post_process_requested: false,
});
let entries = Array.from({ length: count }, (_, index) => entry(count - index));
let pageLoads = 0;

Object.assign(window, {
  __TAURI_OS_PLUGIN_INTERNALS__: { os_type: "windows" },
});
mockWindows("main");
mockIPC(
  (command, args) => {
    if (command === "get_history_entries") {
      pageLoads++;
      const cursor = args?.cursor as number | null;
      const limit = args?.limit as number;
      const remaining = entries.filter(
        (item) => cursor === null || item.id < cursor,
      );
      return {
        entries: remaining.slice(0, limit),
        has_more: remaining.length > limit,
      };
    }
    if (command === "delete_history_entry") {
      return update({ action: "deleted", id: args?.id as number });
    }
  },
  { shouldMockEvents: true },
);

async function update(payload: HistoryUpdatePayload) {
  if (payload.action === "deleted")
    entries = entries.filter((item) => item.id !== payload.id);
  if (payload.action === "added") entries = [payload.entry, ...entries];
  await emit("history-update-payload", payload);
}

Object.assign(window, {
  historyFixture: { update, entry, pageLoads: () => pageLoads },
});
await i18next
  .use(initReactI18next)
  .init({ lng: "en", resources: { en: { translation } } });
createRoot(document.getElementById("root")!).render(
  <div id="history-scroll" style={{ height: 350, overflowY: "auto" }}>
    <HistorySettings />
  </div>,
);
