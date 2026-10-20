//! Log overlay renderer failures separately from transcription worker failures.
//! Resetting the WebView2 background after a hard driver failure did not restore
//! its compositor surface; do not mutate the controller from that callback.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use tauri::{AppHandle, Emitter, Listener, WebviewWindow};
use webview2_com::{
    Microsoft::Web::WebView2::Win32::{
        COREWEBVIEW2_PROCESS_FAILED_KIND_GPU_PROCESS_EXITED,
        COREWEBVIEW2_PROCESS_FAILED_KIND_RENDER_PROCESS_EXITED,
    },
    ProcessFailedEventHandler,
};

static RENDERER_FAILED: AtomicBool = AtomicBool::new(false);
static INSTANCE: AtomicU64 = AtomicU64::new(0);
static PAGE_READY: AtomicBool = AtomicBool::new(false);
static PENDING_SHOW: Mutex<Option<(u64, String, bool)>> = Mutex::new(None);

/// Install one listener for the app lifetime. A replacement starts hidden and
/// waits for React's event listeners; page-load completion alone is too early.
/// The instance token rejects late messages from a destroyed WebView.
pub(super) fn prepare_window(app: &AppHandle) -> String {
    static LISTENER: OnceLock<()> = OnceLock::new();
    LISTENER.get_or_init(|| {
        let handle = app.clone();
        app.listen("recording-overlay-ready", move |event| {
            let Ok(instance) = event.payload().parse::<u64>() else {
                return;
            };
            let handle = handle.clone();
            let ui = handle.clone();
            let _ = handle.run_on_main_thread(move || {
                if instance != INSTANCE.load(Ordering::Acquire) || renderer_failed() {
                    return;
                }
                PAGE_READY.store(true, Ordering::Release);
                let pending = PENDING_SHOW.lock().unwrap().take();
                if let Some((generation, state, capture_ready)) = pending {
                    if super::OVERLAY_VISIBILITY.is_current(generation) && !renderer_failed() {
                        super::show_overlay_state_on_main(&ui, &state);
                        if capture_ready {
                            let _ = ui.emit_to("recording_overlay", "recording-ready", ());
                        }
                    }
                }
            });
        });
    });
    let instance = INSTANCE.fetch_add(1, Ordering::AcqRel) + 1;
    PAGE_READY.store(false, Ordering::Release);
    RENDERER_FAILED.store(false, Ordering::Release);
    format!("window.__HANDY_OVERLAY_INSTANCE__ = {instance};")
}

pub(super) fn defer_show(generation: u64, state: &str) -> bool {
    if PAGE_READY.load(Ordering::Acquire) && !renderer_failed() {
        return false;
    }
    *PENDING_SHOW.lock().unwrap() = Some((generation, state.to_string(), false));
    true
}

pub(super) fn defer_recording_ready() -> bool {
    if PAGE_READY.load(Ordering::Acquire) && !renderer_failed() {
        return false;
    }
    if let Some(pending) = PENDING_SHOW.lock().unwrap().as_mut() {
        pending.2 = true;
    }
    true
}

/// A failed renderer cannot be relied on to animate or acknowledge a hide.
/// Leave the flag set until a replacement is created. Hiding the damaged
/// window does not repair its compositor surface.
pub(super) fn renderer_failed() -> bool {
    RENDERER_FAILED.load(Ordering::Acquire)
}

/// Invalidate the old page before asynchronous native destruction. A new
/// recording may queue its show while the replacement is being constructed.
pub(super) fn begin_rebuild() {
    INSTANCE.fetch_add(1, Ordering::AcqRel);
    PAGE_READY.store(false, Ordering::Release);
    RENDERER_FAILED.store(false, Ordering::Release);
}

pub(super) fn watch_gpu_process(window: &WebviewWindow) {
    let instance = INSTANCE.load(Ordering::Acquire);
    let result = window.with_webview(move |webview| {
        // SAFETY: with_webview runs on the UI thread, and WebView2 invokes this
        // event on that same thread. The handler keeps no raw COM pointers.
        let result = unsafe {
            webview.controller().CoreWebView2().and_then(|core| {
                let handler = ProcessFailedEventHandler::create(Box::new(move |_, args| {
                    let Some(args) = args else {
                        return Ok(());
                    };
                    let mut kind = Default::default();
                    args.ProcessFailedKind(&mut kind)?;
                    if instance == INSTANCE.load(Ordering::Acquire)
                        && (kind == COREWEBVIEW2_PROCESS_FAILED_KIND_GPU_PROCESS_EXITED
                            || kind == COREWEBVIEW2_PROCESS_FAILED_KIND_RENDER_PROCESS_EXITED)
                    {
                        RENDERER_FAILED.store(true, Ordering::Release);
                        PAGE_READY.store(false, Ordering::Release);
                    }
                    log::warn!("Recording overlay WebView2 process failure: {:?}", kind);
                    Ok(())
                }));
                // The core owns the event handler until the overlay is destroyed.
                let mut token = 0;
                core.add_ProcessFailed(&handler, &mut token)
            })
        };
        if let Err(error) = result {
            log::warn!("Failed to watch recording overlay GPU process: {error}");
        }
    });
    if let Err(error) = result {
        log::warn!("Failed to initialize recording overlay GPU watcher: {error}");
    }
}
