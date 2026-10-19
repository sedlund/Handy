//! Log overlay renderer failures separately from transcription worker failures.
//! Resetting the WebView2 background after a hard driver failure did not restore
//! its compositor surface; do not mutate the controller from that callback.

use std::sync::atomic::{AtomicBool, Ordering};
use tauri::WebviewWindow;
use webview2_com::{
    Microsoft::Web::WebView2::Win32::{
        COREWEBVIEW2_PROCESS_FAILED_KIND_GPU_PROCESS_EXITED,
        COREWEBVIEW2_PROCESS_FAILED_KIND_RENDER_PROCESS_EXITED,
    },
    ProcessFailedEventHandler,
};

static RENDERER_FAILED: AtomicBool = AtomicBool::new(false);

/// A failed renderer cannot be relied on to animate or acknowledge a hide.
/// Consume once so later healthy recordings retain their normal fade-out.
pub(super) fn take_renderer_failure() -> bool {
    RENDERER_FAILED.swap(false, Ordering::AcqRel)
}

pub(super) fn watch_gpu_process(window: &WebviewWindow) {
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
                    if kind == COREWEBVIEW2_PROCESS_FAILED_KIND_GPU_PROCESS_EXITED
                        || kind == COREWEBVIEW2_PROCESS_FAILED_KIND_RENDER_PROCESS_EXITED
                    {
                        RENDERER_FAILED.store(true, Ordering::Release);
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
