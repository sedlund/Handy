//! Preserve the recording overlay's transparent WebView2 surface after a
//! display-driver reset. WebView2 recreates its GPU process automatically;
//! restoring its background avoids reloading React and losing live UI state.

use tauri::WebviewWindow;
use webview2_com::{
    Microsoft::Web::WebView2::Win32::COREWEBVIEW2_PROCESS_FAILED_KIND_GPU_PROCESS_EXITED,
    ProcessFailedEventHandler,
};

pub(super) fn restore_transparency(window: &WebviewWindow) {
    // CSS remains transparent, but the native WebView2 background can become
    // opaque after device loss. Touch only the webview layer, not the window's
    // native brush. Reassert on show too, since a driver reset need not exit
    // the browser's GPU process. The transcription GPU setting is unaffected.
    let webview: &tauri::Webview = window.as_ref();
    if let Err(error) = webview.set_background_color(Some(tauri::webview::Color(0, 0, 0, 0))) {
        log::warn!("Failed to restore recording overlay transparency: {error}");
    }
}

pub(super) fn watch_gpu_process(window: &WebviewWindow) {
    let recovery_window = window.clone();
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
                    log::warn!("Recording overlay WebView2 process failure: {:?}", kind);
                    if kind == COREWEBVIEW2_PROCESS_FAILED_KIND_GPU_PROCESS_EXITED {
                        restore_transparency(&recovery_window);
                    }
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
