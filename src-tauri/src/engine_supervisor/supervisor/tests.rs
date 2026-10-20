use super::*;

fn env_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var_os(name).unwrap_or_else(|| panic!("set {name}")))
}

fn test_pcm() -> Vec<f32> {
    crate::audio_toolkit::read_wav_samples(env_path("HANDY_TEST_WAV")).expect("read wav")
}

fn spec() -> LoadSpec {
    LoadSpec {
        path: env_path("HANDY_TEST_MODEL"),
        backend: Backend::Auto,
        device: DeviceSelector::Auto,
    }
}

/// Faults are read by each worker at spawn, so set them before loading.
fn set_fault(fault: Option<&str>, gpu_only: bool) {
    let marker = std::env::temp_dir().join("handy-worker-fault-once");
    let _ = std::fs::remove_file(&marker);
    std::env::remove_var("HANDY_WORKER_LOAD_ERRORS");
    std::env::remove_var("HANDY_WORKER_FAULT_ONCE");
    std::env::remove_var("HANDY_WORKER_FAULT_GPU_ONLY");
    match fault {
        Some(fault) => {
            std::env::set_var("HANDY_WORKER_FAULT", fault);
            if gpu_only {
                std::env::set_var("HANDY_WORKER_FAULT_GPU_ONLY", "1");
            } else {
                std::env::set_var("HANDY_WORKER_FAULT_ONCE", marker);
            }
        }
        None => std::env::remove_var("HANDY_WORKER_FAULT"),
    }
}

/// The workers alive right now.
fn worker_count() -> usize {
    LIVE_WORKERS.load(Ordering::SeqCst)
}

/// Feed 30 ms frames a little faster than a microphone would.
fn feed_paced(stream: &StreamHandle, pcm: &[f32]) {
    for frame in pcm.chunks(480) {
        stream.feed(frame.to_vec());
        thread::sleep(Duration::from_millis(2));
    }
}

fn stream_all(engine: &EngineSupervisor, pcm: &[f32]) -> Result<Option<Finalized>, EngineError> {
    let stream = engine.start_stream(RunOptions::default(), StreamOptions::default(), |_| {})?;
    feed_paced(&stream, pcm);
    stream.finalize(false)
}

/// Block until the owner is waiting on `what` (a [`request_name`]) in
/// the worker, so a cancel lands on that work rather than before or
/// after it, however fast the model is.
fn wait_in_flight(engine: &EngineSupervisor, what: &str) {
    let started = Instant::now();
    while !lock(&engine.shared.in_flight)
        .as_ref()
        .is_some_and(|call| call.what == what)
    {
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "{what} never started"
        );
        thread::sleep(Duration::from_millis(1));
    }
}

/// The test clip repeated to at least `secs` seconds.
fn long_pcm(secs: usize) -> Vec<f32> {
    let clip = test_pcm();
    clip.repeat((secs * 16_000).div_ceil(clip.len()))
}

/// Never two workers: a reload or unload has fully stopped the old
/// worker before anything else happens.
#[test]
#[ignore]
fn one_worker_at_a_time() {
    set_fault(None, false);
    let engine = EngineSupervisor::new(true);
    for _ in 0..3 {
        engine.load(spec()).unwrap();
        assert_eq!(worker_count(), 1);
    }
    engine.unload().wait();
    assert_eq!(worker_count(), 0);
    assert!(engine.loaded().is_none());
    // A device probe with nothing loaded leaves no worker behind.
    assert!(engine.devices().is_some());
    assert_eq!(worker_count(), 0);
}

/// A cancel between two runs from one mark (two chunks of one
/// transcription) stops the second; a fresh mark runs again.
#[test]
#[ignore]
fn cancel_between_runs_stops_the_rest_of_the_mark() {
    set_fault(None, false);
    let engine = EngineSupervisor::new(true);
    engine.load(spec()).unwrap();
    let mark = engine.cancel_mark();
    engine
        .transcribe_since(mark, test_pcm(), RunOptions::default())
        .unwrap();
    engine.cancel();
    assert!(matches!(
        engine.transcribe_since(mark, test_pcm(), RunOptions::default()),
        Err(EngineError::Cancelled)
    ));
    engine
        .transcribe(test_pcm(), RunOptions::default())
        .unwrap();
}

/// A worker crash mid-stream loses the stream; the batch fallback then
/// runs in a fresh worker.
#[test]
#[ignore]
fn crash_mid_stream_falls_back_to_batch() {
    let pcm = test_pcm();
    set_fault(Some("abort@feed"), false);
    let engine = EngineSupervisor::new(true);
    engine.load(spec()).unwrap();
    assert!(stream_all(&engine, &pcm).unwrap().is_none());
    let t = Instant::now();
    let transcript = engine.transcribe(pcm, RunOptions::default()).unwrap();
    println!(
        "batch fallback after crash {:?} on '{}': {}",
        t.elapsed(),
        engine.loaded().unwrap().backend,
        transcript.text
    );
    assert!(engine.loaded().is_some());
    set_fault(None, false);
}

/// A hung finalize is killed at its deadline (sized by the buffered
/// audio, at least the device's call floor) and the batch fallback runs in a fresh
/// worker.
#[test]
#[ignore]
fn hung_finalize_is_killed_and_recovers() {
    let pcm = test_pcm();
    set_fault(Some("hang@finalize"), false);
    let engine = EngineSupervisor::new(true);
    engine.load(spec()).unwrap();
    let t = Instant::now();
    assert!(stream_all(&engine, &pcm).unwrap().is_none());
    println!("stream + hung finalize gave up after {:?}", t.elapsed());
    let t = Instant::now();
    let transcript = engine.transcribe(pcm, RunOptions::default()).unwrap();
    println!(
        "batch fallback after hang {:?}: {}",
        t.elapsed(),
        transcript.text
    );
    set_fault(None, false);
}

/// Cancelling a stream finalize reports `Cancelled`, never "no result"
/// (which would start a batch run), and the next use works.
#[test]
#[ignore]
fn cancel_during_finalize_is_cancelled() {
    let pcm = test_pcm();
    set_fault(Some("hang@finalize"), false);
    let engine = EngineSupervisor::new(true);
    engine.load(spec()).unwrap();
    let stream = engine
        .start_stream(RunOptions::default(), StreamOptions::default(), |_| {})
        .unwrap();
    feed_paced(&stream, &pcm);
    let finalize = thread::spawn(move || stream.finalize(false));
    wait_in_flight(&engine, "stream finalize");
    let t = Instant::now();
    engine.cancel();
    let result = finalize.join().unwrap();
    println!("finalize cancelled {:?} after cancel()", t.elapsed());
    assert!(matches!(result, Err(EngineError::Cancelled)));
    assert!(t.elapsed() < Duration::from_secs(2));
    assert!(!engine.shared.gpu_unavailable());
    engine.transcribe(pcm, RunOptions::default()).unwrap();
    set_fault(None, false);
}

/// Cancelling during a hung feed stops it at once (not at the feed
/// deadline), skips the queued frames, and the finalize reports
/// `Cancelled` even though the stream's worker is gone.
#[test]
#[ignore]
fn cancel_during_hung_feed_is_cancelled() {
    let pcm = test_pcm();
    set_fault(Some("hang@feed"), false);
    let engine = EngineSupervisor::new(true);
    engine.load(spec()).unwrap();
    let stream = engine
        .start_stream(RunOptions::default(), StreamOptions::default(), |_| {})
        .unwrap();
    feed_paced(&stream, &pcm);
    wait_in_flight(&engine, "stream feed");
    let t = Instant::now();
    engine.cancel();
    let result = stream.finalize(false);
    println!("hung feed cancelled {:?} after cancel()", t.elapsed());
    assert!(matches!(result, Err(EngineError::Cancelled)));
    assert!(t.elapsed() < Duration::from_secs(2));
    assert!(!engine.shared.gpu_unavailable());
    engine.transcribe(pcm, RunOptions::default()).unwrap();
    set_fault(None, false);
}

/// An unload reads as unloaded at once, even while it waits behind a
/// run, so a new dictation queues a fresh load instead of assuming the
/// model is there.
#[test]
#[ignore]
fn unload_reads_unloaded_at_once() {
    set_fault(None, false);
    let engine = EngineSupervisor::new(true);
    engine.load(spec()).unwrap();
    let run = {
        let engine = engine.clone();
        thread::spawn(move || engine.transcribe(long_pcm(60), RunOptions::default()))
    };
    wait_in_flight(&engine, "transcription");
    let unload = {
        let engine = engine.clone();
        thread::spawn(move || engine.unload().wait())
    };
    let t = Instant::now();
    while engine.loaded().is_some() {
        assert!(t.elapsed() < Duration::from_secs(1), "still reads loaded");
        thread::sleep(Duration::from_millis(1));
    }
    assert!(lock(&engine.shared.in_flight).is_some(), "run still going");
    run.join().unwrap().unwrap();
    unload.join().unwrap();
    assert!(engine.loaded().is_none());
    assert_eq!(worker_count(), 0);
}

/// An unload that is still waiting its turn can't affect a load
/// requested after it (a new recording right after an "Immediately"
/// unload): the newer model ends up loaded and usable.
#[test]
#[ignore]
fn queued_unload_never_affects_a_later_load() {
    set_fault(None, false);
    let engine = EngineSupervisor::new(true);
    engine.load(spec()).unwrap();
    let run = {
        let engine = engine.clone();
        thread::spawn(move || engine.transcribe(long_pcm(60), RunOptions::default()))
    };
    wait_in_flight(&engine, "transcription");
    drop(engine.unload());
    assert!(
        engine.loaded().is_none(),
        "a new recording must see no model"
    );
    engine.load(spec()).unwrap();
    run.join().unwrap().unwrap();
    assert!(engine.loaded().is_some());
    assert_eq!(worker_count(), 1);
    engine
        .transcribe(test_pcm(), RunOptions::default())
        .unwrap();
}

fn named_gpu(engine: &EngineSupervisor, name: &str) -> DeviceInfo {
    engine
        .devices()
        .unwrap()
        .into_iter()
        .find(|device| device.is_gpu() && device.label().contains(name))
        .unwrap_or_else(|| panic!("test requires {name}"))
}

fn expire_device_discovery(engine: &EngineSupervisor) {
    lock(&engine.shared.gpu_recheck).next = Some(Instant::now());
}

/// Driver faults leave a present GPU blacklisted across discovery scans.
/// A real worker proves both fallback dictation and explicit retry still work.
#[test]
#[ignore]
fn present_failed_gpu_stays_on_fallback_until_explicit_retry() {
    set_fault(None, false);
    let engine = EngineSupervisor::new(true);
    let nvidia = named_gpu(&engine, "NVIDIA");
    let preferred = LoadSpec {
        device: DeviceSelector::Key(nvidia.key.clone()),
        ..spec()
    };
    set_fault(Some("abort@run"), false);
    engine.load(preferred.clone()).unwrap();
    assert!(!engine
        .transcribe(test_pcm(), RunOptions::default())
        .unwrap()
        .text
        .trim()
        .is_empty());
    let fallback = engine.loaded().unwrap();
    assert_ne!(fallback.device_key.as_ref(), Some(&nvidia.key));
    set_fault(None, false);
    for _ in 0..3 {
        expire_device_discovery(&engine);
        assert!(engine.recheck_gpu().unwrap().is_none());
        assert_eq!(engine.loaded().unwrap().device_key, fallback.device_key);
        assert!(!engine
            .transcribe(test_pcm(), RunOptions::default())
            .unwrap()
            .text
            .trim()
            .is_empty());
    }
    engine.retry_gpu("user explicitly retried the accelerator");
    assert_eq!(
        engine.load(preferred).unwrap().device_key.as_ref(),
        Some(&nvidia.key)
    );
    engine.unload().wait();
}

#[test]
#[ignore]
fn observed_gpu_return_notifies_only_after_stream_finalize() {
    set_fault(None, false);
    let restored_events = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let count = restored_events.clone();
    let engine = EngineSupervisor::with_device_handlers(
        true,
        |_| {},
        move |_| {
            count.fetch_add(1, Ordering::SeqCst);
        },
    );
    let gpu = named_gpu(&engine, "NVIDIA");
    let devices = engine.devices().unwrap();
    let preferred = LoadSpec {
        device: DeviceSelector::Key(gpu.key.clone()),
        ..spec()
    };
    engine
        .shared
        .mark_gpu_failed(&gpu.key, gpu.label(), "removed");
    engine
        .shared
        .set_devices(devices.into_iter().filter(|d| d.key != gpu.key).collect());
    // Load CPU directly so the returning device is first observed by discovery.
    // Keep the original preference on the owner, as physical fallback does.
    engine.shared.gpu_unavailable.store(true, Ordering::Release);
    engine.load(preferred).unwrap();
    engine
        .shared
        .gpu_unavailable
        .store(false, Ordering::Release);
    expire_device_discovery(&engine);
    assert_eq!(
        engine.recheck_gpu().unwrap().unwrap().device_key.as_ref(),
        Some(&gpu.key)
    );
    assert_eq!(restored_events.load(Ordering::SeqCst), 0);
    let stream = engine
        .start_stream(RunOptions::default(), StreamOptions::default(), |_| {})
        .unwrap();
    feed_paced(&stream, &test_pcm());
    assert_eq!(restored_events.load(Ordering::SeqCst), 0);
    assert!(!stream
        .finalize(false)
        .unwrap()
        .unwrap()
        .text
        .full
        .trim()
        .is_empty());
    assert_eq!(restored_events.load(Ordering::SeqCst), 1);
    stream_all(&engine, &test_pcm()).unwrap();
    assert_eq!(restored_events.load(Ordering::SeqCst), 1);
    engine.unload().wait();
}

#[test]
#[ignore]
fn optional_upgrade_load_errors_preserve_working_cpu() {
    set_fault(None, false);
    let engine = EngineSupervisor::new(true);
    let nvidia = named_gpu(&engine, "NVIDIA");
    let amd = named_gpu(&engine, "AMD");
    let preferred = LoadSpec {
        device: DeviceSelector::Key(nvidia.key.clone()),
        ..spec()
    };
    engine
        .shared
        .mark_gpu_unavailable("test previous backend failure");
    assert!(!engine.load(preferred).unwrap().on_gpu);
    engine.retry_gpu("user retry");
    std::env::set_var(
        "HANDY_WORKER_LOAD_ERRORS",
        serde_json::to_string(&HashMap::from([
            (nvidia.key.clone(), "backend"),
            (amd.key.clone(), "allocation"),
        ]))
        .unwrap(),
    );
    expire_device_discovery(&engine);
    assert!(!engine.recheck_gpu().unwrap().unwrap().on_gpu);
    assert!(lock(&engine.shared.failed_gpus).contains_key(&amd.key));
    assert!(!engine
        .transcribe(test_pcm(), RunOptions::default())
        .unwrap()
        .text
        .trim()
        .is_empty());
    set_fault(None, false);
    engine.unload().wait();
}

#[test]
#[ignore]
fn returning_gpu_oom_restores_previous_gpu_then_cpu_if_needed() {
    set_fault(None, false);
    let engine = EngineSupervisor::new(true);
    let nvidia = named_gpu(&engine, "NVIDIA");
    let amd = named_gpu(&engine, "AMD");
    let preferred = LoadSpec {
        device: DeviceSelector::Key(nvidia.key.clone()),
        ..spec()
    };
    engine
        .shared
        .mark_gpu_failed(&nvidia.key, nvidia.label(), "test previous failure");
    assert_eq!(
        engine.load(preferred.clone()).unwrap().device_key.as_ref(),
        Some(&amd.key)
    );
    std::env::set_var(
        "HANDY_WORKER_LOAD_ERRORS",
        serde_json::to_string(&HashMap::from([(nvidia.key.clone(), "allocation")])).unwrap(),
    );
    // Simulate a confirmed off/on transition before each fresh attempt.
    engine.shared.set_devices(vec![amd.clone()]);
    expire_device_discovery(&engine);
    assert_eq!(
        engine.recheck_gpu().unwrap().unwrap().device_key.as_ref(),
        Some(&amd.key)
    );
    assert!(!engine
        .transcribe(test_pcm(), RunOptions::default())
        .unwrap()
        .text
        .trim()
        .is_empty());
    std::env::set_var(
        "HANDY_WORKER_LOAD_ERRORS",
        serde_json::to_string(&HashMap::from([
            (nvidia.key.clone(), "allocation"),
            (amd.key.clone(), "allocation"),
        ]))
        .unwrap(),
    );
    // Simulate a confirmed off/on transition before each fresh attempt.
    engine.shared.set_devices(vec![amd.clone()]);
    expire_device_discovery(&engine);
    assert!(
        !engine.recheck_gpu().unwrap().unwrap().on_gpu,
        "failed GPU restoration must still try CPU"
    );
    assert!(!engine
        .transcribe(test_pcm(), RunOptions::default())
        .unwrap()
        .text
        .trim()
        .is_empty());
    set_fault(None, false);
    engine.unload().wait();
}

#[test]
#[ignore]
fn idle_model_unload_does_not_release_a_failed_present_gpu() {
    set_fault(None, false);
    let engine = EngineSupervisor::new(true);
    let nvidia = named_gpu(&engine, "NVIDIA");
    let preferred = LoadSpec {
        device: DeviceSelector::Key(nvidia.key.clone()),
        ..spec()
    };
    engine
        .shared
        .mark_gpu_failed(&nvidia.key, nvidia.label(), "driver fault");
    assert_ne!(
        engine.load(preferred.clone()).unwrap().device_key.as_ref(),
        Some(&nvidia.key)
    );
    expire_device_discovery(&engine);
    engine.unload().wait();
    assert_ne!(
        engine.load(preferred).unwrap().device_key.as_ref(),
        Some(&nvidia.key)
    );
    assert!(!engine
        .transcribe(test_pcm(), RunOptions::default())
        .unwrap()
        .text
        .trim()
        .is_empty());
    engine.unload().wait();
}

/// A GPU-only crash retries the dictation on another healthy device.
/// Failed GPUs stay excluded until an explicit retry or an observed removal and return.
#[test]
#[ignore]
fn gpu_only_crash_recovers_on_another_device() {
    let pcm = test_pcm();
    set_fault(Some("abort@run"), true);
    let engine = EngineSupervisor::new(true);
    let info = engine.load(spec()).unwrap();
    assert!(info.on_gpu, "needs a GPU to test GPU-only faults");
    let transcript = engine
        .transcribe(pcm.clone(), RunOptions::default())
        .unwrap();
    println!(
        "recovered on {}: {}",
        engine.loaded().unwrap().device,
        transcript.text
    );
    assert!(engine.shared.has_gpu_failures());
    assert_ne!(engine.loaded().unwrap().device_key, info.device_key);
    let info = engine.load(spec()).unwrap();
    assert!(
        !info.on_gpu
            || !lock(&engine.shared.failed_gpus)
                .contains_key(info.device_key.as_deref().unwrap_or_default()),
        "later loads must not reuse a failed GPU"
    );
    engine.transcribe(pcm, RunOptions::default()).unwrap();
    set_fault(None, false);
}

/// A GPU driver that crashes while its backend registers, before the
/// hello: only a CPU-only worker survives it, so that is where the model
/// loads. The GPU is marked unavailable, later loads stay CPU-only and devices
/// aren't probed, until "try the GPU again" (a settings change).
#[test]
#[ignore]
fn gpu_backend_init_crash_uses_cpu_only_worker() {
    let pcm = test_pcm();
    set_fault(Some("abort@init"), true);
    let engine = EngineSupervisor::new(true);
    let info = engine.load(spec()).unwrap();
    assert!(!info.on_gpu, "loaded on '{}'", info.backend);
    assert!(engine.shared.gpu_unavailable());
    assert!(engine.devices().is_none());
    engine.transcribe(pcm, RunOptions::default()).unwrap();
    // The fault still fires in any worker with GPU backends.
    assert!(!engine.load(spec()).unwrap().on_gpu);
    set_fault(None, false);
    engine.retry_gpu("test");
    assert!(engine.load(spec()).unwrap().on_gpu);
    assert!(engine.devices().is_some());
}

/// With the CPU accelerator, the model worker is CPU-only (a GPU-only
/// init fault never fires), and a device probe runs beside it rather
/// than stopping it.
#[test]
#[ignore]
fn cpu_setting_uses_cpu_only_worker_and_probe_leaves_it_running() {
    let pcm = test_pcm();
    set_fault(Some("abort@init"), true);
    let engine = EngineSupervisor::new(true);
    let info = engine
        .load(LoadSpec {
            backend: Backend::Cpu,
            ..spec()
        })
        .unwrap();
    assert!(!info.on_gpu);
    assert!(!engine.shared.gpu_unavailable());
    set_fault(None, false);
    assert!(engine.devices().is_some());
    assert_eq!(worker_count(), 1);
    assert!(engine.loaded().is_some());
    engine.transcribe(pcm, RunOptions::default()).unwrap();
}

/// A host that must not use the GPU: even an `Auto` load and the device
/// probe run CPU-only, so a GPU-only init fault never fires.
#[test]
#[ignore]
fn gpu_disallowed_host_never_registers_a_gpu_backend() {
    set_fault(Some("abort@init"), true);
    let engine = EngineSupervisor::new(false);
    let devices = engine.devices().expect("devices not listed");
    assert!(devices.iter().all(|d| !d.is_gpu()), "{devices:?}");
    assert!(!engine.load(spec()).unwrap().on_gpu);
    assert!(!engine.shared.gpu_unavailable());
    set_fault(None, false);
}

/// The mask itself: a CPU-only worker registers no GPU device, including
/// compiled-in ones such as Metal.
#[test]
#[ignore]
fn cpu_only_worker_lists_no_gpu() {
    set_fault(None, false);
    let mut worker = Worker::spawn(true).unwrap();
    let Ok(Response::Hello {
        devices: Some(devices),
    }) = worker.call(&Request::Hello { list_devices: true }, None, HELLO_TIMEOUT)
    else {
        panic!("no device list");
    };
    println!("CPU-only worker devices: {devices:?}");
    assert!(!devices.is_empty());
    assert!(devices.iter().all(|d| !d.is_gpu()));
}

/// An explicit device never falls back to CPU.
#[test]
#[ignore]
fn pinned_device_fails_loudly() {
    let pcm = test_pcm();
    set_fault(Some("abort@run"), true);
    let engine = EngineSupervisor::new(true);
    let devices = engine.devices().expect("devices not listed");
    let gpu = devices.iter().find(|d| d.is_gpu()).expect("needs a GPU");
    let info = engine
        .load(LoadSpec {
            device: DeviceSelector::Index(gpu.index.unwrap()),
            ..spec()
        })
        .unwrap();
    assert!(info.on_gpu);
    assert!(engine.transcribe(pcm, RunOptions::default()).is_err());
    assert!(!engine.shared.gpu_unavailable());
    set_fault(None, false);
}

/// Cancel mid-run: the run stops promptly, without a retry, the model
/// still reads as loaded, and the next run works in a fresh worker.
#[test]
#[ignore]
fn cancel_stops_a_run() {
    let pcm = long_pcm(60);
    set_fault(None, false);
    let engine = EngineSupervisor::new(true);
    engine.load(spec()).unwrap();
    let run = {
        let engine = engine.clone();
        let pcm = pcm.clone();
        thread::spawn(move || engine.transcribe(pcm, RunOptions::default()))
    };
    wait_in_flight(&engine, "transcription");
    let t = Instant::now();
    engine.cancel();
    let result = run.join().unwrap();
    println!("cancelled {:?} after cancel()", t.elapsed());
    assert!(matches!(result, Err(EngineError::Cancelled)));
    assert!(t.elapsed() < Duration::from_secs(2));
    assert!(engine.loaded().is_some());
    assert!(!engine.shared.gpu_unavailable());
    let t = Instant::now();
    engine
        .transcribe(test_pcm(), RunOptions::default())
        .unwrap();
    println!("next run took {:?}", t.elapsed());
}

/// Cancel a hung run: killed at once, no retry, no GPU
/// flag, and the next run starts a fresh worker.
#[test]
#[ignore]
fn cancel_kills_a_hung_run() {
    let pcm = test_pcm();
    set_fault(Some("hang@run"), false);
    let engine = EngineSupervisor::new(true);
    engine.load(spec()).unwrap();
    let run = {
        let engine = engine.clone();
        let pcm = pcm.clone();
        thread::spawn(move || engine.transcribe(pcm, RunOptions::default()))
    };
    wait_in_flight(&engine, "transcription");
    // Cancel only once the worker is actually hung, not on its way there.
    let marker = std::env::temp_dir().join("handy-worker-fault-once");
    while !marker.exists() {
        thread::sleep(Duration::from_millis(1));
    }
    let t = Instant::now();
    engine.cancel();
    let result = run.join().unwrap();
    println!("hung run cancelled {:?} after cancel()", t.elapsed());
    assert!(matches!(result, Err(EngineError::Cancelled)));
    assert!(t.elapsed() < Duration::from_secs(2));
    assert!(!engine.shared.gpu_unavailable());
    engine.transcribe(pcm, RunOptions::default()).unwrap();
    set_fault(None, false);
}

/// A worker hung in native code still exits when its parent goes away
/// (stdin closes), without being killed.
#[test]
#[ignore]
fn hung_worker_exits_when_stdin_closes() {
    set_fault(Some("hang@run"), false);
    let mut worker = Worker::spawn(false).unwrap();
    worker
        .call(
            &Request::Hello {
                list_devices: false,
            },
            None,
            HELLO_TIMEOUT,
        )
        .unwrap();
    let spec = spec();
    let load = Request::Load {
        path: spec.path,
        backend: spec.backend,
        device: spec.device,
    };
    worker.call(&load, None, LOAD_TIMEOUT).unwrap();
    let run = Request::Run {
        options: RunOptions::default(),
    };
    assert!(worker
        .control
        .send(encode_message(&run, Some(&test_pcm())).unwrap()));
    thread::sleep(Duration::from_secs(1));
    worker.control.close_stdin();
    let t = Instant::now();
    let status = loop {
        if let Some(status) = lock(&worker.control.child).try_wait().unwrap() {
            break status;
        }
        assert!(t.elapsed() < SHUTDOWN_GRACE, "worker outlived its stdin");
        thread::sleep(Duration::from_millis(10));
    };
    println!(
        "hung worker exited {:?} after stdin closed: {status}",
        t.elapsed()
    );
    assert!(status.success());
    set_fault(None, false);
}
