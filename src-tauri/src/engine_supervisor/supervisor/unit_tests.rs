use super::*;

fn loaded(key: Option<&str>) -> LoadedInfo {
    LoadedInfo {
        arch: "test".into(),
        variant: "test".into(),
        backend: "test".into(),
        device: key.unwrap_or("CPU").into(),
        device_key: key.map(str::to_string),
        on_gpu: key.is_some(),
        supports_initial_prompt: false,
        supports_long_form: false,
        supports_vocabulary: false,
        capabilities: transcribe_cpp::Capabilities {
            native_sample_rate: 16000,
            languages: vec![],
            translate_target_languages: vec![],
            max_timestamp_kind: transcribe_cpp::TimestampKind::None,
            supports_language_detect: false,
            supports_translate: false,
            supports_streaming: false,
            supports_spec_decode: false,
            max_audio_ms: 0,
        },
    }
}

fn policy_owner() -> Owner {
    Owner {
        shared: Arc::new(Shared {
            gpu_allowed: true,
            ..Shared::default()
        }),
        worker: None,
        spec: None,
        stream: None,
        unloads_done: 0,
    }
}

#[test]
fn discovery_is_throttled_without_postponing_the_next_scan() {
    let mut recheck = GpuRecheck::default();
    let before = Instant::now();
    recheck.schedule();
    let first = recheck.next.unwrap();
    assert!(first >= before + AUTO_DEVICE_RECHECK_INTERVAL);
    assert!(!recheck.due());
    recheck.schedule();
    assert_eq!(recheck.next, Some(first));
    recheck.next = Some(Instant::now());
    assert!(recheck.due());
}

#[test]
fn repeated_enumeration_keeps_present_failed_gpu_blacklisted() {
    let shared = Shared::default();
    let devices = vec![gpu("failed", 12), gpu("healthy", 6)];
    shared.set_devices(devices.clone());
    shared.mark_gpu_failed("failed", "Failed GPU", "device lost");
    for _ in 0..5 {
        shared.set_devices(devices.clone());
        assert_eq!(
            shared.recovery_spec(&auto_spec()).device,
            DeviceSelector::RecoveryKey("healthy".into())
        );
    }
    assert!(lock(&shared.missing_failed_gpus).is_empty());
}

#[test]
fn only_observed_removal_and_return_releases_a_failed_device() {
    let shared = Shared::default();
    let devices = vec![gpu("returning", 12), gpu("unstable", 6)];
    shared.set_devices(devices.clone());
    shared.mark_gpu_failed("returning", "Returning GPU", "removed");
    shared.mark_gpu_failed("unstable", "Unstable GPU", "driver fault");
    shared.set_devices(vec![devices[1].clone()]);
    assert!(lock(&shared.failed_gpus).contains_key("returning"));
    shared.set_devices(devices);
    assert!(!lock(&shared.failed_gpus).contains_key("returning"));
    assert!(lock(&shared.failed_gpus).contains_key("unstable"));
    assert_eq!(
        shared.recovery_spec(&auto_spec()).device,
        DeviceSelector::RecoveryKey("returning".into())
    );
}

#[test]
fn removal_confirmed_before_failure_attribution_still_allows_return() {
    let shared = Shared::default();
    shared.set_devices(vec![gpu("healthy", 6)]);
    shared.mark_gpu_failed("removed", "Removed GPU", "allocation on a removed GPU");
    shared.set_devices(vec![gpu("removed", 12), gpu("healthy", 6)]);
    assert!(!lock(&shared.failed_gpus).contains_key("removed"));
    assert!(lock(&shared.missing_failed_gpus).is_empty());
}

#[test]
fn cpu_recovers_to_amd_when_saved_nvidia_is_absent() {
    let preferred = LoadSpec {
        device: DeviceSelector::Key("nvidia".into()),
        ..auto_spec()
    };
    let devices = vec![gpu("amd", 2)];
    let candidate = upgrade_candidate(&preferred, Some(&loaded(None)), &devices).unwrap();
    assert_eq!(candidate.key, "amd");
    assert_eq!(preferred.device, DeviceSelector::Key("nvidia".into()));
    assert!(upgrade_candidate(&preferred, Some(&loaded(None)), &[]).is_none());
}

#[test]
fn recovery_scales_to_multiple_devices_without_vendor_assumptions() {
    let preferred = LoadSpec {
        device: DeviceSelector::Key("chosen".into()),
        ..auto_spec()
    };
    let devices = vec![
        gpu("chosen", 4),
        gpu("integrated", 2),
        gpu("external", 24),
        gpu("discrete", 12),
    ];
    let mut failed = HashMap::from([("chosen".into(), "Chosen GPU".into())]);
    for key in ["external", "discrete", "integrated"] {
        let next = select_recovery_spec(&preferred, false, &failed, &devices);
        assert_eq!(next.device, DeviceSelector::RecoveryKey(key.into()));
        failed.insert(key.into(), key.into());
    }
    assert!(select_recovery_spec(&preferred, false, &failed, &devices).cpu_only());
    // Recovery upgrades to the saved choice even when another device is larger.
    assert_eq!(
        upgrade_candidate(&preferred, Some(&loaded(None)), &devices)
            .unwrap()
            .key,
        "chosen"
    );
    let missing = LoadSpec {
        device: DeviceSelector::Key("absent".into()),
        ..auto_spec()
    };
    assert_eq!(
        upgrade_candidate(&missing, Some(&loaded(None)), &devices)
            .unwrap()
            .key,
        "external"
    );
    assert_eq!(preferred.device, DeviceSelector::Key("chosen".into()));
}

#[test]
fn automatic_upgrade_prefers_saved_gpu_and_never_reloads_the_active_gpu() {
    let preferred = LoadSpec {
        device: DeviceSelector::Key("nvidia".into()),
        ..auto_spec()
    };
    let devices = vec![gpu("amd", 12), gpu("nvidia", 6)];
    assert_eq!(
        upgrade_candidate(&preferred, Some(&loaded(Some("amd"))), &devices)
            .unwrap()
            .key,
        "nvidia"
    );
    assert!(upgrade_candidate(&preferred, Some(&loaded(Some("nvidia"))), &devices).is_none());
    assert!(upgrade_candidate(&preferred, Some(&loaded(Some("amd"))), &devices[..1]).is_none());
    assert!(
        upgrade_candidate(&auto_spec(), Some(&loaded(Some("amd"))), &devices).is_none(),
        "Auto must not downgrade a healthy GPU"
    );
}

#[test]
fn fallback_status_and_discovery_clear_when_preferred_device_returns() {
    let owner = policy_owner();
    let preferred = LoadSpec {
        device: DeviceSelector::Key("nvidia".into()),
        ..auto_spec()
    };
    owner.update_gpu_recheck(&preferred, &loaded(Some("amd")));
    assert!(owner.shared.fallback_active.load(Ordering::Acquire));
    assert!(lock(&owner.shared.gpu_recheck).next.is_some());
    owner.shared.mark_gpu_failed("amd", "AMD", "old failure");
    owner.update_gpu_recheck(&preferred, &loaded(Some("nvidia")));
    assert!(!owner.shared.fallback_active.load(Ordering::Acquire));
    assert!(
        lock(&owner.shared.gpu_recheck).next.is_none(),
        "unrelated failure history must not keep polling"
    );
}

#[test]
fn backend_initialization_failure_requires_an_explicit_retry() {
    let owner = policy_owner();
    owner
        .shared
        .mark_gpu_unavailable("backend initialization crashed");
    owner.update_gpu_recheck(&auto_spec(), &loaded(None));
    assert!(!lock(&owner.shared.gpu_recheck).due());
    assert!(lock(&owner.shared.gpu_recheck).next.is_none());
    assert!(owner.shared.recovery_spec(&auto_spec()).cpu_only());
}

#[test]
fn healthy_auto_gpu_clears_fallback_without_releasing_failed_devices() {
    let owner = policy_owner();
    owner
        .shared
        .mark_gpu_failed("unstable", "Unstable GPU", "driver fault");
    owner.update_gpu_recheck(&auto_spec(), &loaded(None));
    assert!(owner.shared.fallback_active.load(Ordering::Acquire));
    owner.update_gpu_recheck(&auto_spec(), &loaded(Some("healthy")));
    assert!(!owner.shared.fallback_active.load(Ordering::Acquire));
    assert!(owner.shared.device_restore_pending.load(Ordering::Acquire));
    assert!(lock(&owner.shared.failed_gpus).contains_key("unstable"));
    assert!(lock(&owner.shared.gpu_recheck).next.is_some());
}

#[test]
fn explicit_cpu_and_gpu_disabled_hosts_never_schedule_gpu_rechecks() {
    let owner = policy_owner();
    owner.update_gpu_recheck(&auto_spec().cpu(), &loaded(None));
    assert!(!owner.shared.fallback_active.load(Ordering::Acquire));
    assert!(lock(&owner.shared.gpu_recheck).next.is_none());
    let owner = Owner {
        shared: Arc::new(Shared::default()),
        ..owner
    };
    owner.update_gpu_recheck(&auto_spec(), &loaded(None));
    assert!(lock(&owner.shared.gpu_recheck).next.is_none());
}

#[test]
fn removal_requires_a_fresh_list_without_the_bound_identity() {
    let mut gpu = DeviceInfo {
        name: "GPU".into(),
        description: "GPU".into(),
        kind: "Vulkan".into(),
        device_type: transcribe_cpp::DeviceType::Gpu,
        device_id: Some("pci:01:00.0".into()),
        memory_total: 0,
        index: Some(0),
        key: "bound-gpu-key".into(),
    };
    assert!(!device_missing("bound-gpu-key", None));
    assert!(!device_missing(
        "bound-gpu-key",
        Some(std::slice::from_ref(&gpu))
    ));
    gpu.index = Some(1);
    assert!(!device_missing(
        "bound-gpu-key",
        Some(std::slice::from_ref(&gpu))
    ));
    // A different GPU with the same label must not hide device removal.
    gpu.key = "different-gpu-key".into();
    assert!(device_missing("bound-gpu-key", Some(&[gpu])));
    assert!(device_missing("bound-gpu-key", Some(&[])));
}

fn gpu(key: &str, memory: u64) -> DeviceInfo {
    DeviceInfo {
        name: key.into(),
        description: key.into(),
        kind: "vulkan".into(),
        device_type: transcribe_cpp::DeviceType::Gpu,
        device_id: Some(key.into()),
        memory_total: memory,
        index: Some(0),
        key: key.into(),
    }
}

#[test]
fn failed_nvidia_falls_back_to_amd_without_changing_the_preference() {
    let preferred = LoadSpec {
        device: DeviceSelector::Key("nvidia".into()),
        ..auto_spec()
    };
    let devices = vec![gpu("nvidia", 6), gpu("amd", 2)];
    let failed = HashMap::from([("nvidia".into(), "NVIDIA".into())]);
    let recovery = select_recovery_spec(&preferred, false, &failed, &devices);
    assert_eq!(recovery.device, DeviceSelector::RecoveryKey("amd".into()));
    assert_eq!(recovery.backend, Backend::Auto);
    assert_eq!(preferred.device, DeviceSelector::Key("nvidia".into()));
    // Re-enumeration may remove NVIDIA and renumber AMD; selection uses identity.
    let mut amd = gpu("amd", 2);
    amd.index = Some(1);
    assert_eq!(
        select_recovery_spec(&preferred, false, &failed, &[amd]).device,
        recovery.device
    );
}

#[test]
fn cpu_fallback_is_used_only_when_no_healthy_gpu_remains() {
    let devices = vec![gpu("nvidia", 6), gpu("amd", 2)];
    let mut failed = HashMap::from([("nvidia".into(), "NVIDIA".into())]);
    assert_eq!(
        select_recovery_spec(&auto_spec(), false, &failed, &devices).device,
        DeviceSelector::RecoveryKey("amd".into())
    );
    failed.insert("amd".into(), "AMD".into());
    let recovery = select_recovery_spec(&auto_spec(), false, &failed, &devices);
    assert_eq!(recovery.backend, Backend::Cpu);
    assert_eq!(recovery.device, DeviceSelector::Auto);
}

#[test]
fn healthy_saved_gpu_wins_over_another_larger_gpu_during_recovery() {
    let preferred = LoadSpec {
        device: DeviceSelector::Key("amd".into()),
        ..auto_spec()
    };
    let failed = HashMap::from([("nvidia".into(), "NVIDIA".into())]);
    assert_eq!(
        select_recovery_spec(
            &preferred,
            false,
            &failed,
            &[gpu("amd", 2), gpu("other", 12)]
        )
        .device,
        DeviceSelector::RecoveryKey("amd".into())
    );
}

#[test]
fn explicit_cpu_and_pinned_devices_are_preserved_and_init_faults_use_cpu() {
    let failed = HashMap::from([("nvidia".into(), "NVIDIA".into())]);
    let devices = vec![gpu("amd", 2)];
    let cpu = auto_spec().cpu();
    assert_eq!(
        select_recovery_spec(&cpu, false, &failed, &devices).backend,
        Backend::Cpu
    );
    let pinned = LoadSpec {
        device: DeviceSelector::Index(2),
        ..auto_spec()
    };
    assert_eq!(
        select_recovery_spec(&pinned, true, &failed, &devices).device,
        pinned.device
    );
    assert_eq!(
        select_recovery_spec(&auto_spec(), true, &failed, &devices).backend,
        Backend::Cpu
    );
}

#[test]
fn per_gpu_recovery_notifies_once_and_retry_clears_its_quarantine() {
    let engine = EngineSupervisor::new(true);
    assert!(engine
        .shared
        .mark_gpu_failed("nvidia", "NVIDIA RTX", "test"));
    assert!(!engine.shared.gpu_unavailable());
    assert!(engine.shared.has_gpu_failures());
    // Failure history alone is not an active fallback with no model.
    assert!(!engine.is_in_fallback());
    assert_eq!(
        engine.failed_gpu_devices(),
        vec![("nvidia".into(), "NVIDIA RTX".into())]
    );
    assert!(engine.shared.take_device_fallback(&auto_spec()));
    assert!(!engine.shared.take_device_fallback(&auto_spec()));
    engine.retry_gpu("user explicitly selected a device");
    assert!(!engine.is_in_fallback());
    assert!(engine.failed_gpu_devices().is_empty());
    assert_eq!(
        engine.shared.recovery_spec(&auto_spec()).device,
        DeviceSelector::Auto
    );
}

#[test]
fn only_context_failures_require_worker_replacement() {
    assert!(Failure::Backend("device lost".into()).requires_restart());
    assert!(Failure::Crashed("abort".into()).requires_restart());
    assert!(Failure::Hung(Duration::from_secs(1)).requires_restart());
    assert!(!Failure::Remote("backend error: device lost".into()).requires_restart());
    assert!(!Failure::Cancelled.requires_restart());
    assert!(!Failure::Allocation("device lost".into()).requires_restart());
    assert!(!Failure::Spawn(io::Error::other("could not spawn")).requires_restart());
}

fn auto_spec() -> LoadSpec {
    LoadSpec {
        path: "model.gguf".into(),
        backend: Backend::Auto,
        device: DeviceSelector::Auto,
    }
}

#[test]
fn fallback_notification_is_once_per_gpu_failure_and_not_for_explicit_devices() {
    let shared = Shared::default();
    let spec = auto_spec();
    assert!(!shared.take_device_fallback(&spec));
    shared.mark_gpu_unavailable("test failure");
    assert!(!shared.take_device_fallback(&spec.cpu()));
    let pinned = LoadSpec {
        device: DeviceSelector::Index(0),
        ..spec.clone()
    };
    assert!(!shared.take_device_fallback(&pinned));
    assert!(shared.take_device_fallback(&spec));
    // Repeated reports and later model loads do not repeat the notification.
    shared.mark_gpu_unavailable("same failure");
    assert!(!shared.take_device_fallback(&spec));
}

#[test]
fn retrying_gpu_discards_the_old_notification_and_allows_a_new_failure() {
    let engine = EngineSupervisor::new(true);
    let spec = auto_spec();
    engine.shared.mark_gpu_unavailable("old failure");
    engine.retry_gpu("user changed the accelerator");
    assert!(!engine.shared.take_device_fallback(&spec));
    engine.shared.mark_gpu_unavailable("new failure");
    assert!(engine.shared.take_device_fallback(&spec));
}

#[test]
fn deadlines_scale_with_the_audio_above_a_floor() {
    let gpu = Deadlines::for_device(true);
    assert_eq!(gpu, GPU_DEADLINES);
    assert_eq!(gpu.audio(0), gpu.call_floor);
    assert_eq!(gpu.audio(16_000 / 2), gpu.call_floor);
    assert_eq!(gpu.audio(16_000 * 5), Duration::from_secs(50));
    assert_eq!(gpu.run(16_000), gpu.run_floor);
    assert_eq!(gpu.run(16_000 * 10), Duration::from_secs(100));

    let cpu = Deadlines::for_device(false);
    assert_eq!(cpu, CPU_DEADLINES);
    assert_eq!(cpu.audio(16_000), cpu.call_floor);
    assert_eq!(cpu.audio(16_000 * 10), Duration::from_secs(200));
    assert_eq!(cpu.run(16_000 * 5), cpu.run_floor);
    assert_eq!(cpu.run(16_000 * 60), Duration::from_secs(1200));
    // CPU is never stricter than GPU.
    for secs in [0, 1, 5, 30, 120, 600] {
        assert!(cpu.audio(secs * 16_000) >= gpu.audio(secs * 16_000));
        assert!(cpu.run(secs * 16_000) >= gpu.run(secs * 16_000));
    }

    assert_eq!(ms_to_samples(1500), 24_000);
    assert_eq!(ms_to_samples(-5), 0);
}

#[test]
fn device_status_publishes_before_inference_and_ignores_pending_unloads() {
    let changes = Arc::new(Mutex::new(Vec::new()));
    let observed = changes.clone();
    let mut owner = policy_owner();
    owner.shared = Arc::new(Shared {
        gpu_allowed: true,
        on_device_changed: Some(Box::new(move |info| {
            lock(&observed).push(info.device.clone());
        })),
        ..Shared::default()
    });
    assert!(owner.publish_loaded(&loaded(None)));
    assert!(owner.publish_loaded(&loaded(Some("amd"))));
    assert_eq!(*lock(&changes), vec!["CPU", "amd"]);
    assert_eq!(lock(&owner.shared.loaded).as_ref().unwrap().device, "amd");
    // An unload requested before a load finishes must not publish stale status.
    owner.shared.unloads_requested.store(1, Ordering::SeqCst);
    assert!(!owner.publish_loaded(&loaded(Some("nvidia"))));
    assert_eq!(*lock(&changes), vec!["CPU", "amd"]);
}
