use super::*;

#[test]
fn queued_recording_keeps_all_samples_before_finalize() {
    let router = StreamRouter::new();
    let (rx, _) = router.open();
    let prefix: Vec<_> = (0..16640).map(|sample| sample as f32).collect();
    for frame in prefix.chunks(256) {
        router.feed(frame);
    }
    let commands = prepare_stream_commands(rx).expect("recording is still active");
    let route = router.take().unwrap();
    let (reply, _) = mpsc::channel();
    route
        .tx
        .send(StreamCmd::Finalize {
            reply,
            samples: prefix.len(),
        })
        .unwrap();
    let mut received = Vec::new();
    for command in commands {
        match command {
            StreamCmd::Feed(frame) => received.extend(frame),
            StreamCmd::Finalize { samples, .. } => {
                assert_eq!(samples, received.len());
                break;
            }
            StreamCmd::Cancel => panic!("unexpected cancel"),
        }
    }
    assert_eq!(
        received, prefix,
        "the first samples must survive model startup"
    );
}

#[test]
fn recording_finished_before_model_ready_requests_full_recording() {
    let router = StreamRouter::new();
    let (rx, _) = router.open();
    router.feed(&[1.0, 2.0, 3.0]);
    let route = router.take().unwrap();
    let (reply, result) = mpsc::channel();
    route
        .tx
        .send(StreamCmd::Finalize { reply, samples: 3 })
        .unwrap();
    assert!(prepare_stream_commands(rx).is_none());
    assert!(
        result.recv().unwrap().unwrap().is_none(),
        "the caller must batch the canonical full recording"
    );
}

#[test]
#[ignore]
fn queued_short_recording_during_gpu_load_preserves_first_word() {
    use crate::engine_supervisor::{DeviceSelector, LoadSpec};
    use std::path::PathBuf;
    let engine = Arc::new(EngineSupervisor::new(true));
    let model = PathBuf::from(std::env::var_os("HANDY_TEST_MODEL").expect("set HANDY_TEST_MODEL"));
    let audio = crate::audio_toolkit::read_wav_samples(PathBuf::from(
        std::env::var_os("HANDY_TEST_SHORT_WAV").expect("set HANDY_TEST_SHORT_WAV"),
    ))
    .unwrap();
    let cpu = LoadSpec {
        path: model.clone(),
        backend: Backend::Cpu,
        device: DeviceSelector::Auto,
    };
    assert!(!engine.load(cpu).unwrap().on_gpu);
    let gpu = engine
        .devices()
        .unwrap()
        .into_iter()
        .filter(DeviceInfo::is_gpu)
        .max_by_key(|device| device.memory_total)
        .expect("test requires a GPU");
    let router = StreamRouter::new();
    let (rx, _) = router.open();
    let (release, wait) = mpsc::channel();
    let loading_engine = engine.clone();
    let worker = thread::spawn(move || {
        wait.recv().unwrap();
        let info = loading_engine
            .load(LoadSpec {
                path: model,
                backend: Backend::Auto,
                device: DeviceSelector::Key(gpu.key),
            })
            .unwrap();
        assert!(info.on_gpu);
        assert!(
            prepare_stream_commands(rx).is_none(),
            "finished queued audio must use full-clip inference"
        );
    });
    // All of the recording, including its opening word and finalize,
    // arrives while GPU model loading has not finished.
    for frame in audio.chunks(256) {
        router.feed(frame);
    }
    let route = router.take().unwrap();
    let (reply, result) = mpsc::channel();
    route
        .tx
        .send(StreamCmd::Finalize {
            reply,
            samples: audio.len(),
        })
        .unwrap();
    release.send(()).unwrap();
    assert!(result.recv().unwrap().unwrap().is_none());
    worker.join().unwrap();
    let result = engine.transcribe(audio, RunOptions::default()).unwrap();
    assert!(
        result.text.trim().to_lowercase().starts_with("how"),
        "first word missing: {}",
        result.text
    );
    println!(
        "Queued recording survived CPU-to-GPU switch: {}",
        result.text
    );
    engine.unload().wait();
}

fn languages(codes: &[&str]) -> Vec<String> {
    codes.iter().map(|code| (*code).to_string()).collect()
}

#[test]
fn normal_hosts_preserve_every_transcribe_accelerator_setting() {
    for setting in [
        TranscribeAcceleratorSetting::Auto,
        TranscribeAcceleratorSetting::Cpu,
        TranscribeAcceleratorSetting::Gpu,
    ] {
        assert_eq!(effective_transcribe_accelerator(setting, false), setting);
    }
    assert_eq!(
        available_transcribe_accelerators(false),
        ["auto", "cpu", "gpu"]
    );
    assert_eq!(
        select_transcribe_backend_for_host(TranscribeAcceleratorSetting::Auto, false),
        Backend::Auto
    );
    assert_eq!(
        select_transcribe_backend_for_host(TranscribeAcceleratorSetting::Cpu, false),
        Backend::Cpu
    );
    assert_eq!(
        select_transcribe_backend_for_host(TranscribeAcceleratorSetting::Gpu, false),
        Backend::Auto
    );
}

#[test]
fn emulated_x64_on_arm64_forces_every_transcribe_setting_to_cpu() {
    for setting in [
        TranscribeAcceleratorSetting::Auto,
        TranscribeAcceleratorSetting::Cpu,
        TranscribeAcceleratorSetting::Gpu,
    ] {
        assert_eq!(
            effective_transcribe_accelerator(setting, true),
            TranscribeAcceleratorSetting::Cpu
        );
        assert_eq!(
            select_transcribe_backend_for_host(setting, true),
            Backend::Cpu
        );
    }
    assert_eq!(available_transcribe_accelerators(true), ["cpu"]);
}

#[test]
fn optional_text_transform_falls_back_to_raw_text_after_panic() {
    let raw = "原始轉錄。".to_string();
    let result = fail_open_text_transform(raw.clone(), |_| {
        panic!("simulated optional cleanup failure")
    });

    assert_eq!(result, raw);
}

#[test]
fn non_chinese_output_is_never_converted() {
    let settings = AppSettings {
        chinese_script: ChineseScript::Traditional,
        ..Default::default()
    };
    for evidence in [
        OutputLanguageEvidence::ModelDetected("ja".to_string()),
        OutputLanguageEvidence::UserSelected("en".to_string()),
        OutputLanguageEvidence::TranslatedToEnglish,
    ] {
        let result = post_process_transcription_text(
            "学校に行きます".to_string(),
            &settings,
            false,
            &evidence,
            &languages(&["zh", "ja", "en"]),
        );

        assert_eq!(result, "学校に行きます", "{evidence:?}");
    }
}

#[test]
fn auto_detected_chinese_text_is_converted() {
    let settings = AppSettings {
        chinese_script: ChineseScript::Simplified,
        filler_word_removal_enabled: false,
        ..Default::default()
    };
    let result = post_process_transcription_text(
        "我們今天下午一起去學校圖書館看書，然後再去吃晚飯。".to_string(),
        &settings,
        false,
        &OutputLanguageEvidence::Unknown,
        &languages(&["zh", "en", "ja"]),
    );

    assert_eq!(result, "我们今天下午一起去学校图书馆看书，然后再去吃晚饭。");
}

#[test]
fn portuguese_transcription_does_not_use_english_ui_filler_words() {
    let settings = AppSettings {
        app_language: "en".to_string(),
        selected_language: "pt-BR".to_string(),
        ..Default::default()
    };
    let supported = languages(&["en", "pt"]);
    let evidence = resolve_output_language_evidence(&settings, Some("pt"), &supported, false);

    let result = post_process_transcription_text(
        "eu vi um carro".to_string(),
        &settings,
        false,
        &evidence,
        &supported,
    );

    assert_eq!(
        evidence,
        OutputLanguageEvidence::UserSelected("pt".to_string())
    );
    assert_eq!(result, "eu vi um carro");
}

#[test]
fn norwegian_alias_is_recorded_as_user_selected_evidence() {
    let settings = AppSettings {
        selected_language: "no".to_string(),
        ..Default::default()
    };

    let evidence =
        resolve_output_language_evidence(&settings, Some("nb"), &languages(&["nb"]), false);

    assert_eq!(
        evidence,
        OutputLanguageEvidence::UserSelected("nb".to_string())
    );
}

#[test]
fn auto_language_without_detection_skips_gated_filler_removal() {
    let settings = AppSettings {
        selected_language: "auto".to_string(),
        ..Default::default()
    };
    let evidence =
        resolve_output_language_evidence(&settings, None, &languages(&["en", "pt"]), false);

    // Too short for a reliable text detection, so the gated "um" must
    // survive; the universal "uhm" is removed regardless.
    let result = post_process_transcription_text(
        "um uhm ok".to_string(),
        &settings,
        false,
        &evidence,
        &languages(&["en", "pt"]),
    );

    assert_eq!(evidence, OutputLanguageEvidence::Unknown);
    assert_eq!(result, "um ok");
}

#[test]
fn unknown_evidence_with_confident_text_detection_removes_gated_fillers() {
    let settings = AppSettings {
        selected_language: "auto".to_string(),
        ..Default::default()
    };

    let result = post_process_transcription_text(
        "um so the weather forecast said it would probably rain throughout the whole weekend"
            .to_string(),
        &settings,
        false,
        &OutputLanguageEvidence::Unknown,
        &languages(&["en", "pt", "es", "de"]),
    );

    assert_eq!(
        result,
        "so the weather forecast said it would probably rain throughout the whole weekend"
    );
}

#[test]
fn unknown_evidence_with_portuguese_text_preserves_um() {
    let settings = AppSettings {
        selected_language: "auto".to_string(),
        ..Default::default()
    };

    let result = post_process_transcription_text(
        "eu vi um carro na rua ontem de manhã quando fui ao mercado".to_string(),
        &settings,
        false,
        &OutputLanguageEvidence::Unknown,
        &languages(&["en", "pt", "es", "de"]),
    );

    assert_eq!(
        result,
        "eu vi um carro na rua ontem de manhã quando fui ao mercado"
    );
}

#[test]
fn model_detected_language_upgrades_unknown_evidence_only() {
    assert_eq!(
        with_model_detected_language(OutputLanguageEvidence::Unknown, Some("en".to_string())),
        OutputLanguageEvidence::ModelDetected("en".to_string())
    );
    assert_eq!(
        with_model_detected_language(OutputLanguageEvidence::Unknown, Some("auto".to_string())),
        OutputLanguageEvidence::Unknown
    );
    assert_eq!(
        with_model_detected_language(OutputLanguageEvidence::Unknown, None),
        OutputLanguageEvidence::Unknown
    );
    assert_eq!(
        with_model_detected_language(
            OutputLanguageEvidence::UserSelected("pt".to_string()),
            Some("en".to_string())
        ),
        OutputLanguageEvidence::UserSelected("pt".to_string())
    );
}

#[test]
fn auto_language_uses_single_language_model_as_evidence() {
    let settings = AppSettings {
        selected_language: "auto".to_string(),
        ..Default::default()
    };

    let evidence = resolve_output_language_evidence(&settings, None, &languages(&["en"]), false);

    assert_eq!(
        evidence,
        OutputLanguageEvidence::ModelConstrained("en".to_string())
    );
}

#[test]
fn unsupported_explicit_language_uses_model_fallback_as_evidence() {
    let settings = AppSettings {
        selected_language: "pt".to_string(),
        ..Default::default()
    };

    let evidence =
        resolve_output_language_evidence(&settings, Some("en"), &languages(&["en", "de"]), false);

    assert_eq!(
        evidence,
        OutputLanguageEvidence::ModelConstrained("en".to_string())
    );
}

#[test]
fn ignored_user_language_is_not_output_evidence() {
    let settings = AppSettings {
        // Parakeet V3 ignores language hints and auto-detects even when a
        // selection from the previously active model remains persisted.
        selected_language: "en".to_string(),
        ..Default::default()
    };
    let supported = languages(&["en", "de", "pt"]);

    let evidence = resolve_output_language_evidence(&settings, None, &supported, false);
    assert_eq!(evidence, OutputLanguageEvidence::Unknown);

    let result = post_process_transcription_text(
        "eu vi um carro".to_string(),
        &settings,
        false,
        &evidence,
        &supported,
    );
    assert_eq!(result, "eu vi um carro");
}

#[test]
fn unapplied_transcribe_cpp_language_is_not_output_evidence() {
    let settings = AppSettings {
        selected_language: "en".to_string(),
        ..Default::default()
    };
    let supported = languages(&[]);
    let plan = transcribe_cpp_run_plan(false, "en", &supported, false);

    assert_eq!(plan.language, None);
    assert_eq!(
        resolve_output_language_evidence(&settings, plan.language.as_deref(), &supported, false,),
        OutputLanguageEvidence::Unknown
    );
}

#[test]
fn translated_output_is_treated_as_english() {
    let settings = AppSettings {
        selected_language: "pt".to_string(),
        ..Default::default()
    };

    let evidence =
        resolve_output_language_evidence(&settings, Some("pt"), &languages(&["en", "pt"]), true);

    assert_eq!(evidence, OutputLanguageEvidence::TranslatedToEnglish);
}

#[test]
fn transcribe_cpp_run_plan_skips_english_translation() {
    let plan = transcribe_cpp_run_plan(true, "en", &languages(&["en", "es"]), true);

    assert!(matches!(plan.task, Task::Transcribe));
    assert_eq!(plan.language.as_deref(), Some("en"));
    assert_eq!(plan.target_language, None);
}

#[test]
fn transcribe_cpp_run_plan_translates_supported_non_english() {
    let plan = transcribe_cpp_run_plan(true, "es", &languages(&["en", "es"]), true);

    assert!(matches!(plan.task, Task::Translate));
    assert_eq!(plan.language.as_deref(), Some("es"));
    assert_eq!(plan.target_language.as_deref(), Some("en"));
}

#[test]
fn transcribe_cpp_run_plan_requires_model_translation_support() {
    let plan = transcribe_cpp_run_plan(true, "es", &languages(&["en", "es"]), false);

    assert!(matches!(plan.task, Task::Transcribe));
    assert_eq!(plan.language.as_deref(), Some("es"));
    assert_eq!(plan.target_language, None);
}

#[test]
fn fuzzy_correction_is_skipped_exactly_when_vocabulary_was_sent() {
    let settings = AppSettings {
        selected_language: "en".to_string(),
        custom_words: vec!["Handy".to_string()],
        ..Default::default()
    };
    let evidence = OutputLanguageEvidence::UserSelected("en".to_string());
    let supported = languages(&["en"]);

    // The model was already prompted with the term: leave its output alone.
    let prompted = post_process_transcription_text(
        "i use handy every day".to_string(),
        &settings,
        true,
        &evidence,
        &supported,
    );
    assert_eq!(prompted, "i use handy every day");

    // No vocabulary sent: the fuzzy post-correction fixes the casing.
    let corrected = post_process_transcription_text(
        "i use handy every day".to_string(),
        &settings,
        false,
        &evidence,
        &supported,
    );
    assert_eq!(corrected, "i use Handy every day");
}
