from tasks.clips.helpers.quality_controls import resolve_quality_controls


def test_high_quality_controls_force_fidelity_mode():
    controls = resolve_quality_controls(
        output_quality="high",
        policy_source_max_height=1080,
    )

    assert controls.effective_source_max_height is None
    assert controls.prefer_fresh_source_download is True
    assert controls.allow_upload_reencode is False
    assert controls.smart_cleanup_crf == 10
    assert controls.smart_cleanup_preset == "slow"


def test_medium_quality_controls_keep_balanced_defaults():
    controls = resolve_quality_controls(
        output_quality="medium",
        policy_source_max_height=1080,
    )

    assert controls.effective_source_max_height == 1080
    assert controls.prefer_fresh_source_download is False
    assert controls.allow_upload_reencode is True
    assert controls.smart_cleanup_crf == 21
    assert controls.smart_cleanup_preset == "medium"


def test_low_quality_controls_keep_policy_height_and_default_cleanup():
    controls = resolve_quality_controls(
        output_quality="low",
        policy_source_max_height=720,
    )

    assert controls.effective_source_max_height == 720
    assert controls.prefer_fresh_source_download is False
    assert controls.allow_upload_reencode is True
    assert controls.smart_cleanup_crf == 21
    assert controls.smart_cleanup_preset == "medium"
