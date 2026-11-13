from alfred_voice_assistant.cli import build_parser


def test_threshold_is_disabled_by_default() -> None:
    args = build_parser().parse_args([])

    assert args.threshold is None


def test_text_mode_has_a_short_alias() -> None:
    assert build_parser().parse_args(["--text-mode"]).text_mode
    assert build_parser().parse_args(["--text"]).text_mode


def test_quiet_mode_has_a_short_alias() -> None:
    assert build_parser().parse_args(["--quiet"]).quiet
    assert build_parser().parse_args(["-q"]).quiet


def test_microphone_gain_is_neutral_by_default_and_configurable() -> None:
    assert build_parser().parse_args([]).mic_gain == 1.0
    assert build_parser().parse_args(["--mic-gain", "2.5"]).mic_gain == 2.5


def test_kyutai_stt_is_enabled_by_default_and_can_be_disabled() -> None:
    assert build_parser().parse_args([]).kyutai_stt is True
    assert build_parser().parse_args(["--no-kyutai-stt"]).kyutai_stt is False
