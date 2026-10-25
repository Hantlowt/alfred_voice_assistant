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
