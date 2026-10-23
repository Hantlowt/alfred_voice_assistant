from alfred_voice_assistant.app import RecentTranscripts, safety_override_reason, simulate_action


def test_recent_transcripts_normalizes_and_expires() -> None:
    recent = RecentTranscripts(ttl_seconds=3)

    assert recent.add_if_new("Allume la lumière !", now=10)
    assert not recent.add_if_new("  ALLUME la lumière... ", now=11)
    assert recent.add_if_new("Allume la lumière !", now=14)


def test_simulate_action_never_runs_a_real_action() -> None:
    lines: list[str] = []

    simulate_action("play_music", lines.append)
    simulate_action("do_nothing", lines.append)

    assert lines == ["▶️  [simulation] Lecture de la musique."]


def test_safety_override_catches_negation_and_reported_speech() -> None:
    assert safety_override_reason("N'allume surtout pas la lumière", "turn_on_light") == (
        "négation détectée"
    )
    assert safety_override_reason("Paul a dit d'allumer la lumière", "turn_on_light") == (
        "propos rapporté détecté"
    )
    assert safety_override_reason("J'aime écouter de la musique", "play_music") == (
        "commande directe non détectée"
    )
    assert safety_override_reason("Allume la lumière", "turn_on_light") is None
    assert safety_override_reason("Peux-tu lancer de la musique", "play_music") is None
