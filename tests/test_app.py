from alfred_voice_assistant.app import (
    AlfredGate,
    RecentTranscripts,
    normalize_asr_transcript,
    safety_override_reason,
)
from alfred_voice_assistant.home_assistant import SimulationExecutor


def test_recent_transcripts_normalizes_and_expires() -> None:
    recent = RecentTranscripts(ttl_seconds=3)

    assert recent.add_if_new("Allume la lumière !", now=10)
    assert not recent.add_if_new("  ALLUME la lumière... ", now=11)
    assert recent.add_if_new("Allume la lumière !", now=14)


def test_simulate_action_never_runs_a_real_action() -> None:
    executor = SimulationExecutor()

    assert executor.execute("allume les leds") == "🏠 [simulation uniquement] allume les leds"


def test_safety_override_catches_negation_and_reported_speech() -> None:
    assert safety_override_reason("N'allume surtout pas la lumière", "allume les lumières") == (
        "négation détectée"
    )
    assert safety_override_reason("Paul a dit d'allumer la lumière", "allume les lumières") == (
        "propos rapporté détecté"
    )
    assert safety_override_reason("J'aime quand la clim fonctionne", "allume la clim") == (
        "commande directe non détectée"
    )
    assert safety_override_reason(
        "Alfred, allumez les lumières s'il vous plaît", "allume les lumières"
    ) is None
    assert safety_override_reason("Pouvez-vous éteindre la clim", "éteins la clim") is None


def test_alfred_gate_accepts_current_or_immediately_previous_phrase() -> None:
    gate = AlfredGate(required=True)

    assert not gate.evaluate("bonjour tout le monde").allowed
    assert gate.evaluate("Alfred").allowed
    previous = gate.evaluate("allume les lumières")
    assert previous.allowed
    assert "phrase précédente" in previous.status
    assert not gate.evaluate("allume les leds").allowed


def test_alfred_gate_can_be_disabled() -> None:
    gate = AlfredGate(required=False)

    assert gate.evaluate("allume les lumières").allowed
    assert gate.evaluate("éteins la clim").allowed


def test_known_halogen_asr_errors_are_normalized_before_intent_detection() -> None:
    assert normalize_asr_transcript("Alfred allume la le jeune") == "Alfred allume halogène"
    assert normalize_asr_transcript("éteins Eugène s'il te plaît") == (
        "éteins halogène s'il te plaît"
    )
    assert normalize_asr_transcript("Alfred allume le climat") == "Alfred allume la clim"
    assert normalize_asr_transcript("Alfred est un le salon") == "Alfred éteins le salon"
    assert normalize_asr_transcript("allume l'halogène") == "allume l'halogène"
