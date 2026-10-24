import json
from pathlib import Path
from unittest.mock import patch

from alfred_voice_assistant.home_assistant import (
    HOME_ASSISTANT_ACTIONS,
    HomeAssistantExecutor,
)
from alfred_voice_assistant.intent import ACTIONS


class FakeResponse:
    def __init__(self, payload) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def test_every_executable_action_has_a_home_assistant_mapping() -> None:
    assert set(HOME_ASSISTANT_ACTIONS) == set(ACTIONS) - {"do_nothing"}


def test_home_assistant_uses_the_expected_service_and_entity(tmp_path: Path) -> None:
    token_file = tmp_path / "token"
    token_file.write_text("secret-test-token")
    executor = HomeAssistantExecutor("http://ha.test:8123/", token_file)

    with patch(
        "alfred_voice_assistant.home_assistant.urlopen",
        return_value=FakeResponse([{"entity_id": "light.led", "state": "on"}]),
    ) as mocked:
        message = executor.execute("allume les leds")

    request = mocked.call_args.args[0]
    assert request.full_url == "http://ha.test:8123/api/services/light/turn_on"
    assert json.loads(request.data) == {"entity_id": ["light.led"]}
    assert request.headers["Authorization"] == "Bearer secret-test-token"
    assert "light.led" in message
