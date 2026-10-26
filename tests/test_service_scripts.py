import os
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_service_scripts_have_valid_bash_syntax() -> None:
    subprocess.run(
        ["bash", "-n", "install.sh", "update.sh", "run-service.sh", "service.sh"],
        cwd=ROOT,
        check=True,
    )


def test_service_runner_builds_configured_arguments(tmp_path: Path) -> None:
    config = tmp_path / "service.env"
    config.write_text(
        "\n".join(
            (
                'HOME_ASSISTANT_URL="http://ha.example:8123"',
                'HOME_ASSISTANT_TOKEN_FILE="/tmp/token"',
                'ALFRED_DEVICE="USB Audio"',
                'ALFRED_MIC_GAIN="2.5"',
                'ALFRED_REQUIRE_ALFRED="true"',
                'ALFRED_QUIET="true"',
                'ALFRED_EXECUTE_HOME_ASSISTANT="true"',
                'ALFRED_SPECIALIZATION="true"',
                'ALFRED_INTENT_MODE="flat"',
                'ALFRED_LAYA_BACKEND="auto"',
                'ALFRED_DUPLICATE_WINDOW="4"',
                'ALFRED_THRESHOLD="0.08"',
            )
        )
        + "\n"
    )
    env = {
        **os.environ,
        "ALFRED_CONFIG_FILE": str(config),
        "ALFRED_DRY_RUN": "true",
    }

    completed = subprocess.run(
        ["bash", "run-service.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    arguments = shlex.split(completed.stdout)

    assert arguments[arguments.index("--device") + 1] == "USB Audio"
    assert arguments[arguments.index("--mic-gain") + 1] == "2.5"
    assert arguments[arguments.index("--home-assistant-url") + 1] == (
        "http://ha.example:8123"
    )
    assert "--require-alfred" in arguments
    assert "--quiet" in arguments
    assert "--execute-home-assistant" in arguments
    assert arguments[arguments.index("--threshold") + 1] == "0.08"


def test_update_launcher_resolves_its_symbolic_link(tmp_path: Path) -> None:
    launcher = tmp_path / "alfred-update"
    launcher.symlink_to(ROOT / "update.sh")

    completed = subprocess.run(
        ["bash", str(launcher)],
        env={**os.environ, "ALFRED_UPDATE_DRY_RUN": "true"},
        text=True,
        capture_output=True,
        check=True,
    )

    assert Path(completed.stdout.strip()) == ROOT / "install.sh"
