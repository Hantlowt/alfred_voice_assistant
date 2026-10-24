"""Exécution des actions via l'API REST Home Assistant."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True, slots=True)
class ServiceCall:
    domain: str
    service: str
    entity_ids: tuple[str, ...]


ALL_LIGHTS = (
    "light.halogene_socket_1",
    "light.lampe_socket_1",
    "light.luminaire",
    "light.luminaire_2",
    "light.led",
)
SALON_LIGHTS = ("light.halogene_socket_1", "light.luminaire", "light.led")

HOME_ASSISTANT_ACTIONS = {
    "allume les lumières": ServiceCall("light", "turn_on", ALL_LIGHTS),
    "éteins les lumières": ServiceCall("light", "turn_off", ALL_LIGHTS),
    "allume l'halogène": ServiceCall("light", "turn_on", ("light.halogene_socket_1",)),
    "éteins l'halogène": ServiceCall("light", "turn_off", ("light.halogene_socket_1",)),
    "allume le salon": ServiceCall("light", "turn_on", SALON_LIGHTS),
    "éteins le salon": ServiceCall("light", "turn_off", SALON_LIGHTS),
    "éteins le luminaire du salon": ServiceCall("light", "turn_off", ("light.luminaire",)),
    "allume la salle à manger": ServiceCall("light", "turn_on", ("light.luminaire_2",)),
    "éteins la salle à manger": ServiceCall("light", "turn_off", ("light.luminaire_2",)),
    "allume les leds": ServiceCall("light", "turn_on", ("light.led",)),
    "éteins les leds": ServiceCall("light", "turn_off", ("light.led",)),
    "allume la clim": ServiceCall("scene", "turn_on", ("scene.allumer_climatiseur",)),
    "éteins la clim": ServiceCall("scene", "turn_on", ("scene.eteindre_le_climatiseur",)),
}


class ActionExecutor(Protocol):
    def execute(self, action: str) -> str: ...


class SimulationExecutor:
    def execute(self, action: str) -> str:
        return f"🏠 [simulation uniquement] {action}"


class HomeAssistantExecutor:
    def __init__(self, base_url: str, token_file: Path, timeout: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        try:
            self._token = token_file.expanduser().read_text(encoding="utf-8").strip()
        except OSError as error:
            raise RuntimeError(f"Impossible de lire le token Home Assistant : {token_file}") from error
        if not self._token:
            raise RuntimeError(f"Le fichier de token Home Assistant est vide : {token_file}")

    def check_connection(self) -> None:
        response = self._request("GET", "/api/")
        if response.get("message") != "API running.":
            raise RuntimeError("Réponse inattendue de Home Assistant.")

    def execute(self, action: str) -> str:
        try:
            call = HOME_ASSISTANT_ACTIONS[action]
        except KeyError as error:
            raise RuntimeError(f"Action Home Assistant non configurée : {action}") from error
        body = {"entity_id": list(call.entity_ids)}
        changed = self._request(
            "POST",
            f"/api/services/{call.domain}/{call.service}",
            body,
        )
        changed_ids = [item.get("entity_id", "?") for item in changed if isinstance(item, dict)]
        detail = ", ".join(changed_ids) if changed_ids else "appel accepté, aucun état modifié"
        return f"🏠 Home Assistant : {action} — {detail}"

    def _request(self, method: str, path: str, body: dict | None = None):
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = Request(
            self._base_url + path,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                return json.load(response)
        except HTTPError as error:
            raise RuntimeError(f"Home Assistant a répondu HTTP {error.code} pour {path}") from error
        except (URLError, TimeoutError) as error:
            raise RuntimeError(f"Home Assistant est inaccessible : {error}") from error
