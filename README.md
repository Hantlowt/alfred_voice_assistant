# Alfred Voice Assistant — prototype local

Prototype macOS/Apple Silicon sans wake word et sans entraînement : Vosk transcrit le microphone en
continu, puis Laya Multilingual MLX classe chaque phrase finalisée dans une liste fermée d'actions.
Aucune action réelle n'est exécutée ; elles sont uniquement simulées dans le terminal.

## Prérequis

- Mac Apple Silicon, macOS 14 ou plus récent ;
- Python 3.11 à 3.13 ;
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) ;
- accès Internet au **premier lancement seulement** (environ 41 Mo pour Vosk et 650 Mo pour Laya) ;
- autorisation macOS d'accéder au microphone.

## Lancement

Depuis la racine du dépôt :

```bash
uv run alfred-local
```

`uv` crée l'environnement et installe les dépendances déclarées dans `pyproject.toml`. Au premier
démarrage, l'application télécharge automatiquement :

- `vosk-model-small-fr-0.22` depuis le site officiel Vosk ;
- `aac6fef/laya-multilingual-mlx` depuis Hugging Face.

Les modèles sont stockés par défaut dans `~/.cache/alfred-voice-assistant`. Tous les lancements
suivants chargent exclusivement ces chemins locaux et peuvent donc fonctionner sans réseau.

Pour quitter : `Ctrl+C`.

## Configuration

```bash
# Seuil à 80 %, anti-doublon de 5 secondes
uv run alfred-local --threshold 0.80 --duplicate-window 5

# Voir les microphones, puis en sélectionner un
uv run alfred-local --list-devices
uv run alfred-local --device 2

# Choisir l'emplacement des modèles
uv run alfred-local --models-dir .models
```

La variable `ALFRED_MODELS_DIR` peut aussi définir l'emplacement des modèles. Les options de ligne
de commande ont priorité.

## Comportement

Vosk affiche les transcriptions partielles sur une ligne temporaire. Chaque phrase finalisée est
placée dans une file non bloquante et traitée par un worker Laya distinct, afin que l'inférence ne
bloque jamais le callback audio. Le terminal affiche ensuite :

- la transcription finale ;
- l'intention retenue ;
- toutes les probabilités renvoyées par Laya ;
- la latence de l'appel Laya (chargement du modèle exclu).

Les actions possibles sont `turn_on_light`, `turn_off_light`, `get_weather`, `play_music`,
`stop_music` et `do_nothing`. Une intention sous le seuil est remplacée par `do_nothing`. Les mêmes
phrases répétées pendant la fenêtre anti-doublon sont ignorées.

Les instructions données au modèle imposent `do_nothing` pour les conversations ordinaires, les
propos rapportés, les négations, les phrases qui ne semblent pas adressées à l'assistant et tout cas
ambigu. Ce garde-fou reste probabiliste : le seuil est une protection supplémentaire, pas une
garantie de compréhension. Comme le modèle générique peut malgré tout confondre une négation avec
l'action opposée, un filtre local conservateur force également les négations et les propos rapportés
explicites à `do_nothing`, après l'inférence Laya. Les probabilités brutes restent affichées.
Le même filtre exige une forme de commande directe ou de requête polie avant toute simulation ; une
simple mention de lumière, de météo ou de musique ne suffit pas.

## Utiliser plus tard un export Laya Studio

Le contrat à conserver est `IntentEngine.predict(transcript) -> IntentResult`, défini dans
`src/alfred_voice_assistant/intent.py`. `LayaMlxIntentEngine` est le seul adaptateur qui dépend de
`laya-mlx`. Pour employer un modèle spécialisé exporté depuis
[Laya Studio](https://github.com/Hantlowt/laya-studio), ajoutez une autre implémentation de cette
interface puis instanciez-la dans `cli.py` ; la capture Vosk et le worker asynchrone ne changent pas.

## Développement

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

Références API : [exemple microphone officiel Vosk](https://github.com/alphacep/vosk-api/blob/master/python/example/test_microphone.py),
[modèle français Vosk](https://alphacephei.com/vosk/models),
[API laya-mlx](https://github.com/mizorewww/laya-mlx) et
[checkpoint multilingue](https://huggingface.co/aac6fef/laya-multilingual-mlx).
