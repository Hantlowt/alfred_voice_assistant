# Alfred Voice Assistant — prototype local

Prototype local sans wake word et sans entraînement : Vosk transcrit le microphone en continu, puis
Laya classe chaque phrase finalisée dans une liste fermée d'actions. Le backend est MLX sur Mac
Apple Silicon et PyTorch CPU sur Linux x86-64. Les actions sont simulées par défaut ; leur exécution
réelle via Home Assistant doit être activée explicitement.

## Prérequis

- Mac Apple Silicon sous macOS 14+, ou Linux x86-64 (Lubuntu/Debian/Ubuntu) ;
- Python 3.11 à 3.13 ;
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) ;
- accès Internet au **premier lancement seulement** (environ 41 Mo pour Vosk et 650 Mo pour Laya) ;
- accès au microphone et, pour l'exécution réelle, au serveur Home Assistant ;
- sous Linux/PyTorch CPU : prévoir environ 8 Go de RAM et 10 Go de disque disponibles.

## Installation automatique sur Lubuntu

Après publication de cette version sur la branche `main` :

```bash
curl -fsSL https://raw.githubusercontent.com/Hantlowt/alfred_voice_assistant/main/install.sh | bash
```

Le script installe PortAudio/ALSA, Git, `uv`, clone le dépôt dans
`~/.local/share/alfred_voice_assistant`, installe le backend Laya PyTorch CPU, demande le token Home
Assistant sans l'afficher puis le sauvegarde avec les permissions `600` dans
`~/.config/alfred_voice_assistant/home_assistant_token`.

Ensuite :

```bash
# Repérer l'index du microphone
~/.local/bin/alfred-local --list-devices

# Démarrer avec le microphone 2 et les appels Home Assistant réels
~/.local/bin/alfred-local --device 2 --require-alfred --execute-home-assistant

# Le fragment unique d'un nom fonctionne aussi
~/.local/bin/alfred-local --device "USB Audio" --require-alfred --execute-home-assistant
```

## Lancement

Depuis la racine du dépôt :

```bash
uv run alfred-local
```

`uv` crée l'environnement et installe les dépendances déclarées dans `pyproject.toml`. Au premier
démarrage, l'application télécharge automatiquement :

- `vosk-model-small-fr-0.22` depuis le site officiel Vosk ;
- `aac6fef/laya-multilingual-mlx` depuis Hugging Face.

Sous Linux, le second modèle est remplacé automatiquement par le checkpoint multilingue PyTorch
`convaiinnovations/laya`.

Les modèles sont stockés par défaut dans `~/.cache/alfred-voice-assistant`. Tous les lancements
suivants chargent exclusivement ces chemins locaux et peuvent donc fonctionner sans réseau.

Pour quitter : `Ctrl+C`.

## Configuration

```bash
# Seuil à 80 %, anti-doublon de 5 secondes
uv run alfred-local --threshold 0.80 --duplicate-window 5

# Exiger « alfred » dans la phrase courante ou dans celle juste avant
uv run alfred-local --require-alfred

# Désactiver explicitement cette exigence (c'est aussi la valeur par défaut)
uv run alfred-local --no-require-alfred

# Un seul appel : choix direct parmi les 14 actions (mode par défaut)
uv run alfred-local --intent-mode flat

# Deux appels : opération, puis objet
uv run alfred-local --intent-mode factorized

# Voir les microphones, puis en sélectionner un
uv run alfred-local --list-devices
uv run alfred-local --device 2

# Choisir l'emplacement des modèles
uv run alfred-local --models-dir .models

# Exécuter réellement dans Home Assistant (simulation par défaut)
uv run alfred-local --execute-home-assistant \
  --home-assistant-url http://100.123.216.97:8123 \
  --ha-token-file .secrets/home_assistant_token

# Forcer un backend ; auto est recommandé
uv run alfred-local --laya-backend auto
```

La variable `ALFRED_MODELS_DIR` peut aussi définir l'emplacement des modèles. Les options de ligne
de commande ont priorité.

## Home Assistant

Les appels utilisent `POST /api/services/<domain>/<service>` avec le token lu depuis un fichier ; le
token n'est jamais placé dans le code ou les arguments du processus. Le mapping vérifié sur cette
instance est :

| Action | Appel Home Assistant |
|---|---|
| Toutes les lumières | `light.turn_on/off` sur les cinq entités `light.*` |
| Halogène | `light.halogene_socket_1` |
| Salon | `light.halogene_socket_1`, `light.luminaire`, `light.led` |
| Luminaire du salon | `light.luminaire` |
| Salle à manger | `light.luminaire_2` |
| LEDs | `light.led` |
| Climatisation ON | activation de `scene.allumer_climatiseur` |
| Climatisation OFF | activation de `scene.eteindre_le_climatiseur` |

Lors de l'inventaire, `light.luminaire_2` (salle à manger) était déclaré mais dans l'état
`unavailable`. Le mapping est prêt, mais Home Assistant ne pourra pas piloter cet appareil tant que
son intégration ne l'aura pas rendu disponible.

Sans `--execute-home-assistant`, aucun appel réseau de commande n'est effectué. Au démarrage du mode
réel, la connexion et le token sont vérifiés avant d'ouvrir le microphone. L'adresse peut aussi être
fournie avec `HOME_ASSISTANT_URL`, et le fichier avec `HOME_ASSISTANT_TOKEN_FILE`.

## Comportement

Vosk affiche les transcriptions partielles sur une ligne temporaire. Chaque phrase finalisée est
placée dans une file non bloquante et traitée par un worker Laya distinct, afin que l'inférence ne
bloque jamais le callback audio. Le terminal affiche ensuite :

- la transcription finale ;
- l'intention retenue ;
- toutes les probabilités renvoyées par Laya ;
- la latence de l'appel Laya (chargement du modèle exclu).

Avec `--require-alfred`, une action n'est retenue que si Vosk a transcrit le mot exact `alfred` dans
la phrase courante ou dans la phrase finalisée immédiatement précédente. Cela permet aussi de dire
« Alfred », puis la commande dans une seconde phrase. L'autorisation ne se propage pas à une
troisième phrase. Chaque résultat indique clairement l'état de ce contrôle.

Les seules actions proposées à Laya sont :

- `do_nothing` ;
- `éteins les lumières` ;
- `allume les lumières` ;
- `allume l'halogène` ;
- `allume le salon` ;
- `allume la salle à manger` ;
- `éteins l'halogène` ;
- `éteins le luminaire du salon` ;
- `éteins le salon` ;
- `éteins la salle à manger` ;
- `allume les leds` ;
- `éteins les leds` ;
- `allume la clim` ;
- `éteins la clim`.

Le mode `flat` effectue un seul appel Laya avec cette liste complète. Le mode `factorized` effectue
réellement deux appels successifs sur la même transcription : le premier choisit `allumer`,
`éteindre` ou `do_nothing`, puis le second choisit l'objet (`les lumières`, `l'halogène`, `le salon`,
`la salle à manger`, `le luminaire du salon`, `les leds`, `la clim` ou `do_nothing`). Les deux jeux
de probabilités et leurs latences sont affichés séparément. La confiance utilisée par le seuil est
la plus faible des deux réponses sélectionnées : les deux doivent donc être suffisamment sûres.
Une combinaison absente de la liste d'actions, comme allumer le luminaire spécifique du salon, est
ramenée à `do_nothing`.

Avant l'anti-doublon et l'inférence Laya, une normalisation ciblée corrige les erreurs Vosk connues
`la le jeune` et `eugène` en `halogène`, `climat` en `clim`, ainsi que `est un` en `éteins`. La
transcription originale reste affichée, puis le terminal montre explicitement la transformation avec
`Normalisation ASR`. Cette correction s'applique aux deux modes d'intention.

Une intention sous le seuil est remplacée par `do_nothing`. Les mêmes phrases répétées pendant la
fenêtre anti-doublon sont ignorées. Pour chaque phrase, le terminal distingue l'action proposée par
Laya, les probabilités, l'état du contrôle Alfred, l'**ACTION FINALE** et si son exécution est
autorisée. Le message suivant précise s'il s'agissait d'une simulation ou d'un appel Home Assistant.

Les instructions données au modèle imposent `do_nothing` pour les conversations ordinaires, les
propos rapportés, les négations, les phrases qui ne semblent pas adressées à l'assistant et tout cas
ambigu. Ce garde-fou reste probabiliste : le seuil est une protection supplémentaire, pas une
garantie de compréhension. Comme le modèle générique peut malgré tout confondre une négation avec
l'action opposée, un filtre local conservateur force également les négations et les propos rapportés
explicites à `do_nothing`, après l'inférence Laya. Les probabilités brutes restent affichées.
Le même filtre exige une forme de commande directe ou de requête polie avant toute action ; une
simple mention d'un équipement ne suffit pas.

## Utiliser plus tard un export Laya Studio

Le contrat à conserver est `IntentEngine.predict(transcript) -> IntentResult`, défini dans
`src/alfred_voice_assistant/intent.py`. `LayaIntentEngine` isole les runtimes MLX et PyTorch. Pour
employer un modèle spécialisé exporté depuis
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
[checkpoint multilingue](https://huggingface.co/aac6fef/laya-multilingual-mlx),
[API REST Home Assistant](https://developers.home-assistant.io/docs/api/rest/) et
[Laya PyTorch](https://github.com/NandhaKishorM/laya).
