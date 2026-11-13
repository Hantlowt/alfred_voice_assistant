# Alfred Voice Assistant — prototype local

Prototype local sans wake word et sans entraînement : Kyutai STT transcrit le microphone par défaut, puis
Laya classe chaque phrase finalisée dans une liste fermée d'actions. Une spécialisation légère créée
avec Laya Studio est appliquée par défaut au mode `flat`, sans modifier ni recopier les poids du
modèle. Le backend est MLX sur Mac Apple Silicon et PyTorch CPU sur Linux x86-64. Les actions sont
simulées par défaut ; leur exécution réelle via Home Assistant doit être activée explicitement.

## Prérequis

- Mac Apple Silicon sous macOS 14+, ou Linux x86-64 (Lubuntu/Debian/Ubuntu) ;
- Python 3.11 à 3.13 ;
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) ;
- accès Internet à l'installation ou la mise à jour qui ajoute un modèle ;
- accès au microphone et, pour l'exécution réelle, au serveur Home Assistant ;
- sous Linux/PyTorch CPU : prévoir au moins 8 Go de RAM et 13 Go de disque disponibles. Les poids
  Kyutai représentent à eux seuls environ 2,4 Go et l'inférence sera nettement plus rapide avec CUDA.

## Installation automatique sur Lubuntu

Après publication de cette version sur la branche `main` :

```bash
curl -fsSL https://raw.githubusercontent.com/Hantlowt/alfred_voice_assistant/main/install.sh | bash
```

Le script installe PortAudio/ALSA, Git, `uv`, clone le dépôt dans
`~/.local/share/alfred_voice_assistant`, installe le backend Laya PyTorch CPU, puis demande le token,
l'URL Home Assistant et le microphone. Il installe enfin un service système `systemd`, le démarre et
l'active pour les démarrages suivants. Le token est sauvegardé avec les permissions `600` dans
`~/.config/alfred_voice_assistant/home_assistant_token`.

Une installation existante se met ensuite à jour avec :

```bash
alfred-update
```

Cette commande effectue un `git pull --ff-only`, resynchronise exactement `uv.lock`, conserve le
token et la configuration, télécharge les éventuels nouveaux modèles avant le redémarrage, recrée
les lanceurs puis redémarre le service. Relancer la commande
`curl` initiale produit le même résultat.

Ensuite :

```bash
# Repérer l'index du microphone
~/.local/bin/alfred-local --list-devices

# Démarrer avec le microphone 2 et les appels Home Assistant réels
~/.local/bin/alfred-local --device 2 --require-alfred --execute-home-assistant

# Le fragment unique d'un nom fonctionne aussi
~/.local/bin/alfred-local --device "USB Audio" --require-alfred --execute-home-assistant

# Amplifier par deux une voix distante avant de l'envoyer à Vosk
~/.local/bin/alfred-local --device "USB Audio" --mic-gain 2.0 --require-alfred
```

### Service en arrière-plan

La configuration persistante se trouve dans
`~/.config/alfred_voice_assistant/service.env`. L'installation active par défaut l'exécution réelle
Home Assistant, le contrôle `alfred`, la spécialisation et le mode silencieux. Pour modifier l'URL,
le microphone ou un autre paramètre puis redémarrer automatiquement :

```bash
alfred-service edit
```

Commandes d'exploitation :

```bash
alfred-service status   # état et dernier éventuel message d'erreur
alfred-service logs     # suivre les logs systemd
alfred-service restart  # redémarrer après une intervention
alfred-service stop
alfred-service start
```

Pour une voix distante, réglez par exemple `ALFRED_MIC_GAIN=2.0` dans `service.env`. La valeur
`1.0` laisse le signal intact. Un gain trop fort amplifie le bruit et provoque de l'écrêtement ;
commencez à `2.0`, puis essayez `3.0` seulement si nécessaire. Le gain matériel du microphone,
quand il est disponible dans `alsamixer`, donne généralement un meilleur résultat.

Le service `alfred-voice-assistant.service` est rattaché à `multi-user.target`, fonctionne sous le
compte qui a lancé l'installation et redémarre automatiquement après un échec. Ne lancez donc pas
la commande d'installation avec `sudo` : le script ne l'utilise que pour enregistrer le service.
En mode silencieux, le journal reste vide pendant le fonctionnement normal ; les erreurs fatales y
restent disponibles.

La capture possède aussi son propre watchdog. Si ALSA/PortAudio cesse de fournir des blocs audio
pendant 5 secondes (par exemple après un `xrun` ou une brève déconnexion USB), Alfred ferme puis
rouvre le microphone toutes les 2 secondes. Les modèles de transcription et Laya restent chargés : la reprise
ne provoque donc pas un redémarrage complet du service.

## Lancement

Depuis la racine du dépôt :

```bash
uv run alfred-local
```

`uv` crée l'environnement et installe les dépendances déclarées dans `pyproject.toml`. Au premier
démarrage, l'application télécharge automatiquement :

- `kyutai/stt-1b-en_fr` depuis Hugging Face (transcription par défaut) ;
- `aac6fef/laya-multilingual-mlx` depuis Hugging Face.

Sous Linux, le second modèle est remplacé automatiquement par le checkpoint multilingue PyTorch
`convaiinnovations/laya-multilingual`.

Les modèles sont stockés par défaut dans `~/.cache/alfred-voice-assistant`. Tous les lancements
suivants chargent exclusivement ces chemins locaux et peuvent donc fonctionner sans réseau.

Pour quitter : `Ctrl+C`.

## Configuration

```bash
# Le seuil est désactivé par défaut ; le réactiver explicitement à 8 %
uv run alfred-local --threshold 0.08 --duplicate-window 5

# Debug sans microphone : saisir une phrase puis appuyer sur Entrée
uv run alfred-local --text-mode

# Revenir à l'ancien moteur Vosk
uv run alfred-local --no-kyutai-stt

# Précharger explicitement tous les modèles sans démarrer l'assistant
uv run alfred-local --download-models

# Déploiement silencieux : aucune transcription, probabilité ou confirmation normale
uv run alfred-local --quiet --execute-home-assistant

# Même mode, avec Alfred obligatoire et de vrais appels Home Assistant
uv run alfred-local --text --require-alfred --execute-home-assistant

# Exiger « alfred » dans la phrase courante ou dans celle juste avant
uv run alfred-local --require-alfred

# Conserver malgré tout do_nothing parmi les choix proposés à Laya
uv run alfred-local --require-alfred --no-remove-do-nothing-when-alfred

# Désactiver explicitement cette exigence (c'est aussi la valeur par défaut)
uv run alfred-local --no-require-alfred

# Un seul appel : choix direct parmi les 15 choix (14 commandes + do_nothing, mode par défaut)
uv run alfred-local --intent-mode flat

# Deux appels : opération, puis objet
uv run alfred-local --intent-mode factorized

# Comparer au Laya vanilla, sans transformation latente
uv run alfred-local --intent-mode flat --no-specialization

# Charger un autre export compatible de Laya Studio
uv run alfred-local --specialization-dir ./mon-export-laya

# Voir les microphones, puis en sélectionner un
uv run alfred-local --list-devices
uv run alfred-local --device 2

# Amplifier le signal du microphone avant Vosk
uv run alfred-local --device 2 --mic-gain 2.0

# Choisir l'emplacement des modèles
uv run alfred-local --models-dir .models

# Exécuter réellement dans Home Assistant (simulation par défaut)
uv run alfred-local --execute-home-assistant \
  --home-assistant-url http://100.123.216.97:8123 \
  --ha-token-file .secrets/home_assistant_token

# Forcer un backend ; auto est recommandé
uv run alfred-local --laya-backend auto
```

La variable `ALFRED_MODELS_DIR` peut aussi définir l'emplacement des modèles. Pour le service,
`ALFRED_KYUTAI_STT=true` active Kyutai et `ALFRED_REMOVE_DO_NOTHING_WHEN_ALFRED=true` retire
`do_nothing` des choix Laya lorsque `ALFRED_REQUIRE_ALFRED=true` ; ces deux options sont activées par
défaut. Les options de ligne de commande ont priorité.

## Home Assistant

Les appels utilisent `POST /api/services/<domain>/<service>` avec le token lu depuis un fichier ; le
token n'est jamais placé dans le code ou les arguments du processus. Le mapping vérifié sur cette
instance est :

| Action | Appel Home Assistant |
|---|---|
| Allumer les lumières | `light.turn_on` sur `light.halogene_socket_1` et `light.luminaire_2` |
| Éteindre les lumières | `light.turn_off` sur l'halogène, les deux luminaires et les LEDs |
| Halogène | `light.halogene_socket_1` |
| Allumer le salon | `light.halogene_socket_1` uniquement |
| Éteindre le salon | `light.halogene_socket_1`, `light.luminaire`, `light.led` |
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

Kyutai reçoit désormais les blocs audio en continu et produit ses tokens pendant que la personne
parle, au lieu de recommencer l'inférence après chaque phrase. Après environ 320 ms de silence local,
Alfred applique le « flush trick » décrit par Kyutai : il injecte immédiatement le silence nécessaire
pour vider les 500 ms de décalage du modèle, sans les attendre en temps réel. Le texte partiel reste
visible et la transcription finale est aussitôt transmise au contrôle Alfred puis à Laya. Avec
`--no-kyutai-stt`, Vosk affiche aussi ses transcriptions partielles sur une ligne temporaire. Avec
`--text-mode` (ou `--text`), aucune capture audio ni aucun modèle STT n'est chargé : chaque ligne validée par Entrée
est envoyée au même pipeline que les phrases vocales. Chaque phrase finalisée est
placée dans une file non bloquante et traitée par un worker Laya distinct, afin que l'inférence ne
bloque jamais le callback audio. Le terminal affiche ensuite :

- la transcription finale ;
- l'intention retenue ;
- toutes les probabilités renvoyées par Laya ;
- la latence de l'appel Laya (chargement du modèle exclu).

Avec `--require-alfred`, une action n'est retenue que si la transcription ou le texte contient le
mot exact `alfred` dans la phrase courante ou dans la phrase immédiatement précédente. Cela permet
aussi de dire ou saisir « Alfred », puis la commande dans une seconde phrase. L'autorisation ne se
propage pas à une troisième phrase. Chaque résultat indique clairement l'état de ce contrôle.
Par défaut, ce mode retire également `do_nothing` de l'appel Laya afin que le modèle départage les
14 actions disponibles après détection d'Alfred. `--no-remove-do-nothing-when-alfred` rétablit
l'ancien ensemble de choix.

Les seules actions proposées à Laya sont :

- `do_nothing` ;
- `éteins les lumières` ;
- `allume les lumières` ;
- `allume l'halogène` ;
- `allume le salon` ;
- `allume la salle à manger` ;
- `allume le luminaire du salon` ;
- `éteins l'halogène` ;
- `éteins le luminaire du salon` ;
- `éteins le salon` ;
- `éteins la salle à manger` ;
- `allume les leds` ;
- `éteins les leds` ;
- `allume la clim` ;
- `éteins la clim`.

Le mode `flat` encode la phrase une seule fois puis applique la transformation résiduelle et les
prototypes exportés par Laya Studio. `--no-specialization` rétablit l'appel Laya vanilla avec la
liste complète. Le mode `factorized` effectue réellement deux appels successifs sur la même
transcription : le premier choisit `allumer`,
`éteindre` ou `do_nothing`, puis le second choisit l'objet (`les lumières`, `l'halogène`, `le salon`,
`la salle à manger`, `le luminaire du salon`, `les leds`, `la clim` ou `do_nothing`). Les deux jeux
de probabilités et leurs latences sont affichés séparément. La confiance utilisée par le seuil est
la plus faible des deux réponses sélectionnées : les deux doivent donc être suffisamment sûres.
Toute combinaison opération/objet non déclarée est ramenée à `do_nothing`.

Avant l'anti-doublon et l'inférence Laya, une normalisation ciblée corrige les erreurs Vosk connues
`la le jeune` et `eugène` en `halogène`, `est la logement` ou `éteint la Lorraine` en
`éteins l'halogène`, `climat` ou `clime` en `clim`, `les aides` ou `les l'aide` en `les leds`,
ainsi que `est un`, `et un`, `étant`, `éteint` ou `éteindre` en `éteins` et `à l'homme` en `allume`. Les
corrections de verbes ne s'appliquent que devant l'un des sept objets pris en charge, afin de ne pas
réécrire une conversation ordinaire comme « c'est un très beau salon ». La transcription originale
reste affichée, puis le terminal montre explicitement la transformation avec `Normalisation ASR`.
Cette correction s'applique aux deux modes d'intention.

Après contrôle du mot-clé `alfred`, seule la partie qui le suit est envoyée à Laya : cela évite que
la casse du prénom ou une amorce comme « attends » perturbe l'embedding. La décision de Laya n'est
plus remplacée ni bloquée par des expressions régulières après l'inférence.

Avec `--quiet` (ou `-q`), le programme ne produit aucune sortie normale : transcriptions,
probabilités, chargement des modèles et confirmations Home Assistant sont masqués. Une erreur fatale
reste affichée afin que le service puisse être diagnostiqué.

Le seuil de confiance est désactivé par défaut : la meilleure intention proposée n'est donc pas
rejetée en fonction de son pourcentage. `--threshold VALEUR` permet de réactiver ce filtre entre 0
et 1 ; une intention sous ce seuil est alors remplacée par `do_nothing`. Le contrôle Alfred reste
actif indépendamment du seuil. Les mêmes phrases
répétées pendant la
fenêtre anti-doublon sont ignorées. Pour chaque phrase, le terminal distingue l'action proposée par
Laya, les probabilités, l'état du contrôle Alfred, l'**ACTION FINALE** et si son exécution est
autorisée. Le message suivant précise s'il s'agissait d'une simulation ou d'un appel Home Assistant.

Lorsque `do_nothing` fait partie des choix, les instructions données au modèle le demandent pour les
conversations ordinaires, les propos rapportés, les négations et les cas ambigus. Il n'existe plus
de filtre regex après Laya : l'action choisie par le modèle est directement soumise au seuil puis au
contrôle Alfred.

## Spécialisation Laya Studio

Le fichier `specialization/laya_studio_task.yaml` définit la politique à 15 classes (14 commandes et
`do_nothing`). Laya Studio a généré 150 exemples d'ajustement et 75 de validation. Ils sont complétés
par 64 exemples uniques issus de `specialization/curated_examples.yaml`, qui ciblent notamment les
commandes canoniques, les requêtes polies et les confusions phonétiques. Sur les 148 cas cachés
restant après suppression des recouvrements exacts, la transformation résiduelle atteint **73,65 %**
sur MLX comme sur PyTorch, contre **57,43 %** pour Laya vanilla, soit **+16,22 points**. Elle obtient
également 100 % sur les 14 commandes canoniques et 97,33 % sur les 75 variantes ciblées. Les
résultats, les graines et le réglage de régularisation sont dans `specialization/benchmark.json`.

Les deux exports d'environ 2,3 Mio se trouvent sous
`src/alfred_voice_assistant/specializations/`. Ils contiennent uniquement une matrice, des
prototypes, le manifeste et les sondes de fidélité ; les modèles de base restent téléchargés au
premier lancement. Chaque export a passé le contrôle de rechargement en processus neuf de Laya
Studio.

Un artefact n'est **pas universel entre MLX et PyTorch** : Laya Studio le verrouille au backend, à
l'identifiant du checkpoint et à sa révision. Les deux transformations ont donc été ajustées et
exportées séparément à partir des mêmes exemples. Le chargeur choisit automatiquement celle qui
correspond au backend et refuse explicitement tout mélange.

Le contrat applicatif reste `IntentEngine.predict(transcript) -> IntentResult`, défini dans
`src/alfred_voice_assistant/intent.py`. Un futur export compatible peut être testé avec
`--specialization-dir`, sans modifier la capture Vosk, le worker asynchrone ou Home Assistant.

## Développement

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

Références API : [fonctionnement temps réel de Kyutai STT](https://kyutai.org/stt/),
[checkpoint Kyutai STT](https://huggingface.co/kyutai/stt-1b-en_fr),
[implémentation officielle Kyutai](https://github.com/kyutai-labs/delayed-streams-modeling),
[exemple microphone officiel Vosk](https://github.com/alphacep/vosk-api/blob/master/python/example/test_microphone.py),
[modèle français Vosk](https://alphacephei.com/vosk/models),
[API laya-mlx](https://github.com/mizorewww/laya-mlx) et
[checkpoint multilingue](https://huggingface.co/aac6fef/laya-multilingual-mlx),
[API REST Home Assistant](https://developers.home-assistant.io/docs/api/rest/) et
[Laya PyTorch](https://github.com/NandhaKishorM/laya).
