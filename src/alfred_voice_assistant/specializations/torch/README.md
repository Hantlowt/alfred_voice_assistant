# alfred-actions-pytorch

Portable Laya specialization artifact (format version 1).

- Base model: `convaiinnovations/laya-multilingual`
- Strategy: `residual_embedding_transform`
- Task: Classer une transcription vocale française destinée à un assistant domotique dans une liste fermée d'actions. Les commandes affirmatives, actuelles et directement adressées à l'assistant déclenchent l'action précise. Toute conversation ordinaire, négation, citation, demande rapportée, hypothèse, hésitation, équipement non pris en charge ou phrase ambiguë doit produire do_nothing.
- Decision labels: do_nothing, éteins les lumières, allume les lumières, allume l'halogène, allume le salon, allume la salle à manger, éteins l'halogène, éteins le luminaire du salon, éteins le salon, éteins la salle à manger, allume les leds, éteins les leds, allume la clim, éteins la clim
- License: Apache-2.0

```python
from laya_studio import SpecializedLaya
agent = SpecializedLaya.from_pretrained(".")
result = agent.predict("your input")
```

Laya model weights by Convai Innovations and upstream contributors.
