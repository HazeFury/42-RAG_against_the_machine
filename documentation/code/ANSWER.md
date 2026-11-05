# Commande : `answer`

## 1. À quoi sert cette commande ?

La commande `answer` exécute le **pipeline RAG complet de bout en bout pour une seule question** :
1. Elle **cherche** les $k$ meilleurs morceaux de code/doc avec le retriever lexical.
2. Elle **extrait** le texte exact de ces morceaux directement dans les fichiers sur le disque pour construire le contexte (sans dépasser la limite de 2048 tokens).
3. Elle charge le modèle local **`Qwen/Qwen3-0.6B`** de façon optimisée matériellement (`bfloat16`, auto-détection GPU/MPS avec repli propre sur CPU) et génère une réponse technique directe, factuelle et sourcée.
4. Elle affiche dans le terminal la question, la liste des sources utilisées avec leurs coordonnées, et la réponse rédigée.

---

## 2. Syntaxe CLI

```bash
uv run python -m src answer "<query>" [--k 5] [--index_dir data/processed] [--model_name Qwen/Qwen3-0.6B]
```

### Paramètres :
- `<query>` *(str, obligatoire)* : La question posée.
- `--k` *(int, défaut : 5)* : Nombre de sources à fournir au LLM comme contexte.
- `--index_dir` *(str, défaut : "data/processed")* : Dossier où trouver l'index BM25.
- `--model_name` *(str, défaut : "Qwen/Qwen3-0.6B")* : Modèle Hugging Face à utiliser.

### Exemple de sortie terminal :
```text
Question: What activation formats does the fused batched MoE layer return in vLLM?

Retrieved sources:
  1. data/raw/vllm-0.10.1/vllm/model_executor/layers/fused_moe/fused_batched_moe.py [28412:28661]

Answer:
The fused batched MoE layer returns the activation formats `mk.FusedMoEActivationFormat.BatchedExperts`.
```

---

## 3. Déroulement détaillé du code exécuté

```text
[Terminal]
   │
   ▼
1. src/__main__.py: main()
   │  Fire dispatche vers RAGCLI.answer()
   ▼
2. src/cli.py: RAGCLI.answer()
   │  - Vérifie que query n'est pas vide et k > 0
   │  - Appelle self.pipeline.answer(query, k, index_dir, model_name)
   │  - Reçoit l'objet MinimalAnswer
   │  - Formate et affiche proprement les sources et la réponse dans le terminal
   ▼
3. src/pipeline.py: RagPipeline.answer()
   │  a. Étape 1 : Retrieval (Recherche)
   │     sources = self.search(query=query, k=k, index_dir=index_dir)
   │     -> Renvoie list[MinimalSource]
   │  b. Étape 2 : Initialisation paresseuse (Lazy Loading)
   │     generator = self._get_generator(model_name)
   │     context_builder = self._get_context_builder(model_name)
   │  c. Étape 3 : Augmentation du contexte
   │     context = context_builder.build_context(sources)
   │  d. Étape 4 : Génération LLM
   │     generated_answer = generator.generate_answer(question=query, context=context)
   │  e. Retourne MinimalAnswer(question_id, question, retrieved_sources, answer)
   ▼
4. src/generation/context_builder.py: ContextBuilder.build_context()
   │  - Pour chaque MinimalSource :
   │    -> Lit le fichier avec encoding="utf-8", errors="ignore" (utilise un cache _file_cache)
   │    -> Extrait la tranche exacte : content[first:last]
   │    -> Formate l'en-tête : "--- Source: file_path [chars first:last] ---\n<texte>"
   │    -> Compte les tokens avec le tokenizer de Qwen (avec repli robuste en cas d'erreur)
   │    -> Empile les blocs tant que total_tokens <= 2048 (max_context_tokens)
   ▼
5. src/generation/llm_generator.py: LLMGenerator.generate_answer()
   │  - load_model() :
   │    -> Importe torch et transformers UNIQUEMENT ici (Lazy Loading)
   │    -> Détecte l'accélération matérielle (CUDA, MPS ou CPU)
   │    -> Charge Qwen/Qwen3-0.6B en bfloat16 avec device_map="auto" et low_cpu_mem_usage=True
   │    -> Repli défensif sur CPU en bfloat16 en cas de contrainte environnementale
   │  - format_prompt() :
   │    -> Construit les messages avec le prompt système strict de grounding
   │    -> Applique le chat template : tokenizer.apply_chat_template(messages, enable_thinking=False)
   │  - Inférence :
   │    -> Transfère proprement les tenseurs d'entrée sur model.device (compatible BatchEncoding & mocks)
   │    -> Inférence accélérée : with torch.inference_mode():
   │    -> model.generate(**inputs, max_new_tokens=128, do_sample=False, use_cache=True)
   │    -> Décodage du texte et nettoyage d'éventuelles balises <think>...</think> résiduelles via regex
   │    -> Renvoie la chaîne de caractères finale
```

---

## 4. Les points clés à expliquer à l'oral

1. **Le Lazy Loading (crucial)** : Le modèle de 600M de paramètres pèse ~1,2 Go. Si on importait `torch` et `transformers` au début du fichier `cli.py`, même un simple `search` ou `index` mettrait plusieurs secondes à démarrer et consommerait inutilement de la RAM. Grâce au lazy loading, `torch` n'est importé que si on tape `answer` ou `answer_dataset`.
2. **Exécution optimisée en `bfloat16` & Hardware-Aware** : Le code détecte automatiquement la présence d'un GPU/MPS ou bascule sur CPU, en chargeant les poids en précision `bfloat16` avec `device_map="auto"` et `low_cpu_mem_usage=True`. Cela divise par deux la mémoire requise par rapport au `float32` sans aucune dégradation de qualité.
3. **Pas d'Ollama, inférence Python pure** : Le sujet interdit formellement les démons externes type Ollama car l'environnement de correction 42 ne fait qu'un `uv sync`. Tout tourne avec `transformers` et `torch` en local.
4. **Vitesse d'inférence (`torch.inference_mode()` & `max_new_tokens=128`)** : L'utilisation de `inference_mode()` désactive la construction du graphe d'autodiff, et limiter la génération à 128 nouveaux tokens garantit des réponses concises, rapides et directes.
5. **Désactivation du mode thinking (`enable_thinking=False`)** : Qwen3 possède par défaut un mode "thinking" (chaîne de pensée). L'activer sur CPU ferait générer des centaines de tokens de réflexion lente inutile. En le désactivant et en filtrant les balises `<think>`, le modèle répond directement au fait.

