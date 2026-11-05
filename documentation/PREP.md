# Fiche de Préparation à la Soutenance (Oral RAG)

Ce document contient les réponses idéales, claires, concises et argumentées aux **5 grandes questions classiques** posées lors de la soutenance du projet RAG à 42.

Pour chaque question, tu trouveras :
- **L'accroche synthétique** (ce que tu peux dire directement à l'oral en 30 secondes).
- **L'argumentation technique détaillée** (avec les références exactes à notre code).
- **La version anglaise** (si le correcteur ou la soutenance se déroule en anglais).

---

## Question 1 : "What is Retrieval-Augmented Generation (RAG) and why is it useful?"

### 1.1 Réponse synthétique à l'oral
> "Le RAG est une architecture qui découple la mémoire des connaissances de la capacité de raisonnement du LLM. 
> Au lieu de réentraîner un modèle (ce qui est long, coûteux et vite obsolète), on indexe la documentation et le code d'une bibliothèque dans un moteur de recherche. Lorsqu'une question est posée, le système récupère d'abord les extraits les plus pertinents (*Retrieval*), les injecte dans le prompt (*Augmentation*), et le LLM n'a plus qu'à formuler une réponse factuelle basée uniquement sur ces sources (*Generation*).
> C'est utile car cela élimine les hallucinations, garantit des informations à jour en temps réel, offre une traçabilité parfaite (fichier et indices exacts), et fonctionne à coût minime sur CPU."

### 1.2 Détails techniques du projet
1. **Élimination des hallucinations** : Les LLMs ont tendance à inventer des paramètres inexistants. Grâce au prompt de grounding dans `src/generation/llm_generator.py`, le modèle ne répond qu'à partir de code existant.
2. **Traçabilité totale** : Chaque réponse est adossée à un modèle `MinimalSource` contenant le `file_path` et les indices réels de caractères `[first_character_index:last_character_index]`.
3. **Économie de ressources** : Pas besoin de GPU pour réentraîner un modèle de plusieurs milliards de paramètres ; notre système tourne en local sur CPU avec un index BM25 qui se construit en 8 secondes.

### 1.3 English Version
> "Retrieval-Augmented Generation (RAG) is an architecture that decouples external knowledge storage from a language model's reasoning capabilities. Instead of fine-tuning or retraining a model—which is computationally expensive, prone to catastrophic forgetting, and quickly outdated—we index our corpus in a dedicated retrieval engine. When a question is asked, the system retrieves the most relevant source snippets, injects them into the model's context window, and prompts the LLM to generate an answer grounded strictly in that evidence. It is useful because it drastically mitigates hallucinations, guarantees real-time freshness, provides exact source traceability, and runs efficiently on local hardware."

---

## Question 2 : "Walk me through your complete RAG pipeline, from raw documents to a generated answer."

### 2.1 Réponse synthétique à l'oral
> "Notre pipeline se découpe en 4 grandes étapes séquentielles :
> 1. **Indexation** : `corpus_loader.py` filtre les fichiers utiles de vLLM. `chunkers.py` découpe le Python avec l'AST et la doc par titres en chunks $\le 2000$ caractères. `tokenizer.py` éclate le camelCase/snake_case et injecte les mots du chemin. `indexer.py` calcule et sauvegarde la matrice BM25 dans `data/processed/`.
> 2. **Recherche (Retrieval)** : `lexical_retriever.py` charge l'index une seule fois en mémoire (singleton), interroge les $k \times 3$ meilleurs candidats et applique une déduplication spatiale pour fusionner les doublons dans le même fichier.
> 3. **Augmentation** : `context_builder.py` rouvre les fichiers sur disque, extrait la tranche exacte `content[first:last]` et assemble un prompt sous la limite de 2048 tokens.
> 4. **Génération** : `llm_generator.py` charge `Qwen/Qwen3-0.6B` en *lazy loading* sur CPU, désactive le mode thinking, et génère une réponse concise et factuelle."

### 2.2 Schéma du flux de données
```text
data/raw/ (Code & Docs)
       │
       ▼ [corpus_loader.py] : Filtrage & normalisation des chemins relatifs
       ▼ [chunkers.py]      : AST Python + Titres Markdown (<= 2000 car.)
       ▼ [tokenizer.py]     : camelCase/snake_case split + Path injection
       ▼ [indexer.py]       : Matrice BM25s -> data/processed/ (Index + Metadata)
       │
User Query
       │
       ▼ [lexical_retriever.py] : BM25 top-(k*3) + Déduplication spatiale (NMS)
       │                         -> list[MinimalSource]
       ▼ [context_builder.py]   : Lecture content[first:last] + Budget 2048 tokens
       ▼ [llm_generator.py]     : Inférence CPU Qwen3-0.6B (Grounding, no thinking)
       │
Generated Answer
```

### 2.3 English Version
> "Our pipeline follows four sequential stages:
> First, **Indexing**: `corpus_loader` discovers source files; `chunkers` segments Python via AST and Markdown via headers under a strict 2000-character ceiling; `tokenizer` decomposes identifiers and injects path keywords; and `indexer` builds and persists the BM25 index and chunk coordinates to disk.
> Second, **Retrieval**: `lexical_retriever` loads the index into memory once, queries top candidate chunks, and performs spatial deduplication to merge overlapping spans from the same file.
> Third, **Augmentation**: `context_builder` reads the physical files from disk, slices exact character spans `content[first:last]`, and builds a structured prompt enforcing a 2048-token budget.
> Finally, **Generation**: `llm_generator` lazily loads `Qwen/Qwen3-0.6B` on CPU, disables thinking tokens, and generates a concise, grounded answer."

---

## Question 3 : "What is the difference between TF-IDF and BM25 as retrieval methods?"

### 3.1 Réponse synthétique à l'oral
> "BM25 est une évolution probabiliste moderne de TF-IDF qui résout deux défauts majeurs de ce dernier :
> 1. **La saturation de la fréquence des termes ($k_1$)** : En TF-IDF standard, plus un mot est répété dans un document, plus son score grimpe linéairement (effet de bourrage). En BM25, la fonction est asymptotique : répéter 50 fois un mot n'apporte que très peu de points supplémentaires par rapport à 3 fois.
> 2. **La normalisation par la longueur du document ($b$)** : TF-IDF favorise artificiellement les très longs documents qui contiennent naturellement plus de mots. BM25 divise la fréquence par le ratio de la longueur du document sur la longueur moyenne du corpus, pénalisant les pavés verbeux.
> Pour notre codebase, BM25 évite qu'une grosse classe de 2000 caractères avec une boucle répétant un nom de méthode n'écrase un petit extrait concis contenant la vraie réponse."

### 3.2 Formules comparées

- **TF-IDF classique** :
  $$\text{Score} = \text{TF}(t, d) \times \log\left(\frac{N}{\text{DF}(t)}\right)$$

- **Okapi BM25** :
  $$\text{Score}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot \frac{\text{TF}(q, D) \cdot (k_1 + 1)}{\text{TF}(q, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

- **Paramètres clés de BM25** :
  - $k_1$ (entre 1.2 et 1.5) : Calibre la vitesse de saturation du TF.
  - $b$ (généralement 0.75) : Règle l'intensité de la pénalité de longueur du document ($|D| / \text{avgdl}$).

### 3.3 English Version
> "BM25 is a probabilistic retrieval framework that refines classical TF-IDF by addressing two fundamental limitations:
> First, **Term Frequency Saturation**: In standard TF-IDF, score scales monotonically with term count, making it vulnerable to keyword stuffing. BM25 introduces parameter $k_1$, which sets an asymptotic ceiling on term frequency so marginal occurrences contribute diminishing score gains.
> Second, **Document Length Normalization**: Standard TF-IDF unfairly biases in favor of long documents that naturally accumulate higher word counts. BM25 introduces parameter $b$, penalizing documents that exceed the average corpus length.
> For source code, BM25 ensures that a large file repeating a function name in a loop does not overshadow a concise, highly relevant snippet."

---

## Question 4 : "What implementation choices did you make and why? What trade-offs did you consider?"

### 4.1 Réponse synthétique à l'oral
> "Nous avons fait 5 choix d'ingénierie majeurs :
> 1. **BM25s plutôt qu'un vector store (Embeddings)** : BM25s s'indexe en 8 secondes, tourne sans GPU, et excelle sur les identifiants techniques exacts (`FP8_MIN`, `PagedAttention`), là où les embeddings introduisent du flou sémantique.
> 2. **Stockage exclusif des coordonnées dans l'index** : Pour économiser la RAM, `metadata.json` ne stocke que `(file_path, first, last)` (~4 Mo). Le texte brut n'est relu que pour les $k$ vainqueurs au moment de l'augmentation.
> 3. **Découpage AST pour le code** : Respecte les fonctions et classes complètes au lieu de couper arbitrairement au milieu d'un bloc.
> 4. **Déduplication spatiale (NMS)** : Fusionne les chunks adjacents du même fichier s'ils font $\le 2000$ car., libérant des slots pour d'autres fichiers dans le top-$k$.
> 5. **Lazy Loading de PyTorch/Qwen** : `torch` et les poids de 1,2 Go ne sont chargés que lors de `answer`, ce qui laisse les commandes `index`, `search` et `evaluate` instantanées."

### 4.2 Analyse des compromis (*Trade-offs*)

| Choix d'implémentation | Avantages obtenus | Compromis consenti (*Trade-off*) |
| :--- | :--- | :--- |
| **BM25 lexical vs Vectoriel** | Vitesse extrême (8 s d'indexation), empreinte RAM nulle, match exact des symboles | Incapacité à matcher des paraphrases pures sans aucun mot-clé commun |
| **Stockage des coordonnées seules** | Index ultra-léger (4 Mo), empreinte mémoire minimale | Petite surcharge d'I/O disque lors de `answer` pour relire les $k$ fichiers |
| **AST Python vs Fenêtre fixe** | Cohérence syntaxique parfaite des fonctions | Parsing plus lent qu'un simple `split()` de caractères, gestion de cas de repli |
| **`enable_thinking=False`** | Réponse immédiate sur CPU, économie de 500 tokens de calcul | Moins de capacité de raisonnement multi-étapes complexe |

### 4.3 English Version
> "We made five strategic architectural choices:
> 1. **Lexical BM25 over Dense Vector Embeddings**: Source code queries depend on exact identifier matches (e.g. `skip_sampler_cpu_output`). BM25 indexes in ~8 seconds without GPU dependencies and avoids semantic fuzziness.
> 2. **Coordinate-Only Metadata Storage**: We store only `(file_path, first, last)` coordinates in `metadata.json` (~4 MB), deferring raw text reads until top-$k$ augmentation to minimize RAM usage.
> 3. **AST-Guided Python Chunking**: Respects function and class boundaries rather than slicing mid-statement.
> 4. **Spatial Deduplication & Merging**: Merges adjacent overlapping chunks from the same file up to 2000 characters, avoiding redundant top-$k$ slots.
> 5. **Lazy Loading of LLM**: Imports `torch` and loads weights only when `answer` is called, keeping `index`, `search`, and `evaluate` instantaneous."

---

## Question 5 : "If you had more time, what would you improve in your system?"

### 5.1 Réponse synthétique à l'oral
> "Si nous avions plus de temps, nous apporterions 4 améliorations à fort retour sur investissement :
> 1. **Recherche Hybride (BM25 + Dense Embeddings avec RRF)** : Combiner la précision exacte de BM25 avec un petit modèle d'embeddings de code (ex: `bge-small-en`) via la formule Reciprocal Rank Fusion ($1 / (60 + \text{rank})$) pour attraper les questions rédigées en langage naturel pur.
> 2. **Cross-Encoder Re-ranking** : Prendre les 30 meilleurs candidats renvoyés par BM25 en 1 ms, et les faire réordonner par un mini cross-encoder pour propulser le bon extrait en position #1 (Recall@1).
> 3. **Recherche Parent-Document** : Indexer des petits morceaux granulaires de 300 caractères pour matcher avec précision, mais renvoyer au LLM la fonction parent complète (jusqu'à 2000 car.) pour un contexte enrichi.
> 4. **Indexation Incrémentale** : Suivre les empreintes SHA-256 des fichiers dans un `manifest.json` pour ne réindexer que les fichiers modifiés en quelques millisecondes."

### 5.2 Les axes d'amélioration en détail
1. **Reciprocal Rank Fusion (RRF)** :
   $$\text{RRF\_Score}(d) = \frac{1}{60 + \text{rank}_{\text{BM25}}(d)} + \frac{1}{60 + \text{rank}_{\text{Dense}}(d)}$$
   Permet d'unir le meilleur des deux mondes (mots-clés exacts + proximité conceptuelle) sans devoir normaliser des scores incompatibles.
2. **Re-ranking par Cross-Encoder** :
   Un bi-encoder ou BM25 évalue la requête et le document séparément. Un cross-encoder prend le couple `(query, document)` en entrée et calcule l'attention croisée token par token, offrant une précision de classement chirurgicale.
3. **HyDE (*Hypothetical Document Embeddings*)** :
   Demander d'abord au LLM d'imaginer à quoi ressemblerait le code Python qui résout le problème, puis chercher les documents qui ressemblent à ce code hypothétique.

### 5.3 English Version
> "With additional time, we would prioritize four major improvements:
> 1. **Hybrid Retrieval with Reciprocal Rank Fusion (RRF)**: Fuse BM25 lexical search with a dense code-embedding model (like `bge-code`), combining exact keyword matching with semantic vector similarity via $RRF(d) = \sum \frac{1}{60 + \text{rank}(d)}$.
> 2. **Cross-Encoder Re-ranking**: Retrieve top-30 candidates with fast BM25, then pass `(query, chunk)` pairs to a lightweight cross-encoder to refine top-5 rankings and maximize Recall@1.
> 3. **Hierarchical Parent-Document Retrieval**: Index fine-grained 300-character chunks for precise retrieval matching, but expand to the surrounding class or function block during prompt augmentation.
> 4. **Incremental Indexing**: Maintain a SHA-256 hash manifest to re-index only modified files across Git commits, reducing indexation time to milliseconds."

