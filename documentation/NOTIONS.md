# Guide des Notions Clés pour la Soutenance (RAG)

Ce document rassemble l'ensemble des notions théoriques, des formules mathématiques, des acronymes et du vocabulaire technique indispensables pour maîtriser le sujet à 100 % et réussir la soutenance (peer-evaluation et étape de *Recode*).

---

## Sommaire

1. [Acronymes Décryptés](#1-acronymes-décryptés)
2. [Qu'est-ce que le RAG et Pourquoi l'Utilise-t-on ?](#2-quest-ce-que-le-rag-et-pourquoi-lutilise-t-on-)
3. [Les 4 Étapes Fondamentales du RAG](#3-les-4-étapes-fondamentales-du-rag)
4. [La Recherche Lexicale : BM25 et Tokenisation](#4-la-recherche-lexicale--bm25-et-tokenisation)
5. [Le Chunking : Stratégies et Enjeux](#5-le-chunking--stratégies-et-enjeux)
6. [Les Métriques d'Évaluation : IoU 1D et Recall@k](#6-les-métriques-dévaluation--iou-1d-et-recallk)
7. [Génération LLM : Grounding et Gestion des Biais](#7-génération-llm--grounding-et-gestion-des-biais)
8. [Le Lexique à Placer à l'Oral](#8-le-lexique-à-placer-à-loral)
9. [Questions Pièges Fréquentes en Soutenance](#9-questions-pièges-fréquentes-en-soutenance)

---

## 1. Acronymes Décryptés

- **RAG (*Retrieval-Augmented Generation*)** : Technique combinant un moteur de recherche documentaire (*Retrieval*) et un modèle génératif (*Generation*) pour produire des réponses basées sur des sources externes sans réentraîner le modèle.
- **LLM / SLM (*Large / Small Language Model*)** : Modèle de langage. Notre projet utilise un **SLM** (`Qwen/Qwen3-0.6B`) avec 600 millions de paramètres, optimisé pour tourner rapidement en local sur CPU.
- **BM25 (*Best Matching 25*)** : Algorithme de recherche d'information probabiliste (famille Okapi BM25) calculant la pertinence d'un document pour une requête de mots-clés.
- **TF-IDF (*Term Frequency - Inverse Document Frequency*)** : Ancêtre de BM25. Évalue l'importance d'un mot dans un document par rapport à l'ensemble du corpus.
- **IoU (*Intersection over Union*)** : Indice de Jaccard mesurant le taux de chevauchement entre deux plages. Ici appliqué en **1D** sur des indices de caractères.
- **AST (*Abstract Syntax Tree*)** : Arbre syntaxique abstrait. Représentation arborescente de la structure du code source utilisée par le compilateur ou l'interpréteur Python (`ast.parse()`).
- **NMS (*Non-Maximum Suppression*)** : Déduplication spatiale (empruntée à la vision par ordinateur). Permet d'écarter ou de fusionner des extraits redondants d'un même fichier pour diversifier les résultats.
- **KV Cache (*Key-Value Cache*)** : Mécanisme mémoire des architectures Transformers stockant les vecteurs clés et valeurs déjà calculés pour accélérer la génération token par token.
- **UUID (*Universally Unique Identifier*)** : Chaîne d'identification unique (128 bits) générée pour chaque question (`question_id`).

---

## 2. Qu'est-ce que le RAG et Pourquoi l'Utilise-t-on ?

### Le Problème Fondamental des LLM
Un modèle de langage classique souffre de trois défauts majeurs :
1. **Connaissances figées dans le temps** : Il ne connaît que les données sur lesquelles il a été entraîné (ex: impossible de connaître le code d'une bibliothèque privée ou sortie hier).
2. **Hallucinations** : Lorsqu'il ne connaît pas une réponse technique précise (ex: nom exact d'un paramètre), il invente avec une grande assurance.
3. **Absence de traçabilité** : Un LLM seul est incapable d'indiquer le fichier ou le numéro de ligne exact où se trouve l'information.

### Pourquoi pas le *Fine-Tuning* (Réentraînement) ?
- **Coût astronomique** en puissance de calcul (GPU) et en temps.
- **Oubli catastrophique (*Catastrophic Forgetting*)** : Le modèle perd ses capacités initiales en apprenant de nouvelles données.
- **Manque de fraîcheur** : Dès qu'une fonction change dans le dépôt Git, il faudrait tout réentraîner.

### La Solution RAG
Le RAG résout ce problème en **séparant la mémoire de stockage et la capacité de raisonnement** :
- Le corpus technique (vLLM) est stocké dans un moteur de recherche indexé.
- Au moment où l'utilisateur pose une question, on va chercher les extraits exacts (*Retrieval*).
- On fournit ces extraits au modèle dans son prompt (*Augmenting*).
- Le modèle n'a plus qu'à synthétiser l'information sans rien inventer (*Generating*).

---

## 3. Les 4 Étapes Fondamentales du RAG

```text
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  1. INDEXING    │ ───►  │  2. RETRIEVING  │ ───►  │  3. AUGMENTING  │ ───►  │  4. GENERATING  │
│ Découpage (AST) │       │ Recherche BM25  │       │ Extraction des  │       │ Rédaction par   │
│ & tokens en RAM │       │ & Déduplication │       │ tranches brutes │       │ Qwen sur CPU    │
└─────────────────┘       └─────────────────┘       └─────────────────┘       └─────────────────┘
```

1. **Indexing (Indexation)** : Préparation des données hors-ligne. Découpage du corpus en chunks $\le 2000$ caractères et création de l'index BM25.
2. **Retrieving (Récupération)** : Prise en compte de la question, recherche des $k$ passages les plus pertinents et filtrage des doublons spatiaux.
3. **Augmenting (Augmentation)** : Lecture des fichiers sur le disque pour extraire les vraies chaînes de caractères `texte[first:last]` et assemblage du prompt dans la limite du budget de tokens (2048 tokens).
4. **Generating (Génération)** : Inférence locale du LLM pour rédiger une réponse factuelle, concise et sourcée.

---

## 4. La Recherche Lexicale : BM25 et Tokenisation

### Comment fonctionne BM25 ?
BM25 calcule un score de pertinence entre une requête $Q$ et un document $D$ :
$$Score(D, Q) = \sum_{q \in Q} IDF(q) \cdot \frac{TF(q, D) \cdot (k_1 + 1)}{TF(q, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{avgdl}\right)}$$

- **TF (*Term Frequency*)** : Nombre de fois où le mot $q$ apparaît dans le chunk $D$. Plus il apparaît, plus le score monte, mais avec un effet de **saturation** contrôlé par $k_1$ (faire du bourrage de mots-clés n'augmente pas le score à l'infini).
- **IDF (*Inverse Document Frequency*)** : Rareté du mot dans tout le corpus. Un mot très rare (ex: `PagedAttention`) rapporte énormément de points ; un mot présent partout (ex: `import` ou `return`) ne rapporte presque rien.
- **Pénalisation de longueur ($b$)** : Les documents anormalement longs sont pénalisés pour ne pas avantager artificiellement les gros blocs au détriment des extraits concis.

### Les Spécificités de la Tokenisation de Code
Sur du texte ordinaire, un tokenizer sépare par les espaces et les virgules. Sur du code, cela échoue lamentablement :
- Si la question demande : *"What does the async LLM engine do?"*
- Le code source contient : `class AsyncLLMEngine:` ou `async_llm_engine.step()`.
- Un tokenizer naïf ne voit jamais le mot `"async"` seul !

**Notre solution (implémentée dans `src/indexing/tokenizer.py`) :**
1. **Identifier Splitting** : Découpage des identifiants `camelCase`, `PascalCase` et `snake_case`. On conserve **à la fois** le terme décomposé (`async`, `llm`, `engine`) ET l'identifiant unifié (`asyncllmengine`).
2. **Path Token Injection** : Injection des mots du chemin de fichier (ex: `docs/features/lora.md` injecte `['features', 'lora']` avec un poids 2x). Cela permet de retrouver des chunks qui décrivent une fonctionnalité sans répéter son nom dans chaque ligne.

---

## 5. Le Chunking : Stratégies et Enjeux

### Pourquoi pas des chunks de taille fixe (ex: tous les 1000 caractères) ?
Un découpage aveugle au nombre fixe de caractères risque de couper :
- En plein milieu d'une condition `if`,
- Au milieu d'une déclaration de fonction,
- En séparant la docstring d'une classe de son implémentation.

### Nos Deux Stratégies Distinctes
1. **Code Python (`chunk_python_code`)** :
   - Utilise l'AST de Python (`ast.parse()`).
   - Isole les définitions de fonctions (`def`, `async def`) et de classes (`class`).
   - Regroupe les petits blocs et imports voisins jusqu'au plafond des 2000 caractères.
2. **Markdown & Documentation (`chunk_markdown_text`)** :
   - Découpe selon la hiérarchie sémantique des titres Markdown (`#`, `##`, `###`), puis des doubles sauts de ligne (`\n\n`).
   - Fusionne les titres orphelins sans corps de texte (>10 caractères) avec la section suivante afin de préserver le contexte sémantique et booster le Recall.
3. **L'Invariant des 2000 Caractères et la Borne Basse (400 Caractères)** :
   - *Borne haute (2000)* : Le sujet et la `moulinette` 42 **invalident l'intégralité du rendu** si un seul chunk dépasse 2000 caractères. Toute valeur supérieure est rejetée dès la CLI.
   - *Borne basse (400)* : Sans borne minimale, une valeur infime (ex: 10 ou 100) découperait le corpus en centaines de milliers de micro-morceaux, provoquant un freeze machine par épuisement mémoire (OOM). La borne à 400 sécurise l'exécution.
   - Si un bloc dépasse `max_chunk_size`, `slice_range_by_lines()` le découpe ligne par ligne avec un chevauchement adaptatif (`min(overlap, max_chunk_size // 4)`) assurant une progression positive constante.

---

## 6. Les Métriques d'Évaluation : IoU 1D et Recall@k

### Le Calcul de l'IoU 1D (Intersection over Union)
Pour savoir si un extrait trouvé $R = [r_{\text{start}}, r_{\text{end}}]$ correspond à la réponse attendue $G = [g_{\text{start}}, g_{\text{end}}]$ :

1. **Règle éliminatoire** : Si $R.\text{file\_path} \neq G.\text{file\_path}$, alors $\text{IoU} = 0.0$.
2. **Calcul des longueurs de plages** :
   $$\text{Intersection} = \max(0, \min(r_{\text{end}}, g_{\text{end}}) - \max(r_{\text{start}}, g_{\text{start}}))$$
   $$\text{Union} = (r_{\text{end}} - r_{\text{start}}) + (g_{\text{end}} - g_{\text{start}}) - \text{Intersection}$$
   $$\text{IoU} = \frac{\text{Intersection}}{\text{Union}}$$
3. **Critère de succès** : Si $\text{IoU} \ge 0.05$ (5 % de recouvrement), la source est comptabilisée comme **trouvée**.

```text
Intervalle R (Retrieved) :   [===================]
Intervalle G (Ground Truth) :            [===================]
                                         |<- Inter ->|
                             |<------------- Union ---------->|
```

### Le Recall@k (Rappel à $k$)
Le Recall@k mesure la capacité du système à retrouver **toutes** les sources nécessaires parmi ses $k$ premiers résultats :
$$\text{Recall@}k(q) = \frac{\text{Nombre de sources vraies de } q \text{ trouvées dans le top-}k}{\text{Nombre total de sources attendues pour } q}$$

- **Recall@1** : La bonne source est-elle en première position ?
- **Recall@5** : La bonne source est-elle dans les 5 premiers résultats ? (C'est la métrique reine évaluée à 42 : seuil à **80 % sur Docs** et **50 % sur Code**).
- *Différence avec la Précision* : La précision mesure le pourcentage d'extraits utiles parmi les $k$ proposés ; le rappel mesure le pourcentage d'informations requises que l'on n'a pas manquées.

---

## 7. Génération LLM : Grounding et Gestion des Biais

### Le Grounding (Ancrage Factuel)
Le *Grounding* est la consigne stricte donnée au modèle de ne baser son raisonnement **que sur les documents fournis** dans son prompt. Si l'information n'y figure pas, le modèle a pour consigne d'indiquer qu'il ne dispose pas d'éléments suffisants.

### Le Biais de Refus des Petits Modèles (*Refusal Bias*)
Lors de nos premiers tests avec `Qwen3-0.6B`, le modèle répondait systématiquement *"The provided context does not contain sufficient information..."* même quand le chunk contenait exactement la réponse !
- **Pourquoi ?** Un modèle de 600M de paramètres est prudent. Face à un prompt disant *"Do not extrapolate under any circumstances"*, si la question demande *"fused batched MoE"* et que le code renvoie `BatchedExperts` sans répéter la question mot pour mot, le modèle panique et prend la porte de sortie de secours (le refus).
- **La solution** : Recadrer le prompt système pour lui expliquer que le contexte provient de fichiers de code et de documentation, et qu'il doit s'appuyer sur les signatures de fonctions, les classes et les chemins de fichiers.

### Paramètres et Optimisations de Génération
- `torch.bfloat16` & `device_map="auto"` : Empreinte mémoire divisée par deux et inférence accélérée avec repli automatique sur CPU.
- `torch.inference_mode()` : Désactivation du calcul de gradients pour maximiser le débit.
- `do_sample=False` : Recherche gloutonne (*Greedy Search*) pour un résultat 100 % déterministe et reproductible.
- `enable_thinking=False` & nettoyage regex : Suppression du mode réflexion interne de Qwen3 pour débuter immédiatement sur la réponse factuelle et supprimer toute balise `<think>...</think>`.
- `max_new_tokens=128` : Limite stricte pour des réponses techniques concises, réduisant la latence globale par question.

---

## 8. Le Lexique à Placer à l'Oral

Pour démontrer une excellente maîtrise technique devant les correcteurs, voici les termes précis à employer :

| Terme Français / Anglais | Signification & Contexte d'usage |
| :--- | :--- |
| **In-Memory Cache (Singleton)** | Expliquer que l'index BM25 reste chargé en RAM pour traiter 200 questions en 0,45 seconde sans relire le disque. |
| **Lazy Loading (Chargement paresseux)** | Expliquer que `torch` et `transformers` ne sont importés que lors des commandes `answer`, laissant `search` et `index` ultra-légers. |
| **Domain Boosting (Boost de domaine)** | Multiplicateur de score (x2.0) ajustant les poids des fichiers selon le domaine cible (`docs` vs `code`) pour maximiser le Recall. |
| **Spatial Deduplication (NMS)** | Expliquer qu'on fusionne les extraits d'un même fichier s'ils se chevauchent afin de ne pas gaspiller les cartouches du top-$k$. |
| **Syntactic Boundary (Frontière syntaxique)** | Expliquer que le chunker Python utilise l'AST pour ne pas couper le code en plein milieu d'une structure logique. |
| **Token Budget (Budget de tokens)** | Expliquer qu'on empile les extraits jusqu'au plafond de 2048 tokens calculé par le tokenizer de Qwen. |
| **Subword Tokenization** | Découpage des identifiants `camelCase` et `snake_case` pour assurer la correspondance avec le langage naturel. |
| **Greedy Decoding (Recherche gloutonne)** | Inférence sans température (`do_sample=False`) pour garantir un résultat déterministe et reproductible. |
| **1D IoU Overlap** | Le calcul d'intersection sur union entre deux plages d'indices de caractères avec un seuil de 5 %. |

---

## 9. Questions Pièges Fréquentes en Soutenance

### Q1 : "Pourquoi avoir choisi BM25 plutôt qu'un modèle d'embeddings vectoriels (Dense Retrieval) ?"
> **Réponse** : "Sur une codebase technique, le vocabulaire est hautement spécifique (noms de fonctions, constantes comme `FP8_MIN`, flags comme `skip_sampler_cpu_output`). Les modèles d'embeddings ont tendance à noyer ces mots-clés rares dans des vecteurs de sémantique floue. BM25 excelle sur la précision lexicale exacte, fonctionne sans GPU, et permet d'indexer 47 000 chunks en **8 secondes** avec une empreinte mémoire minime, atteignant 85 % de Recall sur la doc et 76,8 % sur le code."

### Q2 : "Comment garantissez-vous qu'aucun chunk ne dépasse 2000 caractères ?"
> **Réponse** : "La contrainte est verrouillée à trois niveaux : d'abord dans nos fonctions de chunking (avec une assertion stricte et un algorithme de repli par découpage de lignes), ensuite lors de la déduplication spatiale où deux morceaux ne fusionnent que si `union <= 2000`, et enfin par des tests unitaires dédiés dans `tests/test_chunkers.py`."

### Q3 : "Pourquoi la première commande `answer` met-elle 2 ou 3 secondes à démarrer alors que `search` est instantané ?"
> **Réponse** : "C'est l'illustration exacte de notre architecture en **Lazy Loading**. La commande `search` n'a besoin que de BM25 et de NumPy, donc elle s'exécute en quelques millisecondes. La commande `answer` est la seule qui initialise les poids PyTorch du modèle `Qwen/Qwen3-0.6B` (~1,2 Go). En isolant ce chargement, nous évitons de charger PyTorch quand on ne fait que de la recherche ou de l'évaluation."

### Q4 : "Si le bon fichier est trouvé mais que le premier caractère commence 50 caractères après la référence, est-ce comptabilisé ?"
> **Réponse** : "Oui, grâce au seuil de l'IoU fixé à **0.05** (5 %). Il n'est pas nécessaire d'avoir découpé le texte exactement aux mêmes indices que le créateur du dataset ; dès qu'il y a un chevauchement d'au moins 5 % dans le bon fichier, la source est considérée comme trouvée."

### Q5 : "Que se passe-t-il si l'utilisateur saisit une requête vide, uniquement des espaces ou des caractères spéciaux ?"
> **Réponse** : "Le système gère ces cas limites de façon défensive sans jamais planter ni afficher de traceback : `cli.py` et `tokenizer.py` filtrent les chaînes vides ou sans caractères alphanumériques et retournent immédiatement une liste vide `[]` ou un message clair, garantissant zéro crash lors de la correction."

### Q6 : "Pourquoi imposer une limite basse de 400 caractères sur `max_chunk_size` alors que le sujet ne spécifie qu'une limite haute ?"
> **Réponse** : "Si un utilisateur ou un test malveillant tente d'indexer avec une valeur de 10 ou 100 caractères, cela découpe le corpus en centaines de milliers de micro-fragments, provoquant une explosion combinatoire des tokens et un freeze/OOM de la machine d'évaluation. Encadrer `max_chunk_size` entre 400 et 2000 caractères préserve la stabilité de l'environnement tout en garantissant des performances optimales de Recall (>= 80% sur la doc et >= 50% sur le code)."
