# Commande : `search`

## 1. À quoi sert cette commande ?

La commande `search` permet de poser une **question unique** (ou des mots-clés) dans le terminal et d'obtenir immédiatement la liste des **$k$ extraits les plus pertinents** du corpus, avec leur chemin de fichier et leur plage exacte de caractères (`[first:last]`).

Elle teste uniquement l'étape de **Retrieval (récupération lexicale)** sans lancer de modèle de langage.

---

## 2. Syntaxe CLI

```bash
uv run python -m src search "<query>" [--k 5] [--index_dir data/processed]
```

### Paramètres :
- `<query>` *(str, obligatoire)* : La question ou les mots-clés recherchés.
- `--k` *(int, défaut : 5)* : Le nombre de sources à retourner.
- `--index_dir` *(str, défaut : "data/processed")* : Le dossier où l'index BM25 et les métadonnées ont été stockés.

### Exemple de sortie :
```text
data/raw/vllm-0.10.1/vllm/attention/backends/rocm_flash_attn.py [1242:1995]
data/raw/vllm-0.10.1/vllm/attention/ops/paged_attn.py [1173:1196]
data/raw/vllm-0.10.1/vllm/attention/ops/rocm_aiter_paged_attn.py [350:394]
```

---

## 3. Déroulement détaillé du code exécuté

```text
[Terminal]
   │
   ▼
1. src/__main__.py: main()
   │  Fire dispatche vers RAGCLI.search()
   ▼
2. src/cli.py: RAGCLI.search()
   │  - Vérifie que la query n'est pas vide, contient au moins une lettre et >= 3 mots signifiants, et que k > 0
   │  - Vérifie que le dossier index_dir existe bien
   │  - Appelle self.pipeline.search(query, k, index_dir)
   │  - Affiche les résultats reçus ligne par ligne : "file_path [first:last]"
   ▼
3. src/pipeline.py: RagPipeline.search()
   │  - Récupère l'instance singleton du retriever via self._get_retriever(index_dir)
   │  - Appelle retriever.retrieve(query=query, k=k)
   ▼
4. src/retrieval/lexical_retriever.py: LexicalRetriever._load_index()
   │  - Ne s'exécute qu'au premier appel (chargement paresseux en mémoire)
   │  - Charge l'index binaire : bm25s.BM25.load(index_dir / "bm25_index", load_corpus=False)
   │  - Charge les métadonnées : safe_read_json(index_dir / "metadata.json")
   │  - Conserve tout en attributs de classe (self.retriever, self.metadata)
   ▼
5. src/retrieval/lexical_retriever.py: LexicalRetriever.retrieve()
   │  a. Tokenisation de la requête :
   │     - tokenize(query) avec le même tokenizer qu'à l'indexation
   │     - Découpe camelCase / snake_case, minuscules, suppression des stopwords
   │     - Si aucun token ne ressort (ex: "!@#$%"), renvoie une liste vide []
   │  b. Interrogation BM25 élargie :
   │     - Calcule fetch_k = min(max(k * 10, 50), len(metadata)) pour extraire jusqu'à 50 candidats
   │     - Récupère les scores BM25 et les identifiants de chunks
   │     - Si doc_type est renseigné, applique le multiplicateur de domaine (_compute_multiplier)
   │  c. Déduplication spatiale & Fusion intelligente :
   │     - Pour chaque candidat, retrouve ses coordonnées dans self.metadata
   │     - Si deux chunks candidats proviennent du MÊME fichier et se chevauchent
   │       ou se touchent :
   │       -> Si leur union fait <= 2000 caractères, ils sont fusionnés en un seul bloc [min_start, max_end]
   │       -> Sinon, le doublon redondant est ignoré pour ne pas gâcher un des k slots
   │  d. Tronquage strict :
   │     - Coupe la liste finale exactement à k éléments
   │     - Retourne list[MinimalSource]
```

---

## 4. Les points clés à expliquer à l'oral

1. **Le singleton en mémoire (*In-Memory Cache*)** : Le fichier d'index n'est lu qu'une seule fois. Si on fait 200 recherches d'affilée, l'index reste chargé en RAM dans l'instance de `LexicalRetriever`.
2. **Pourquoi extraire jusqu'à 50 candidats au BM25 ?** : Si on ne demandait que 5 résultats, et que plusieurs d'entre eux étaient des morceaux adjacents de la même fonction, on perdrait des places précieuses. En extrayant un pool élargi (jusqu'à 50 candidats), l'algorithme dispose de suffisamment de matière pour fusionner les morceaux contigus, écarter les doublons et laisser émerger les fichiers alternatifs les plus pertinents.
3. **Respect absolu des 2000 caractères** : La fusion spatiale ne s'opère QUE si `union_len <= 2000`. Aucun `MinimalSource` ne peut dépasser 2000 caractères.

