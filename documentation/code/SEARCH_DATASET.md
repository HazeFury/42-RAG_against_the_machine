# Commande : `search_dataset`

## 1. À quoi sert cette commande ?

La commande `search_dataset` prend en entrée un fichier JSON contenant une liste de questions (par exemple `data/datasets/AnsweredQuestions/dataset_code_public.json`), exécute la recherche lexicale pour **toutes les questions à la suite** avec une barre de progression `tqdm`, et sauvegarde le résultat dans un fichier JSON structuré selon le modèle imposé `StudentSearchResults`.

C'est ce fichier généré qui est ensuite évalué par la `moulinette` officielle de 42 pour calculer le **Recall@5**.

---

## 2. Syntaxe CLI

```bash
uv run python -m src search_dataset \
  --dataset_path <path_to_dataset.json> \
  [--k 10] \
  [--save_directory data/output/search_results/<Scope>] \
  [--index_dir data/processed]
```

### Paramètres :
- `--dataset_path` *(str, obligatoire)* : Chemin vers le fichier dataset JSON (`AnsweredQuestions` ou `UnansweredQuestions`).
- `--k` *(int, défaut : 10)* : Nombre de sources à récupérer par question (10 pour la soutenance).
- `--save_directory` *(str, défaut : "data/output/search_results")* : Dossier où écrire le fichier de résultat.
- `--index_dir` *(str, défaut : "data/processed")* : Dossier de l'index BM25.

### Exemple concret :
```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```

---

## 3. Déroulement détaillé du code exécuté

```text
[Terminal]
   │
   ▼
1. src/__main__.py: main()
   │  Fire dispatche vers RAGCLI.search_dataset()
   ▼
2. src/cli.py: RAGCLI.search_dataset()
   │  - Valide que dataset_path est un fichier existant et que k > 0
   │  - Appelle self.pipeline.search_dataset(...)
   │  - Affiche le message de confirmation : "Saved student_search_results to <path>"
   ▼
3. src/pipeline.py: RagPipeline.search_dataset()
   │  a. Lecture et validation Pydantic :
   │     - Lit le fichier JSON avec safe_read_json(dataset_path)
   │     - Valide le contenu avec RagDataset.model_validate(raw_data)
   │     - Extrait la liste des questions (list[str]) et des IDs (list[str])
   │  b. Récupération par lot :
   │     - Récupère le retriever en mémoire via self._get_retriever(index_dir)
   │     - Appelle retriever.retrieve_batch(queries, k=k, show_progress=True)
   ▼
4. src/retrieval/lexical_retriever.py: LexicalRetriever.retrieve_batch()
   │  - Tokenise l'ensemble des questions
   │  - Récupère les scores BM25 par lot pour une vitesse maximale
   │  - Applique la déduplication spatiale pour chaque question avec barre tqdm
   │  - Renvoie une liste de listes : list[list[MinimalSource]]
   ▼
5. src/pipeline.py: Assemblage et Sauvegarde
   │  - Pour chaque question, crée un objet MinimalSearchResults(question_id, question, retrieved_sources)
   │  - Enveloppe le tout dans StudentSearchResults(search_results=..., k=k)
   │  - Conserve le nom de fichier original (dataset_docs_public.json)
   │  - Crée automatiquement les dossiers parents si nécessaire
   │  - Écrit le JSON avec safe_write_json(out_file, student_res.model_dump())
   │    -> Utilise f.flush() et os.fsync(f.fileno()) pour garantir l'écriture physique immédiate sur disque
```

---

## 4. Format du fichier JSON généré (`StudentSearchResults`)

Le fichier produit ressemble exactement au schéma attendu par la `moulinette` :

```json
{
  "search_results": [
    {
      "question_id": "189c8b8a-e59c-4fca-92ad-c02df42cbe40",
      "question": "What activation formats does the fused batched MoE layer return in vLLM?",
      "retrieved_sources": [
        {
          "file_path": "data/raw/vllm-0.10.1/vllm/model_executor/layers/fused_moe/fused_batched_moe.py",
          "first_character_index": 28412,
          "last_character_index": 28661
        }
      ]
    }
  ],
  "k": 10
}
```

---

## 5. Les points clés à expliquer à l'oral

1. **Vitesse de traitement par lot (*Batch Processing*)** : Le sujet demande de traiter 200 questions en moins de **90 secondes**. Notre implémentation traite les 100 questions en **~0,45 seconde** car l'index n'est chargé qu'une seule fois en RAM et BM25s opère des calculs vectorisés en C/NumPy.
2. **Scoping des dossiers** : Il est essentiel de ranger les sorties dans un sous-dossier `UnansweredQuestions/` ou `AnsweredQuestions/`, car les deux datasets partagent le même nom de fichier (`dataset_docs_public.json`) et s'écraseraient sinon.
3. **Robustesse I/O** : L'utilisation de `os.fsync` empêche les soucis de cache Linux où l'explorateur de l'IDE ne verrait pas le fichier immédiatement.

