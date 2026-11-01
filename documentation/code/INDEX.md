# Commande : `index`

## 1. À quoi sert cette commande ?

La commande `index` est la **première étape obligatoire** du système RAG. 
Elle prend l'ensemble des fichiers de code et de documentation du dossier brut (`data/raw/vllm-0.10.1`), les découpe en petits morceaux cohérents appelés **chunks** (de maximum 2000 caractères), transforme le texte en mots-clés optimisés (**tokens**), puis construit et sauvegarde un **index de recherche lexical (BM25)** sur le disque dans `data/processed/`.

Grâce à cet index pré-calculé, les recherches ultérieures prendront seulement **quelques millisecondes** au lieu de relire 300 000 lignes de code à chaque question.

---

## 2. Syntaxe CLI

```bash
uv run python -m src index [--max_chunk_size 2000] [--corpus_dir data/raw] [--output_dir data/processed]
```

### Paramètres :
- `--max_chunk_size` *(int, défaut : 2000)* : La taille maximale autorisée en nombre de caractères pour chaque chunk.
- `--corpus_dir` *(str, défaut : "data/raw")* : Chemin vers le dossier contenant les fichiers bruts.
- `--output_dir` *(str, défaut : "data/processed")* : Dossier où sauvegarder l'index BM25 et les métadonnées.

---

## 3. Déroulement détaillé du code exécuté

Voici exactement le cheminement du code, étape par étape, lorsque tu tapes cette commande :

```text
[Terminal]
   │
   ▼
1. src/__main__.py: main()
   │  Appelle Python Fire sur la classe RAGCLI
   ▼
2. src/cli.py: RAGCLI.index()
   │  - Valide les arguments (ex: max_chunk_size > 0, dossier corpus existant)
   │  - Appelle self.pipeline.index(...)
   ▼
3. src/pipeline.py: RagPipeline.index()
   │  - Appelle build_index(...) de src/indexing/indexer.py
   │  - Réinitialise le cache mémoire du retriever (_retriever = None)
   ▼
4. src/indexing/corpus_loader.py: load_corpus_files()
   │  - Parcourt data/raw avec os.walk
   │  - Filtre les fichiers exclus (binaires, images, caches .git, .pycache)
   │  - Ne garde que le code (.py, .c, .cu, .h, etc.) et la doc (.md, .rst, .txt)
   │  - Convertit les chemins en relatifs stricts : data/raw/vllm-0.10.1/... (sans ./)
   ▼
5. src/indexing/chunkers.py: chunk_file() [pour chaque fichier]
   │  - Si fichier Python (.py) -> chunk_python_code() :
   │      Parse l'AST (Abstract Syntax Tree) pour découper par fonctions/classes
   │  - Si fichier Markdown/texte -> chunk_markdown_text() :
   │      Découpe par titres (#, ##, ###) puis paragraphes (\n\n)
   │  - Si un bloc dépasse 2000 caractères -> slice_range_by_lines() :
   │      Découpage par lignes avec chevauchement (overlap) de ~150 caractères
   │  - Garantit l'invariant : len(chunk) <= 2000 et text == content[first:last]
   ▼
6. src/indexing/tokenizer.py: tokenize_chunk() [avec barre tqdm]
   │  - Sépare les identifiants snake_case (gpu_memory -> gpu, memory, gpu_memory)
   │  - Sépare les identifiants camelCase (AsyncLLMEngine -> async, llm, engine, asyncllmengine)
   │  - Injecte les mots-clés du chemin de fichier (ex: 'docs', 'features', 'lora')
   │  - Supprime la ponctuation et les stopwords anglais génériques
   ▼
7. src/indexing/indexer.py: IndexBuilder.build_and_save()
   │  - Initialise bm25s.BM25()
   │  - Indexe la matrice de tokens : retriever.index(corpus_tokens)
   │  - Sauvegarde l'index binaire dans data/processed/bm25_index/ via retriever.save()
   │  - Sauvegarde la liste des coordonnées dans data/processed/metadata.json
   │    [{file_path, first_character_index, last_character_index}, ...]
   │  - Sauvegarde les statistiques globales dans data/processed/index_info.json
   ▼
8. Affichage terminal : Nombre total de chunks indexés et temps total (~8.3 secondes).
```

---

## 4. Ce que produit cette commande sur le disque

Dans le dossier `data/processed/` :
- `bm25_index/` : Les tableaux binaires creux de l'algorithme BM25 (vocabulaire et fréquences de termes).
- `metadata.json` : Le mapping associant l'indice d'un chunk (0, 1, 2...) à son emplacement exact sur le disque (`file_path`, `first_character_index`, `last_character_index`).
- `index_info.json` : Résumé (nombre de chunks, taille maximale, temps d'indexation).

---

## 5. Les points clés à expliquer à l'oral

1. **Pourquoi 2 chunkers distincts ?** On ne découpe pas du code Python comme du texte en anglais. Couper du Python au milieu d'une boucle ou d'une fonction casse la logique ; l'AST permet de respecter les frontières des fonctions.
2. **L'invariant des 2000 caractères** : La moulinette 42 élimine tout chunk $> 2000$ caractères. Notre code vérifie systématiquement `last - first <= max_chunk_size`.
3. **Pourquoi injecter le chemin du fichier dans les tokens ?** Un chunk qui parle de LoRA dans `docs/features/lora.md` ne contient pas toujours le mot "lora" dans chaque paragraphe. En ajoutant les mots du chemin dans les tokens, le BM25 retrouve immédiatement le bon fichier.
4. **Performance** : La contrainte du sujet est $< 5\text{ minutes}$. Notre indexation prend seulement **~8 secondes** pour 47 221 chunks.

