# Commande : `answer_dataset`

## 1. À quoi sert cette commande ?

La commande `answer_dataset` est l'étape finale du pipeline RAG pour un dataset complet. 
Elle prend en entrée le fichier de résultats de recherche généré précédemment par `search_dataset` (conforme à `StudentSearchResults`), puis pour **chaque question** :
1. Extrait les morceaux de texte correspondants aux sources trouvées.
2. Demande à `Qwen/Qwen3-0.6B` de rédiger la réponse technique.
3. Sauvegarde l'ensemble des questions, sources et réponses dans un nouveau fichier JSON conforme au modèle imposé `StudentSearchResultsAndAnswer`.

---

## 2. Syntaxe CLI

```bash
uv run python -m src answer_dataset \
  --student_search_results_path <path_to_search_results.json> \
  --save_directory data/output/search_results_and_answer/<Scope> \
  [--model_name Qwen/Qwen3-0.6B]
```

### Paramètres :
- `--student_search_results_path` *(str, obligatoire)* : Chemin vers le fichier produit par `search_dataset`.
- `--save_directory` *(str, obligatoire)* : Dossier où sauvegarder le fichier final contenant les réponses.
- `--model_name` *(str, défaut : "Qwen/Qwen3-0.6B")* : Modèle LLM utilisé.

### Exemple concret :
```bash
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

---

## 3. Déroulement détaillé du code exécuté

```text
[Terminal]
   │
   ▼
1. src/__main__.py: main()
   │  Fire dispatche vers RAGCLI.answer_dataset()
   ▼
2. src/cli.py: RAGCLI.answer_dataset()
   │  - Valide que le fichier student_search_results_path existe
   │  - Appelle self.pipeline.answer_dataset(...)
   │  - Affiche : "Saved student_search_results_and_answer to <path>"
   ▼
3. src/pipeline.py: RagPipeline.answer_dataset()
   │  a. Lecture et validation :
   │     - Lit le fichier JSON d'entrée avec safe_read_json()
   │     - Valide avec StudentSearchResults.model_validate(raw_data)
   │  b. Initialisation unique du modèle :
   │     - Instancie self._get_generator(model_name)
   │     - Instancie self._get_context_builder(model_name)
   │     - Le modèle et le tokenizer sont chargés 1 seule fois en RAM
   │  c. Boucle de génération par question (avec barre tqdm) :
   │     Pour chaque item dans student_results.search_results :
   │       1. context = context_builder.build_context(item.retrieved_sources)
   │       2. gen_text = generator.generate_answer(question=item.question, context=context)
   │       3. Crée MinimalAnswer(
   │            question_id=item.question_id,
   │            question=item.question,
   │            retrieved_sources=item.retrieved_sources,
   │            answer=gen_text
   │          )
   │  d. Assemblage final et Sauvegarde :
   │     - Enveloppe la liste dans StudentSearchResultsAndAnswer(search_results=answers, k=student_results.k)
   │     - Détermine le chemin de sortie en conservant le nom de fichier d'origine
   │     - Écrit avec safe_write_json() garantissant l'écriture physique avec os.fsync
```

---

## 4. Format du fichier JSON généré (`StudentSearchResultsAndAnswer`)

Le fichier JSON final produit a la structure suivante :

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
      ],
      "answer": "The fused batched MoE layer returns the activation formats `mk.FusedMoEActivationFormat.BatchedExperts`."
    }
  ],
  "k": 10
}
```

---

## 5. Les points clés à expliquer à l'oral

1. **Pourquoi découper en 2 étapes (`search_dataset` puis `answer_dataset`) ?**
   - La recherche lexicale sur 100 questions prend **moins d'une seconde**.
   - La génération LLM sur CPU pour 100 questions prend plusieurs minutes (~4-5 secondes par question).
   - Ce découpage permet à l'évaluateur de vérifier immédiatement la qualité de la recherche (Recall@5) sans devoir attendre toute la génération de texte.
2. **Gestion de la RAM** : Le modèle est chargé une seule fois au début de la boucle et réutilisé pour chaque question, évitant les fuites mémoires ou les temps de rechargement.
3. **Le cache de fichier dans `ContextBuilder`** : Quand plusieurs questions ou plusieurs chunks renvoient au même gros fichier source, `ContextBuilder._file_cache` conserve le texte en mémoire au lieu de relire le disque à chaque fois.

