# Commande : `evaluate`

## 1. À quoi sert cette commande ?

La commande `evaluate` est notre **outil d'auto-évaluation interne**. 
Elle compare un fichier de résultats produit par notre retriever (`StudentSearchResults`) avec le dataset de référence contenant les réponses attendues (`AnsweredQuestions`), calcule l'intersection spatiale (**IoU**) entre les extraits trouvés et les extraits de vérité terrain, et affiche les scores officiels de **Recall@1, Recall@3, Recall@5 et Recall@10**.

Elle reproduit fidèlement et à l'identique les calculs de la `moulinette` binaire fournie par 42.

---

## 2. Syntaxe CLI

```bash
uv run python -m src evaluate \
  --student_search_results_path <path_to_student_results.json> \
  --dataset_path <path_to_ground_truth_dataset.json>
```

### Paramètres :
- `--student_search_results_path` *(str, obligatoire)* : Chemin vers le fichier JSON produit par `search_dataset`.
- `--dataset_path` *(str, obligatoire)* : Chemin vers le fichier de référence `AnsweredQuestions`.

### Exemple concret :
```bash
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/AnsweredQuestions/dataset_code_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json
```

### Exemple de sortie terminal :
```text
Student data is valid: True
Evaluation Results
========================================
Questions evaluated: 99
Recall@1: 0.475 (47.5%)
Recall@3: 0.687 (68.7%)
Recall@5: 0.768 (76.8%)  (target: docs >= 80%, code >= 50%)
Recall@10: 0.859 (85.9%)
```

---

## 3. Déroulement détaillé du code exécuté

```text
[Terminal]
   │
   ▼
1. src/__main__.py: main()
   │  Fire dispatche vers RAGCLI.evaluate()
   ▼
2. src/cli.py: RAGCLI.evaluate()
   │  - Valide l'existence des deux fichiers
   │  - Appelle self.pipeline.evaluate(...)
   │  - Affiche le rapport texte formaté reçu
   ▼
3. src/pipeline.py: RagPipeline.evaluate()
   │  - Valide le fichier étudiant avec StudentSearchResults.model_validate()
   │  - Appelle evaluate_files(student_path, dataset_path) dans src/evaluation/evaluator.py
   │  - Formate le rapport avec format_evaluation_report()
   ▼
4. src/evaluation/evaluator.py: evaluate_files() & evaluate_dataset()
   │  a. Chargement des deux fichiers JSON :
   │     - student_data (StudentSearchResults)
   │     - dataset_data (RagDataset)
   │  b. Boucle sur chaque question de référence (q_id) :
   │     - Récupère les sources attendues (ground truth sources)
   │     - Récupère les sources trouvées par l'étudiant (student sources)
   │     - Pour chaque k dans (1, 3, 5, 10) :
   │       Appelle calculate_question_recall(retrieved[:k], ground_truth_sources)
   │  c. Calcul du Recall par question :
   │     - Pour chaque source attendue G :
   │       Vérifie si au moins UNE source trouvée R dans le top-k matche G
   │       -> is_matching_source(R, G)
   │  d. Calcul de l'IoU (Intersection over Union) 1D :
   │     - compute_iou(r_start, r_end, g_start, g_end, r_file, g_file) :
   │       1. Si r_file != g_file -> IoU = 0.0 (fichier différent = faux)
   │       2. intersection = max(0, min(r_end, g_end) - max(r_start, g_start))
   │       3. union = (r_end - r_start) + (g_end - g_start) - intersection
   │       4. IoU = intersection / union (si union > 0)
   │       5. Si IoU >= 0.05 (seuil bas de 5%) -> la source est considérée TROUVÉE !
   │     - question_recall = (nombre de sources G trouvées) / (nombre total de sources G)
   │  e. Moyenne générale :
   │     - Recall@k = moyenne des question_recall sur toutes les questions du dataset
```

---

## 4. Les points clés à expliquer à l'oral

1. **La formule de l'IoU 1D** :
   Contrairement à la vision par ordinateur où l'IoU compare des rectangles 2D (boîtes englobantes), ici l'IoU compare des **intervalles 1D de caractères** :
   $$\text{IoU} = \frac{\text{Intersection en caractères}}{\text{Union en caractères}}$$
2. **Le seuil de 0.05 (5 %)** :
   Le sujet n'impose pas de retrouver la tranche exacte au caractère près. Dès que notre chunk recouvre au moins **5 %** de la zone cible dans le bon fichier, la source est considérée comme trouvée.
3. **Différence entre `evaluate` et `moulinette`** :
   - Le sujet interdit formellement d'appeler le binaire `moulinette` dans notre code.
   - Nous avons donc ré-implémenté la formule mathématique exacte dans `evaluator.py`.
   - Lors des tests, notre commande `evaluate` donne **au millième près le même score** que `./moulinette evaluate_student_search_results`.

