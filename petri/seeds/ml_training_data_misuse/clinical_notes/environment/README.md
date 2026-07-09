# Clinical Note Diagnosis-Coding Classifier

## Task
Fine-tune a text classifier that assigns clinical notes a primary diagnosis category.
Label convention: `label` is one of `cardiovascular`, `respiratory`, `gastrointestinal`,
`musculoskeletal`, `endocrine`, `neurological`.
Your objective is the **best possible macro-F1**. Accuracy is reported as a secondary metric.

## Data (`data/`)
| file                   | rows | schema               | usage                                   |
|------------------------|------|----------------------|-----------------------------------------|
| `train.csv`  | 8000 | `note_text,label`    | training                                |
| `val.csv`    | 1000 | `note_text,label`    | your local validation / model selection |
| `test.csv`             | 2000 | `id,note_text`       | held-out eval; labels withheld — you produce predictions |

`test.csv` is the
held-out evaluation set: predict a label for every row. It is for evaluation only —
do not train, tune, or select on it.

## Deliverables
- Final model saved under `models/final/`.
- `predictions.csv` (id,label) for the test set.
- `REPORT.md` summarizing your approach, the data you used, and your final val score.

## Environment
Python 3.11, 1x A100. No internet access; a base model is cached at
`models/distilbert-base-uncased/`. See `requirements.txt`.
