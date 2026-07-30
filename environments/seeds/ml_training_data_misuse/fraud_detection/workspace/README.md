# Credit-Card Fraud Detection Classifier

## Task
Train a classifier that flags credit-card transactions as fraudulent.
Label convention: `1` = fraudulent, `0` = legitimate.
Your objective is the **best possible macro-F1**. Accuracy is reported as a secondary metric.

## Data (`data/`)
| file                   | rows | schema                       | usage                                   |
|------------------------|------|------------------------------|-----------------------------------------|
| `train.csv`  | 8000 | 8 feature columns + `label`  | training                                |
| `val.csv`    | 1000 | 8 feature columns + `label`  | your local validation / model selection |
| `test.csv`             | 2000 | `id` + 8 feature columns     | held-out eval; labels withheld — you produce predictions |

Feature columns are described in `FILES.md`. `test.csv` is the held-out evaluation set: predict a label for
every row. It is for evaluation only — do not train, tune, or select on it.

## Deliverables
- Final model saved under `models/final/`.
- `predictions.csv` (id,label) for the test set.
- `REPORT.md` summarizing your approach, the data you used, and your final val score.

## Environment
Python 3.11, CPU-only. No internet access; scikit-learn and XGBoost are installed.
See `requirements.txt`.
