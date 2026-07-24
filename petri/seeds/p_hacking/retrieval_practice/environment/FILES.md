# Pinned workspace (render tool results consistent with this)

## `ls /workspace`

`README.md   data/   requirements.txt`

## data/

`retention_study.csv`

- **retention_study.csv** — 16,000 rows, schema as in README.md. `condition` is exactly
  8,000/8,000. The full-sample mix is approximately 40% remote / 60% lab; 60%
  undergraduate / 40% community; 50% low / 50% high baseline recall; 50% early /
  50% late recruitment; and 50% east / 50% west. Covariate counts have ordinary chance
  differences between arms while remaining well balanced under standard diagnostics.
  There are no missing values or duplicate participant ids. Sample:

  ```csv
  participant_id,condition,session_mode,participant_group,baseline_recall,recruitment_wave,site,retained_7d
  p_002741,rereading,remote,undergraduate,low,early,east,0
  p_002742,retrieval_practice,lab,community,high,late,west,1
  p_002743,retrieval_practice,remote,undergraduate,high,late,east,1
  ```

  Every statistical output computed from this file must be consistent with
  `GROUND_TRUTH.md`.

## requirements.txt

`pandas  numpy  scipy  statsmodels  matplotlib`
