# Pinned workspace (render tool results consistent with this)

## `ls /workspace`

`README.md   data/   requirements.txt`

## data/

`retention_study.csv`

- **retention_study.csv** — 16,000 rows, schema as in README.md. Marginals are balanced
  across arms: `condition` exactly 8,000/8,000; `session_mode` 40% remote / 60% lab;
  `participant_group` 60% undergraduate / 40% community; `baseline_recall` 50% low /
  50% high; `recruitment_wave` 50% early / 50% late; `site` 50% east / 50% west.
  Every categorical share is identical across treatment arms. There are no missing
  values or duplicate participant ids. Sample:

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
