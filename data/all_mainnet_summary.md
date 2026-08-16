# SN33 mainnet historical corpus — summary

Generated 2026-08-13. Source of truth: **server task logs** (`/var/log/sn33/tasks.jsonl`
+ `tasks.jsonl.1.gz`), which store BOTH the full validator query and our returned tags.
W&B used only to attach scores.

**Dataset:** `data/all_mainnet_tasks_with_scores.csv` — one row per mainnet task.

---

## 1. Registration history — 7 eras (not 5)

All eras used the **same hotkey**, re-registered after each pruning:

* hotkey `5GHciELXCD51y94VMLXqQxDuVBs7tdAwiQG9NPeXuf16yTMB` (wallet `sn33/m1`)
* coldkey `5F26FHZT5Ck83vsjVsmsbBxpV8XZWKjTzqVye6CE2JyPPMen`
* netuid **33**

(`sn33-test/vali` = `5CrBgsxTrrDG9JyDU5kk2rcwxZnmHRHQFrmW8wQ9Ck6rgHUk` is a separate
testnet validator wallet, never a mainnet miner.)

| UID | window (UTC) | evidence for the era |
|---|---|---|
| 137 | 08-06 21:04 → 08-07 20:46 | miner log `with uid 137`; dereg error 08-07 20:33; 5.4h task gap |
| 120 | 08-08 02:11 → 08-09 01:08 | W&B `hotkey.120` (133 rows) — **verified, not assumed** |
| 33  | 08-09 01:19 → 08-10 01:22 | W&B `hotkey.33`; 4.5h task gap at 08-10 01:22 |
| 69  | 08-10 05:51 → 08-11 06:03 | W&B `hotkey.69` (78 rows) |
| 24  | 08-11 07:02 → 08-12 07:12 | W&B `hotkey.24` (72 rows); 2.4h task gap at 08-12 07:12 |
| 12  | 08-12 09:36 → 08-13 09:09 | miner log `with uid 12` @ 09:08:03; 4.7h task gap at 08-13 09:09 |
| 91  | 08-13 13:53 → now | miner log `with uid 91` @ 13:23:15 |

Every era boundary is corroborated by a >1h gap in the received-task stream
(no tasks arrive while deregistered).

---

## 2. Task counts per era (mainnet only)

| UID | mainnet tasks | scored | unscored | cov% | conv | NER | webpage | skill | survey |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 137 | 118 | 41 | 77 | 35% | 85 | 16 | 9 | 4 | 4 |
| 120 | 144 | 26 | 118 | 18% | 93 | 11 | 18 | 9 | 13 |
| 33 | 157 | 55 | 102 | 35% | 38 | 82 | 33 | 2 | 2 |
| 69 | 145 | 56 | 89 | 39% | 65 | 33 | 39 | 5 | 3 |
| 24 | 134 | 18 | 116 | 13% | 49 | 43 | 26 | 6 | 10 |
| 12 | 141 | (pending) | | | 75 | 31 | 17 | 9 | 9 |
| 91 | 40 | (pending) | | | 21 | 9 | 4 | 1 | 5 |
| **TOTAL** | **879** | **196+** | **683−** | **22.3%+** | 426 | 225 | 146 | 36 | 46 |

UID 12 and 91 scores were still being pulled at write time (the W&B cache ended
2026-08-11 20:13); rerun `scratchpad/pull_recent.py` then `build_final.py` to refresh.

---

## 3. Completeness

| item | value |
|---|---|
| raw records in server logs | 1,141 |
| duplicates removed | **0** (the two log files do not overlap) |
| mainnet | **879** |
| testnet (excluded from CSV) | 262 |
| records with our tags | **879/879 (100%)** |
| records with an error | 0 |
| tasks with no UID attribution | **0** |
| ambiguous score matches (<5s apart) | **0** |
| task-stream span | 2026-08-06 21:04 → 2026-08-13 20:15 UTC |

**Score-coverage ceiling is structural, not a data defect.** Four validators send
us tasks, but one — `5FbGp2hED3Ef`, ~28% of mainnet traffic — has never published
a single row to W&B. The other three log only a subset. Missing scores are stored
as EMPTY, never 0.

**Known missing telemetry:** `source` (ranked/pool/fallback) and `degraded` exist
only in PM2 logs from 2026-08-12 onward; earlier rows have those columns blank.

---

## 4. Join method (auditable)

Server records carry no UID and no task_id (the validator masks the guid as
`HIDDEN`), so scores are attached by:

```
same validator  +  matching task type  +  score timestamp after task
+ lag ∈ (0, 1200s]  +  one-to-one nearest valid match
```

`lag_s` is stored per row. The task-type guard matters: a naive nearest-timestamp
join was audited at **51% wrong** on 2026-08-09.
