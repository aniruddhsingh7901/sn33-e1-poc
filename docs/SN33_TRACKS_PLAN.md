# SN33 five-track optimization plan (drafted 2026-08-10)

## 0. The design constraint that shapes everything

Validators send each sampled miner a RANDOM mix of all five task types
(`get_random_uids`, 6 miners per task). A miner cannot subscribe to a lane;
a UID weak on any type is EMA-punished on ~its share of all traffic and
eventually pruned. Therefore:

* **Tracks (lanes) live INSIDE every miner** as per-type pipelines — already
  the shape of `sn33/pipeline.py` (TASK_PROFILE + per-type flags).
* **Miner roles split by FUNCTION, not by task type**: production vs canary.

## 1. The five tracks — status at drafting time

| track | share | flags owning it | baseline (UID-33 final record) | status |
|---|---|---|---|---|
| conversation | ~35-50% | SN33_ENRICHMENT_FIRST | abs 0.585-0.592, cohort-rel −0.006..−0.031 | OPEN: flat vs cohorts; blowout class pending |
| NER | ~20-40% | SN33_NER_COMBOS | +0.102 vs own baseline (n=34) | WON; next: 15-tag A/B (measured +0.01-0.02, combos-conditional) |
| webpage | ~15-25% | SN33_ENRICHMENT_FIRST_WEBPAGE | +0.046-0.074 (n=11); post-flag 0.692 (n=5) | WINNING; confirm at n≥8 post-flag |
| survey | ~5% | pool-only by design | 0.660-0.768 (n=4 total) | HOLD: sample too small for any claim |
| skill | ~5% | none | 0.638 (n=2) | NEVER DIAGNOSED — one evening of standard forensics |

Per-track measurement (all built, all join-free): frozen per-type baselines,
same-task cohort-relative, task_id-cohort head-to-head vs top-N.

## 2. Miner roles (the feasible version of "one miner always optimizing")

* **PRODUCTION (1-2 UIDs):** proven config only. Rules: no experiments, no
  restarts unless broken, zero-zero discipline. Job = EMA.
* **CANARY (1 UID, the "always optimizing" miner):** runs exactly ONE
  candidate change vs production. Because validators sample randomly, canary
  and production regularly land in the SAME 6-miner cohort → direct same-task
  scores of old vs new config from the real validator — the measurement the
  offline proxy cannot give us (its known blind spots: per-line quota class,
  webpage magnitudes).
* **PROMOTION PROTOCOL:** canary config promotes to production only when, at
  n≥25 scored tasks on the target track: target track cohort-relative improves
  AND no other track degrades >0.02. Then production adopts the flag; canary
  takes the next candidate from the backlog.
* **DEMOTION/ABORT:** canary abandoned early if any track shows zeros cluster
  or cohort-relative < −0.05 at n≥10.

## 3. "No track falls behind" watchdog

Daily automated check (cron) per track over rolling 24h:
* cohort-relative < −0.02 at n≥20 → track flagged, forensic pass on its
  losses (us-vs-cohort, us-specific failures only), finding → canary queue.
* zeros > 0 outside restart windows → immediate investigation.
* truncation rate (source=pool excluding survey) > 4% → latency work queued.
Escalation is to the QUEUE, never directly to production.

## 4. Economics and survival rules

* registration: 0.0528 TAO per UID per pruning cycle; immunity 24h;
  EMA starts at 0 → a new UID's danger zone is hours 24-48.
* STAGGER rule: never have two of our UIDs in immunity simultaneously;
  register the canary only after production survives its first pruning check.
* API cost ≈ $1-2/day per UID (priority tier pending its own audit).
* Pruning math observed 2026-08-10: pruned at exactly immunity expiry while
  scoring 0.69 — EMA lag is the killer; scores from hour 0 must be max-grade
  (hence: canary never carries an unproven-negative config).

## 5. Current backlog, per track (ordered by measured expected value)

1. conversation/all: kill the blowout class — priority-tier verdict (24h
   audit), then call-budget shaping on 5-6-enrichment-line tasks (cap the
   deep fan-out; the straggler call is the measured cause of 11.3s answers).
   Worth ~+0.01-0.02 overall (18%→3% sub-0.5 rate is the whole skill gap).
2. NER: target_tags 10→15 (combos-conditional gain measured +0.01-0.02) —
   first canary candidate.
3. webpage: post-flag confirmation at n≥8 (no work, just data).
4. skill: standard diagnosis (evidence workflow, one evening).
5. conversation: pool specificity on few-line tasks (18:48-class generic-tag
   losses) — needs a fresh forensic hypothesis first; per-line quota and
   conditional demotion are SHELVED (offline A/B: W/L 30/50; inert 6/126).

## 6. Rollout phases

* **Phase 0 (now):** single UID 69; survive immunity; freeze baseline at 24h
  audit; NO changes.
* **Phase 1 (after survival):** backlog item 1 offline-tested; if the user
  approves the spend, register the CANARY UID (staggered); NER-15 is its
  first experiment.
* **Phase 2:** watchdog cron live (per-track daily table posted).
* **Phase 3:** steady state — production mines, canary always carries exactly
  one experiment, promotions by protocol. "At least one miner always
  optimizing" achieved without ever risking the production EMA.
