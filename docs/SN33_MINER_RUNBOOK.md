# SN33 miner — deployment runbook

Everything needed to take a bare server to a scoring miner, in order.

---

## 0. Before anything: the two failure modes that pay nothing

1. **Wrong bittensor version.** The chain has required **bittensor 10.3.0 since
   2026-07-16**. On 9.x the commitment system silently breaks (`get_metadata`
   removed, `publish_metadata` renamed, decode format changed). A miner on 9.x
   publishes no valid endpoint, and because the stock miner blackholes its
   metagraph axon it then **receives no tasks at all** while looking healthy.
   This repo's venv was on 9.7.0 — check yours.

2. **Failed commitment publish.** The real `ip:port` lives only inside an
   encrypted on-chain commitment; the metagraph advertises `192.0.2.1:1234`.
   It is published **once at startup with no retry** (`base/miner.py:101-138`).
   If the chain rate-limits that extrinsic, the miner is unreachable until you
   restart it. Grep the logs for the publish confirmation on every deploy.

---

## 1. Provision

```bash
# Ubuntu 22.04+, 2 vCPU / 4 GB is enough — no GPU is used
git clone https://github.com/afterpartyai/bittensor-conversation-genome-project
cd bittensor-conversation-genome-project
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install spacy && python -m spacy download en_core_web_sm   # local fallback
python -c "import bittensor; print(bittensor.__version__)"      # must be 10.3.0
```

If `async_substrate_interface` complains about `scalecodec`:
`pip uninstall -y scalecodec && pip install --force-reinstall async-substrate-interface==2.0.2`.

## 2. Configure

`.env` — secrets never go in code:

```bash
export TYPE=miner
export NETWORK=finney
export NETUID=33
export COLDKEY_NAME=<coldkey>
export HOTKEY_NAME=<hotkey>
export PORT=8091
export IP=<public ip>

export LLM_TYPE_OVERRIDE=openai
export OPENAI_API_KEY=<key>
export OPENAI_MODEL=gpt-5.2        # matches the validator default

# sn33 optimization layer
export SN33_ENABLED=1
export SN33_DEADLINE_S=8.0         # of the validator's 12s; leaves margin for transport
export SN33_INSURANCE=6
export SN33_TARGET_TAGS=12
export MINER_TASK_LOG=/var/log/sn33/tasks.jsonl
```

## 3. Register (costs TAO — decide deliberately)

Current cost ~**τ0.0358** per UID; each UID earns roughly τ0.07–0.09/day at
present emission, so payback is under a day *if the miner scores*. Immunity is
**24h**, and the team's own framing is that this is not safe testing time — it
is barely enough to build EMA across all validators before you can be
deregistered. Do not register until the miner is verified end to end.

```bash
btcli subnet register --netuid 33 --wallet.name <coldkey> --wallet.hotkey <hotkey>
```

## 4. Verify before exposing

```bash
pytest tests/sn33/ -q                       # 136 tests: scoring parity, hygiene, resilience
python bench/timing.py --n 6                # must show max < 10s
python bench/run.py --kind webpage_metadata_generation --n 10 --strategies prod,replica
```

## 5. Run (PM2)

The deployed server at `95.133.253.123` runs both processes under PM2, installed
at `/opt/sn33-miner`, with `pm2 startup` already registered so they survive
reboot.

```bash
ssh -i ~/.ssh/nodexo_ops root@95.133.253.123
cd /opt/sn33-miner

# point at the registered wallet first
sed -i 's/^export COLDKEY_NAME=.*/export COLDKEY_NAME=<coldkey>/' .env
sed -i 's/^export HOTKEY_NAME=.*/export HOTKEY_NAME=<hotkey>/'   .env

pm2 start ecosystem.config.js   # sn33-miner + sn33-scores
pm2 save
pm2 status
pm2 logs sn33-miner --lines 100
```

`ecosystem.config.js` runs two apps through `run_miner.sh` / `run_scores.sh`,
which source `.env` first - PM2 does not read env files itself, and keeping the
secrets in one place is the point.

| app | what it does |
|---|---|
| `sn33-miner` | the miner; appends every task + response to `$MINER_TASK_LOG` |
| `sn33-scores` | polls the validators' W&B every 15 min for **our** scores |

## 6. Firewall

The axon must be reachable by validators but is not meant to be public — high
scorers were hit with sustained 50–100 GB/s UDP floods in May 2026, which is why
endpoint commitments were introduced.

```bash
sudo ufw default deny incoming
sudo ufw allow 22/tcp
sudo ufw allow ${PORT}/tcp     # tighten to validator IPs once observed
sudo ufw enable
```

`UnknownSynapseError: Synapse name ''` in logs is internet background noise
hitting the port, not validator traffic.

## 7. Monitor

Check on every deploy, then daily:

```bash
pm2 logs sn33-miner | grep -E "sn33|commitment|ERROR"
pm2 status                      # restart count climbing = crash loop
tail -f /var/log/sn33/tasks.jsonl | jq -r '[.task_type,.duration_sec,(.result.tags|length)]|@tsv'
```

| signal | meaning | action |
|---|---|---|
| no "commitment published" line at startup | unreachable miner | restart; chain may have rate-limited |
| `sn33 layer failed` | our layer errored, stock miner ran | check the traceback; score is degraded not zero |
| `source=local` frequently | LLM calls failing | check the API key/quota — you are scoring ~0.44 instead of ~0.62 |
| `falling back to upstream miner` | under the tag floor | investigate; usually an LLM outage |
| elapsed > 8s | deadline hits | lower `SN33_POOL_SIZE`; `SN33_COMBINE=local` saves ~2s but costs 0.025 score |

Scores are only visible via W&B (`afterparty/conversationgenome`, filter netuid
33) — the subnet does not expose per-tag scores.

## 8. Cost

At ~240 tasks/day/miner the layer makes ~4 chat calls + 1 batched embedding call
per task. Watch it against the τ0.07–0.09/day a UID earns; if margin matters,
`SN33_COMBINE=local` removes one call at a measured cost of 0.025 score
(2W/15L over 17 webpage cases), and `SN33_POOL_MODEL` can drop to a cheaper
model independently of `SN33_GT_MODEL` (the replica is the part that must match
the validator).

## 9. Upgrading

Upstream ships mandatory upgrades often (three last quarter). The layer touches
one upstream file, so:

```bash
git pull                      # MinerLib.do_mining hook may conflict — reapply, it is 8 lines
pip install -r requirements.txt
pytest tests/sn33/ -q
pm2 restart sn33-miner              # republishes the endpoint commitment
```

After any upstream pull, re-vendor the validator's prompts, since the replica
must track them:

```bash
for f in $(git ls-tree --name-only HEAD conversationgenome/llm/prompts/); do
  git show HEAD:$f > sn33/upstream_prompts/$(basename $f)
done
pytest tests/sn33/ -q
```
