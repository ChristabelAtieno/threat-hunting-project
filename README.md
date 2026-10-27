# CloudTrail Threat Hunting

Detects suspicious AWS activity in CloudTrail and turns it into incidents a SOC analyst can act on.

```
CloudTrail (local files | S3 trail bucket | LookupEvents API, all regions)
  └─ CloudTrailFeatures      time, errors, MFA/root, per-identity history, hourly bursts, behavioural surprisal,
                              graph (role assumptions, shared IPs, peer communities)
       └─ CloudTrailDetector
            ├─ Isolation Forest       unusual combinations for this account
            └─ Behavioural baseline   novel use of an identity (new IP / API / region / client / hour)
  └─ Detection rules          known-bad: log tampering, root, no-MFA login, IAM persistence, exposure, bursts (MITRE ATT&CK)
  └─ Incidents                one per identity per hour, with severity and plain-English reasons
```

The two detectors complement each other. Rules catch dangerous activity even when it is routine-looking.
The model catches activity nobody wrote a rule for. Rule hits on work an identity does routinely
(e.g. CI attaching role policies daily) only alert when the model also finds the event unusual.

## Setup

```bash
python -m venv venv && venv/Scripts/activate      # Linux/macOS: source venv/bin/activate
pip install -r requirements-dev.txt
pytest                                              # offline tests
```

AWS credentials come from the standard chain (env vars, `~/.aws`, SSO, or an IAM role), never from code.
Minimum permissions: [deploy/iam/scorer-policy.json](deploy/iam/scorer-policy.json).

MLflow stores models. Locally the default store works. For production, run a shared tracking
server and set `MLFLOW_TRACKING_URI` everywhere (training, scoring, container).

## 1. Train

The model learns what *normal* looks like, so train it on **your own account's** history.
The flaws.cloud dataset is useful for development but its baseline is attacker activity.

```bash
python main.py train --source lookup --days 30              # LookupEvents, all regions (management events)
python main.py train --source s3 --s3-uri s3://TRAIL-BUCKET/AWSLogs/ACCOUNT_ID/CloudTrail/
python main.py train --path extracted/flaws_cloudtrail_logs  # local *.json.gz (also the default)
python main.py train --source parquet --path processed_parquet_lookup/events.parquet  # retrain from a saved snapshot
```

Training from `lookup` or `s3` saves a snapshot of exactly what the model saw to
`processed_parquet_<source>/events.parquet`. This tool's own CloudTrail reads and any attack
simulation traffic are excluded from training, so neither becomes "normal".

Training does this:
1. Splits the data by time and trains on the earlier 80%.
2. Injects 7 labelled attack scenarios into the later 20%: stolen-key recon, privilege escalation,
   logging tampering, crypto-mining in an unused region, permission probing, root login without
   MFA, and a key used from a new IP.
3. Reports what was caught and how noisy normal activity is (`incidents/day`).
4. Retrains on all the data, registers the model in MLflow and **promotes it to
   `models:/isolation_forest_model@champion` only if at least 80% of scenarios are caught**
   (`--min-detection`). Otherwise it stays registered and you can promote it by hand:
   `python main.py promote --version N`.

The evaluation report is saved to the MLflow run as `evaluation.json`. If the account has no
established human or workload identity (only AWS service-linked roles, for example), the two
scenarios that need a realistic baseline are reported as `N/A` and left out of the promotion
score instead of counting as misses.

### Hyperparameter tuning

```bash
python main.py train --source lookup --days 30 --tune 12 --alert-budget 0.02
```

There are no labels, so tuning uses the evaluation as its scoring function. Each candidate is trained on
the earlier data and ranked by: (1) staying within the **alert budget** (share of normal events
that alert), (2) attack scenarios caught, (3) attack events the **model alone** catches (the part
these settings control) and (4) fewer alerts. The current defaults are always trial 0. Each trial
is a nested MLflow run, and the full comparison is saved as `tuning.json`.

Searched: `n_estimators`, `max_samples`, `max_features` and the two contamination settings
(how alert volume is split between the Isolation Forest and the behavioural baseline). Without
`--tune`, `CONTAMINATION` and `BEHAVIOUR_CONTAMINATION` (default 0.005 each) can be set directly.

Treat the alert budget as a SOC decision (how many alerts analysts can review), not a model setting.
Retune whenever the account's activity changes a lot, for example after onboarding new teams.

### Graph features

CloudTrail is a graph of identities, roles, IPs and services. `scripts/graph_features.py` adds what
single-event features cannot see:

| Feature | Catches |
|---|---|
| Role surprisal, first-time role edge, role chaining, cross-account hop | Lateral movement and privilege hops, e.g. a user who always assumes `readonly` suddenly assumes `ci-deployer` |
| Identities per IP (normally, and within the hour) | One attacker machine using several stolen keys |
| Peer communities (Louvain over shared services) | Touching a service none of the identity's peers use |
| New connections per identity per hour | Account takeover forming many new edges at once |

Evaluated by ablation (same data and scenarios, with and without the graph features): the
`role_hop_known_role` scenario went from 67% to 100% detected, with no other scenario or the
alert rate getting worse. The rule `new_role_assumption_path` (MITRE T1078.004) flags first-time
role chaining or cross-account assumption.

### Weak labels and analyst verdicts

Every training run also produces **weak labels** (`weak_labels/<run_id>/weak_labels.parquet`, and
the MLflow artifact `weak_labels/`). They are the training data for a future supervised triage model.

- **Labelling functions** vote attack / benign / abstain: the two models, graph, critical rules,
  IAM persistence, burst rules, and benign voters (routine work, AWS service activity, calm behaviour).
- **Votes are combined by family** (models, graph, rules, routine, AWS service), because correlated
  sources such as the two models would otherwise double-count the same evidence.
- **Combiner:** a Dawid-Skene label model or a plain majority vote, whichever scores higher on gold
  labels. On a tie the simpler majority vote wins. A tie between attack and benign evidence is labelled benign.
- **Gold labels:** injected attacks count as attacks, and real events from identities seen in
  training are assumed benign. Identities that are new since training stay unlabelled, because their truth
  is unknown. Analyst verdicts override everything.

Record analyst verdicts, which are the most valuable labels you can collect:

```bash
python main.py verdict --incident-id "<id from incidents.jsonl>" --verdict malicious --analyst sam --note "confirmed key theft"
python main.py verdict --incident-id "<id>" --verdict benign --note "planned pentest"
```

Verdicts are stored in `verdicts.csv` (or `VERDICTS_FILE`) and used as gold labels on the next training run.
Use `--no-weak-labels` to skip labelling.

### Identities

Assumed-role sessions are grouped by **role**, because session names are often random
(`botocore-session-…`, instance IDs). The exceptions: SSO roles (`AWSReservedSSO_*`) keep one
baseline per person, and sessions with a `sourceIdentity` are tracked as role + that person.
Incidents still show the full session ARN in their events.

## 2. Score

```bash
python main.py score --hours 1 --regions all \
    --state-file state.json --output incidents.jsonl [--csv events.csv] [--fail-on-high]
```

- `--regions all` queries every enabled region, because attackers use regions nobody watches.
- `--state-file` remembers the last scored event, so a job on a schedule never re-alerts.
- `--output` appends one JSON incident per line. That format is ready for a SIEM, S3, Slack and similar.
- `--fail-on-high` exits with code 2 when a high or critical incident exists, so the scheduler can page.

Example incident:

```json
{"severity": "high", "actor": "arn:aws:iam::111122223333:user/alice",
 "first_seen": "2026-01-20 02:00:00+00:00", "event_count": 20,
 "detected_by": ["model:behaviour_baseline", "rule:recon_burst"], "mitre": ["T1580"],
 "reasons": ["20+ distinct APIs by one identity within an hour (enumeration)",
             "source IP 203.0.113.66 never seen in this account",
             "first time this identity used client aws-cli",
             "identity is never active at 02:00 UTC"],
 "source_ips": ["203.0.113.66"], "regions": ["us-east-1"], "api_calls": {"ListUsers": 1, "...": 1},
 "event_ids": ["..."]}
```

Each run also prints a summary that includes drift signals (`unseen_identity_rate`, `unseen_ip_rate`).
If the model flags far more than it was trained to, the run logs a warning to retrain.

## 3. Live test in AWS

```bash
python main.py simulate                         # dry run: lists every call it would make
python main.py simulate --i-own-this-account    # runs it and prints a run id
# wait ~15 minutes for CloudTrail to surface the events
python main.py check-simulation --run-id <id>   # which simulated events were alerted, and the incidents
pytest -m live                                  # fetch + score smoke tests against real AWS
```

The simulator makes read-only calls, plus dangerous calls (`StopLogging`, `DeleteTrail`,
`CreateAccessKey`, `AttachUserPolicy`, `CreateLoginProfile`) aimed at **random names that do
not exist**. Those fail and nothing is created, changed or deleted, but CloudTrail still records
them. Every call is tagged `threat-hunt-sim/<run_id>` in its user agent. Run it only in an
account you own, and tell whoever watches that account first: it is designed to look like an attack.

## 4. Deploy

```bash
docker build -t threat-hunt .
docker run --rm -e AWS_REGION=us-east-1 -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
    -v $(pwd)/state:/app/state threat-hunt          # default command: score the last hour
```

Run it every 15–60 minutes, for example as an ECS scheduled task, an EventBridge-triggered task,
a Kubernetes CronJob or plain cron. Give it an IAM role with the scorer policy and keep
`/app/state` on persistent storage.

### GitHub Actions (scheduled)

| Workflow | When | What |
|---|---|---|
| `score.yml` | every 15 min | scores new events with the champion; fails the run (GitHub emails you) on high/critical incidents |
| `retrain.yml` | Mondays 03:00 UTC | trains on 30 days with 12 tuning trials and the analyst verdicts; promotes only if the evaluation gate passes |

The repo is public, so logs and artifacts are world-readable. Everything sensitive lives in a
private S3 bucket under `threat-hunt/`: `mlflow.db` (registry), `mlartifacts/` (models),
`incidents/`, `logs/`, `state/`, `weak_labels/`, `snapshots/` and `verdicts.csv`. The logs show counts only.

One-time setup:

1. Create a private S3 bucket (block public access on).
2. AWS credentials, preferably an OIDC role: add GitHub's OIDC provider in IAM, create a role with
   `deploy/iam/github-oidc-trust.json` as the trust policy and `deploy/iam/github-actions-policy.json`
   as permissions (replace the placeholders), and set the repo variable `AWS_ROLE_ARN`.
   Otherwise, add the secrets `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` for a user with that policy.
3. Repo variables (Settings → Secrets and variables → Actions → Variables): `THREAT_HUNT_BUCKET`,
   and optionally `AWS_REGION` (default `us-east-1`).
4. Run **retrain** once by hand (Actions → retrain → Run workflow). Scoring needs a champion.

Analyst loop:

```bash
aws s3 cp s3://BUCKET/threat-hunt/incidents/2026/10/26/ . --recursive   # read incidents
cat *.jsonl > incidents.jsonl
aws s3 cp s3://BUCKET/threat-hunt/verdicts.csv verdicts.csv              # (if it exists)
python main.py verdict --incident-id ID --verdict benign --analyst me
aws s3 cp verdicts.csv s3://BUCKET/threat-hunt/verdicts.csv             # next retrain uses it
```

Notes: GitHub may start scheduled runs several minutes late (the state file means no events are
missed or alerted twice), and pauses schedules on public repos after 60 days without a commit.

## Production checklist

- [ ] Train on 30–90 days of your own account and review `evaluation.json`.
- [ ] Run the live simulation and confirm the scenarios are caught.
- [ ] Decide where incidents go (SIEM / Slack / ticketing) and wire `incidents.jsonl` to it.
- [ ] Run a shared MLflow tracking server (with an S3 artifact store).
- [ ] Schedule scoring with persistent state and an IAM role.
- [ ] Schedule retraining (e.g. weekly), which promotes automatically only when the evaluation passes.
- [ ] Review the first weeks of incidents with analysts, then tune the contamination settings and rule thresholds in `scripts/detections.py`.
- [ ] For organisations with high event volume, read from the trail's S3 bucket instead of LookupEvents (2 req/s limit per region).

## Limitations

- LookupEvents returns **management events only**. S3 object access and Lambda invokes need
  data events from a trail's S3 bucket.
- The evaluation scenarios are synthetic attack shapes, not ground truth. Use them to catch
  regressions, and use the live simulation and analyst feedback to measure real performance.
- Burst features count activity per identity per clock hour. Scoring fetches from the top of the
  hour so the counts are complete.

## Layout

| Path | Purpose |
|---|---|
| `main.py` | CLI: `train`, `score`, `verdict`, `promote`, `simulate`, `check-simulation` |
| `scripts/load_data.py` | Data sources: local, S3, LookupEvents |
| `scripts/features_engineer.py` | Feature transformer (fit on training, reused at inference) |
| `scripts/anomaly_model.py` | Isolation Forest + behavioural baseline |
| `scripts/detections.py` | Rules, MITRE mapping, severity, reasons |
| `scripts/graph_features.py` | Role-assumption, shared-IP and peer-community graph |
| `scripts/tuning.py` | Hyperparameter search against the alert budget |
| `scripts/weak_labels.py` | Labelling functions, label model, gold-label evaluation, verdicts |
| `scripts/inference.py` | Scoring, incident grouping, state |
| `scripts/evaluate.py` | Time split + injected attack scenarios |
| `scripts/train_models.py` | Training, MLflow logging, registry, promotion gate |
| `scripts/simulate_attacks.py` | Safe live attack simulation and check |
| `notebook/` | Original exploration |
