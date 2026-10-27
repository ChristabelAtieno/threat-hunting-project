"""
Weak supervision: turn several imperfect detectors into probabilistic attack / benign labels.

Labelling functions (LFs) each look at an event and vote ATTACK (1), BENIGN (0) or ABSTAIN (-1).
They are deliberately different kinds of evidence (the two anomaly models, high-precision rules,
burst rules, the graph, routine work, AWS's own service activity), because labels built only from
anomaly detectors would teach a supervised model "unusual", not "malicious".

The label model (Dawid-Skene, fitted with EM) learns how reliable each LF is from how the LFs agree
and disagree, without any labels, and returns P(attack) per event. It assumes LFs err independently;
correlated LFs (the two models, the two burst rules) make it somewhat overconfident, which is why its
probabilities are judged against gold labels and not taken at face value.

Gold labels (injected attacks, live simulations, analyst verdicts) are only used to judge LFs and the
label model, never as votes, so the judging stays honest.
"""
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.detections import ROUTINE_CALLS, _count
from scripts.features_engineer import prepare_events
from scripts.inference import score_events

logger = logging.getLogger(__name__)

ATTACK, BENIGN, ABSTAIN = 1, 0, -1
CRITICAL_RULES = {'cloud_logging_tampering', 'root_account_activity', 'console_login_without_mfa'}
BURST_RULES = {'recon_burst', 'error_burst'}
VERDICT_COLUMNS = ['event_id', 'verdict', 'analyst', 'recorded_at', 'note']


def _has_rule(scored, names):
    return scored['rules'].map(lambda rules: bool(names & set(rules))).to_numpy()


def _vote(attack=None, benign=None, n=None):
    votes = np.full(n, ABSTAIN, dtype=np.int8)
    if benign is not None:
        votes[np.asarray(benign, dtype=bool)] = BENIGN
    if attack is not None:
        votes[np.asarray(attack, dtype=bool)] = ATTACK
    return votes


def labelling_functions(scored, features, prepared):
    """Returns {name: votes} for events scored by inference.score_events (same row order, sort=False)."""
    n = len(scored)
    routine = ((features['actor_frequency'] > 0)
               & (_count(features['actor_eventName_frequency']) >= ROUTINE_CALLS)
               & (features['actor_sourceIPAddress_frequency'] > 0)
               & (features['actor_userAgentFamily_frequency'] > 0)
               & (features['has_error'] == 0)).to_numpy()
    aws_service = (prepared['userIdentity.type'].eq('AWSService')
                   | prepared['actor'].str.contains(':assumed-role/AWSServiceRoleFor', regex=False)).to_numpy()
    no_rules = scored['rules'].map(len).eq(0).to_numpy()
    graph = ((_has_rule(scored, {'new_role_assumption_path'}))
             | ((_count(features['ip_identities_in_hour']) >= 3) & (features['sourceIPAddress_frequency'] == 0)).to_numpy()
             | (_count(features['actor_new_edges_in_hour']) >= 10).to_numpy())
    return {
        # models: vote attack when they flag, abstain otherwise ("not flagged" is weak evidence of benign)
        'isolation_forest': _vote(attack=scored['forest_anomaly'], n=n),
        'behaviour_baseline': _vote(attack=scored['behaviour_anomaly'], n=n),
        # AWS services act from many internal endpoints and form "new connections" constantly
        'graph': _vote(attack=graph & ~aws_service, n=n),
        # rules
        'critical_rules': _vote(attack=_has_rule(scored, CRITICAL_RULES), n=n),
        'iam_persistence': _vote(attack=_has_rule(scored, {'iam_persistence_or_privilege_change'}) & ~routine, n=n),
        'burst_rules': _vote(attack=_has_rule(scored, BURST_RULES), n=n),
        # benign evidence
        'routine_work': _vote(benign=routine & no_rules, n=n),
        'aws_service_activity': _vote(benign=aws_service & no_rules, n=n),
        'calm_behaviour': _vote(benign=(features['behaviour_surprisal'].to_numpy()
                                        < 0.5 * scored.attrs['surprisal_threshold']) & ~scored['forest_anomaly'].to_numpy(),
                                n=n),
    }


def apply_lfs(events, model):
    """Scores events and returns (scored, votes DataFrame with one column per LF)."""
    scored = score_events(events, model, sort=False)
    scored.attrs['surprisal_threshold'] = model.named_steps['detector'].surprisal_threshold_
    features = model.named_steps['features'].transform(events).reset_index(drop=True)
    prepared = prepare_events(events).reset_index(drop=True)
    return scored, pd.DataFrame(labelling_functions(scored, features, prepared))


class LabelModel:
    """
    Dawid-Skene with abstains: each LF j has P(vote | true class) over {attack, benign, abstain}.
    EM alternates between estimating P(attack) per event and each LF's confusion table.
    """

    def __init__(self, n_iter=100, smoothing=1.0, tol=1e-6):
        self.n_iter, self.smoothing, self.tol = n_iter, smoothing, tol

    def fit(self, votes):
        V = votes.to_numpy()
        self.lfs_ = list(votes.columns)
        # start from a majority vote among non-abstaining LFs (no votes -> 0.5)
        attack, benign = (V == ATTACK).sum(1), (V == BENIGN).sum(1)
        q = np.where(attack + benign > 0, (attack + 0.5) / (attack + benign + 1), 0.5)
        for _ in range(self.n_iter):
            self._m_step(V, q)
            new_q = self._e_step(V)
            if np.max(np.abs(new_q - q)) < self.tol:
                q = new_q
                break
            q = new_q
        return self

    def _m_step(self, V, q):
        self.prior_ = float(np.clip(q.mean(), 1e-4, 1 - 1e-4))
        self.confusion_ = {}
        for j, name in enumerate(self.lfs_):
            table = {}
            for cls, weight in ((ATTACK, q), (BENIGN, 1 - q)):
                counts = np.array([weight[V[:, j] == v].sum() for v in (ATTACK, BENIGN, ABSTAIN)]) + self.smoothing
                table[cls] = (counts / counts.sum()).tolist()
            self.confusion_[name] = table

    def _e_step(self, V):
        index = {ATTACK: 0, BENIGN: 1, ABSTAIN: 2}
        log_attack = np.full(len(V), np.log(self.prior_))
        log_benign = np.full(len(V), np.log(1 - self.prior_))
        for j, name in enumerate(self.lfs_):
            cols = np.vectorize(index.get)(V[:, j])
            log_attack += np.log(np.asarray(self.confusion_[name][ATTACK])[cols])
            log_benign += np.log(np.asarray(self.confusion_[name][BENIGN])[cols])
        return 1 / (1 + np.exp(np.clip(log_benign - log_attack, -50, 50)))

    def predict_proba(self, votes):
        return self._e_step(votes[self.lfs_].to_numpy())

    def to_dict(self):
        return {'prior_attack': self.prior_, 'confusion': self.confusion_,
                'note': 'confusion[lf][true_class] = P(vote = attack, benign, abstain)'}


# LFs that look at the same evidence are correlated: counting them separately double-counts that
# evidence (two model votes would outvote one independent benign vote). Combiners therefore see one
# vote per family: attack if any member votes attack, else benign if any votes benign, else abstain.
LF_FAMILIES = {
    'models': ['isolation_forest', 'behaviour_baseline', 'calm_behaviour'],
    'graph': ['graph'],
    'rules': ['critical_rules', 'iam_persistence', 'burst_rules'],
    'routine': ['routine_work'],
    'aws_service': ['aws_service_activity'],
}


def family_votes(votes):
    families = {}
    for family, members in LF_FAMILIES.items():
        V = votes[[m for m in members if m in votes.columns]].to_numpy()
        families[family] = np.where((V == ATTACK).any(1), ATTACK, np.where((V == BENIGN).any(1), BENIGN, ABSTAIN))
    return pd.DataFrame(families, index=votes.index).astype(np.int8)


def majority_vote_proba(votes, prior=0.01):
    """Share of non-abstaining LFs voting attack; events nobody voted on get the prior."""
    V = votes.to_numpy()
    attack, benign = (V == ATTACK).sum(1), (V == BENIGN).sum(1)
    return np.where(attack + benign > 0, attack / np.maximum(attack + benign, 1), prior)


COMBINERS = ('label_model', 'majority_vote')


def _prf(predicted, gold):
    tp = int((predicted & gold).sum())
    precision = tp / max(int(predicted.sum()), 1)
    recall = tp / max(int(gold.sum()), 1)
    return {'precision': round(precision, 3), 'recall': round(recall, 3),
            'f1': round(2 * precision * recall / max(precision + recall, 1e-9), 3)}


def lf_summary(votes, gold=None):
    """Snorkel-style LF report: coverage, overlaps, conflicts and, where gold exists, accuracy."""
    V = votes.to_numpy()
    voted = V != ABSTAIN
    summary = {}
    for j, name in enumerate(votes.columns):
        others = np.delete(V, j, axis=1)
        conflict = voted[:, j] & ((others != ABSTAIN) & (others != V[:, [j]])).any(1)
        row = {'polarity': sorted({'attack' if v == ATTACK else 'benign' for v in V[voted[:, j], j]}),
               'coverage': round(float(voted[:, j].mean()), 4),
               'overlaps': round(float((voted[:, j] & (np.delete(voted, j, axis=1)).any(1)).mean()), 4),
               'conflicts': round(float(conflict.mean()), 4)}
        if gold is not None:
            g = gold.notna().to_numpy() & voted[:, j]
            row['gold_accuracy'] = round(float((V[g, j] == gold.to_numpy()[g]).mean()), 3) if g.any() else None
        summary[name] = row
    return summary


def compare_to_gold(votes, probabilities, gold):
    """Precision/recall/F1 on gold labels for the label model, a majority vote (by family) and each LF alone."""
    known = gold.notna().to_numpy()
    y = gold.to_numpy()[known].astype(int) == ATTACK
    V = votes.to_numpy()[known]
    F = family_votes(votes).to_numpy()[known]
    attack, benign = (F == ATTACK).sum(1), (F == BENIGN).sum(1)
    report = {'label_model': _prf(probabilities[known] > 0.5, y),
              'majority_vote': _prf(attack > benign, y),
              'gold_events': int(known.sum()), 'gold_attacks': int(y.sum())}
    report['lfs'] = {name: _prf(V[:, j] == ATTACK, y) for j, name in enumerate(votes.columns)
                     if (V[:, j] == ATTACK).any() or name in ('isolation_forest', 'behaviour_baseline')}
    best_single = max(r['f1'] for r in report['lfs'].values()) if report['lfs'] else 0.0
    report['best_single_lf_f1'] = best_single
    report['label_model_beats_best_lf'] = report['label_model']['f1'] > best_single
    # Use the label model only when it beats a plain majority vote; on a tie the simpler method wins
    report['chosen_combiner'] = ('label_model' if report['label_model']['f1'] > report['majority_vote']['f1']
                                 else 'majority_vote')
    chosen_f1 = report[report['chosen_combiner']]['f1']
    report['combined_beats_best_lf'] = chosen_f1 > best_single
    return report


def read_verdicts(path):
    """Analyst verdicts (event_id -> 1 malicious / 0 benign); latest verdict per event wins."""
    if not path or not Path(path).exists():
        return pd.Series(dtype=float)
    verdicts = pd.read_csv(path, dtype=str).dropna(subset=['event_id', 'verdict']).drop_duplicates('event_id', keep='last')
    return verdicts.set_index('event_id')['verdict'].str.lower().map({'malicious': ATTACK, 'benign': BENIGN}).dropna()


def record_verdict(path, incident, verdict, analyst='', note=''):
    """Appends one row per event of an incident to the verdicts CSV."""
    if verdict not in ('malicious', 'benign'):
        raise ValueError("verdict must be 'malicious' or 'benign'")
    rows = pd.DataFrame({'event_id': incident['event_ids'], 'verdict': verdict, 'analyst': analyst,
                         'recorded_at': pd.Timestamp.now(tz='UTC').isoformat(), 'note': note})
    rows.to_csv(path, mode='a', index=False, header=not Path(path).exists(), columns=VERDICT_COLUMNS)
    return len(rows)


def weak_label(events, model, gold=None, combiner='label_model'):
    """
    Labels events with the chosen combiner ('label_model' or 'majority_vote', normally picked by
    evaluate_weak_labels on gold data). Returns (labels, label_model, summary):
      labels   eventID, one column per LF vote, p_attack, label (1/0), weight (confidence 0-1),
               and gold where known (gold overrides the label with weight 1)
    """
    scored, votes = apply_lfs(events, model)
    if combiner not in COMBINERS:
        raise ValueError(f"combiner must be one of {COMBINERS}")
    families = family_votes(votes)
    label_model = LabelModel().fit(families)
    p = (label_model.predict_proba(families) if combiner == 'label_model'
         else majority_vote_proba(families, label_model.prior_))

    labels = votes.add_prefix('lf_')
    labels.insert(0, 'eventID', scored['eventID'].astype(str).to_numpy())
    labels['p_attack'] = np.round(p, 4)
    # strictly above 0.5: a tie between attack and benign evidence is benign, given how rare attacks are
    labels['label'] = (p > 0.5).astype(int)
    labels['weight'] = np.round(np.abs(2 * p - 1), 4)
    if gold is not None:
        g = pd.Series(gold).reindex(range(len(labels))) if not isinstance(gold, pd.Series) else gold.reset_index(drop=True)
        labels['gold'] = g.to_numpy()
        known = g.notna().to_numpy()
        labels.loc[known, 'label'] = g[known].astype(int).to_numpy()
        labels.loc[known, 'weight'] = 1.0
    summary = {'lfs': lf_summary(votes, gold if gold is not None else None),
               'combiner': combiner,
               'label_model': label_model.to_dict(),
               'labelled_attack_share': round(float(labels['label'].mean()), 4)}
    return labels, label_model, summary


def gold_from(events, verdicts=None, injected=None):
    """Gold labels aligned to events: injected scenarios are attacks, then analyst verdicts override."""
    gold = pd.Series(np.nan, index=range(len(events)))
    if injected is not None:
        gold[np.asarray(injected, dtype=bool)] = ATTACK
    if verdicts is not None and len(verdicts):
        mapped = events['eventID'].astype(str).reset_index(drop=True).map(verdicts)
        gold = mapped.where(mapped.notna(), gold)
    return gold


def evaluate_weak_labels(model, holdout, history=None, verdicts=None, seed=42):
    """
    Judges the LFs and the label model on gold labels: the holdout with injected attacks
    (evaluate.inject_attacks) where injected events are attacks and real events are ASSUMED benign
    (analyst verdicts override both). The label model is only worth using if it beats every single LF.
    """
    from scripts.evaluate import inject_attacks

    data = inject_attacks(holdout, history, seed)
    injected = data.pop('scenario').notna().to_numpy()
    _, votes = apply_lfs(data, model)
    families = family_votes(votes)
    label_model = LabelModel().fit(families)
    gold = gold_from(data, verdicts, injected)
    # Real events from identities seen in training are assumed benign. Events from identities that
    # first appear after training have unknown truth (a never-seen identity may well be an intruder),
    # so they get no gold label unless an analyst gave one.
    known_actors = set(prepare_events(holdout if history is None else history)['actor'])
    known = prepare_events(data)['actor'].isin(known_actors).to_numpy()
    gold = gold.where(gold.notna() | ~pd.Series(known), BENIGN)
    report = compare_to_gold(votes, label_model.predict_proba(families), gold)
    report['lfs_summary'] = lf_summary(votes, gold)
    report['assumption'] = ('real events from identities seen in training are treated as benign; events from '
                            'identities new since training are left unlabelled unless an analyst gave a verdict')
    report['unlabelled_new_identity_events'] = int((~known & ~injected & gold.isna().to_numpy()).sum())
    return report


def save(labels, summary, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(directory / 'weak_labels.parquet', index=False)
    (directory / 'weak_label_summary.json').write_text(json.dumps(summary, indent=2, default=str))
    return directory
