import numpy as np
import pandas as pd
import pytest
from scripts.evaluate import time_split
from scripts.train_models import build_pipeline
from scripts.weak_labels import (ABSTAIN, ATTACK, BENIGN, LabelModel, apply_lfs, evaluate_weak_labels, gold_from,
                                 read_verdicts, record_verdict, weak_label)
from tests.synthetic import make_account_activity


@pytest.fixture(scope='module')
def activity():
    return make_account_activity(days=21)


def simulated_votes(n=20000, attack_rate=0.02, seed=0):
    """Three LFs with known quality: a precise one, a noisy one, and a benign-only one."""
    rng = np.random.default_rng(seed)
    y = rng.random(n) < attack_rate
    precise = np.where(y & (rng.random(n) < 0.8), ATTACK, ABSTAIN)
    precise[~y & (rng.random(n) < 0.002)] = ATTACK
    noisy = np.where(rng.random(n) < np.where(y, 0.6, 0.05), ATTACK, ABSTAIN)
    benign = np.where(~y & (rng.random(n) < 0.7), BENIGN, ABSTAIN)
    return y, pd.DataFrame({'precise': precise, 'noisy': noisy, 'benign_only': benign})


def test_label_model_learns_which_sources_to_trust():
    y, votes = simulated_votes()

    model = LabelModel().fit(votes)
    p = model.predict_proba(votes)

    # the precise source's attack votes are trusted far more than the noisy one's
    assert model.confusion_['precise'][ATTACK][0] / model.confusion_['precise'][BENIGN][0] > \
           model.confusion_['noisy'][ATTACK][0] / model.confusion_['noisy'][BENIGN][0]
    assert model.prior_ == pytest.approx(0.02, abs=0.01)
    predicted = p >= 0.5
    f1 = 2 * (predicted & y).sum() / (predicted.sum() + y.sum())
    noisy_f1 = 2 * ((votes['noisy'] == ATTACK) & y).sum() / ((votes['noisy'] == ATTACK).sum() + y.sum())
    assert f1 > noisy_f1


def test_lfs_vote_on_real_pipeline_output(activity):
    model = build_pipeline().fit(activity)
    events = activity.iloc[:200].copy()
    events.loc[events.index[0], ['eventName', 'eventSource']] = ['StopLogging', 'cloudtrail.amazonaws.com']

    scored, votes = apply_lfs(events, model)

    assert len(votes) == len(events) and set(votes.columns) >= {'isolation_forest', 'critical_rules', 'routine_work'}
    assert votes['critical_rules'].iat[0] == ATTACK
    assert (votes['aws_service_activity'] == BENIGN).any()  # AWS service events vote benign
    assert set(np.unique(votes.to_numpy())) <= {ATTACK, BENIGN, ABSTAIN}


def test_weak_label_output_and_gold_override(activity):
    model = build_pipeline().fit(activity)
    events = activity.iloc[:300].reset_index(drop=True)
    verdicts = pd.Series({events['eventID'].iat[5]: ATTACK})

    labels, _, summary = weak_label(events, model, gold_from(events, verdicts))

    assert {'eventID', 'p_attack', 'label', 'weight', 'gold'} <= set(labels.columns)
    assert labels['p_attack'].between(0, 1).all() and labels['weight'].between(0, 1).all()
    assert labels.loc[5, 'label'] == 1 and labels.loc[5, 'weight'] == 1.0  # analyst verdict wins
    assert summary['labelled_attack_share'] < 0.1


def test_label_model_on_injected_attacks(activity):
    train, holdout = time_split(activity)

    report = evaluate_weak_labels(build_pipeline().fit(train), holdout, history=train)

    assert report['gold_attacks'] > 50
    assert report['label_model']['recall'] >= 0.8
    assert report['label_model']['precision'] >= 0.5
    assert 'isolation_forest' in report['lfs'] and 'coverage' in report['lfs_summary']['routine_work']


def test_verdicts_round_trip(tmp_path):
    path = tmp_path / 'verdicts.csv'
    incident = {'incident_id': 'x', 'event_ids': ['e1', 'e2']}

    record_verdict(path, incident, 'benign', analyst='sam')
    record_verdict(path, {'incident_id': 'y', 'event_ids': ['e2']}, 'malicious')  # later verdict wins

    assert read_verdicts(path).to_dict() == {'e1': BENIGN, 'e2': ATTACK}
    with pytest.raises(ValueError):
        record_verdict(path, incident, 'maybe')


def test_majority_vote_combiner_and_choice(activity):
    model = build_pipeline().fit(activity)
    events = activity.iloc[:200].reset_index(drop=True)

    labels, _, summary = weak_label(events, model, combiner='majority_vote')

    assert summary['combiner'] == 'majority_vote'
    assert labels['p_attack'].between(0, 1).all()
    with pytest.raises(ValueError):
        weak_label(events, model, combiner='magic')
    train, holdout = time_split(activity)
    report = evaluate_weak_labels(build_pipeline().fit(train), holdout, history=train)
    assert report['chosen_combiner'] in ('label_model', 'majority_vote')


def test_tied_votes_label_benign(activity):
    from scripts.weak_labels import majority_vote_proba
    votes = pd.DataFrame({'a': [ATTACK, ATTACK, ABSTAIN], 'b': [BENIGN, ABSTAIN, ABSTAIN]})

    p = majority_vote_proba(votes, prior=0.01)

    assert list(p) == [0.5, 1.0, 0.01]
    assert list((p > 0.5).astype(int)) == [0, 1, 0]  # the rule weak_label uses


def test_correlated_sources_get_one_family_vote():
    from scripts.weak_labels import family_votes, majority_vote_proba
    # both models flag it, AWS-service context says benign: one family each -> tie -> benign
    votes = pd.DataFrame({'isolation_forest': [ATTACK], 'behaviour_baseline': [ATTACK], 'calm_behaviour': [ABSTAIN],
                          'graph': [ABSTAIN], 'critical_rules': [ABSTAIN], 'iam_persistence': [ABSTAIN],
                          'burst_rules': [ABSTAIN], 'routine_work': [ABSTAIN], 'aws_service_activity': [BENIGN]})

    families = family_votes(votes)

    assert families.loc[0, 'models'] == ATTACK and families.loc[0, 'aws_service'] == BENIGN
    assert majority_vote_proba(families)[0] == 0.5  # labelled benign (strictly > 0.5 needed)
