"""Repeats the architecture comparison across a grid of data-generating processes.

Varies three properties of the design -- the noise scale, the presence of an
interaction between the two informative features, and whether the Level-0
learners are forced apart by giving each a different feature subset -- and
records whether the Stacking Ensemble improves on its best base learner in each
condition.
"""
import argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from generate_dataset import (CELL_MEANS, CELL_COV, POST_TYPES, POST_TYPE_P,
                              CONTENT_THEMES, POST_TIMES, CAPTION_LENGTH_RANGE,
                              NUM_HASHTAGS_RANGE, SCORE_WEIGHTS, SCORE_THRESHOLD)

CAT = ["post_type", "content_theme", "post_time"]
NUM = ["caption_length", "num_hashtags"]
SEED = 42


def make_corpus(n, seed, noise_scale=1.0, interaction=0.0):
    rng = np.random.default_rng(seed)
    post_type = rng.choice(POST_TYPES, size=n, p=POST_TYPE_P)
    theme = rng.choice(CONTENT_THEMES, size=n)
    ptime = rng.choice(POST_TIMES, size=n)
    cap = rng.integers(CAPTION_LENGTH_RANGE[0], CAPTION_LENGTH_RANGE[1] + 1, size=n)
    tags = rng.integers(NUM_HASHTAGS_RANGE[0], NUM_HASHTAGS_RANGE[1] + 1, size=n)
    V = (post_type == "Video").astype(int)
    S = (theme == "Scholarship").astype(int)

    counts = np.empty((n, 3))
    for cell in CELL_MEANS:
        mask = (V == cell[0]) & (S == cell[1])
        mean = np.array(CELL_MEANS[cell], dtype=float)
        if cell == (1, 1):
            mean = mean + interaction * np.array([60.0, 8.0, 12.0])
        cov = np.array(CELL_COV[cell], dtype=float) * (noise_scale ** 2)
        counts[mask] = rng.multivariate_normal(mean, cov, size=int(mask.sum()))
    counts = np.rint(counts).clip(min=0).astype(int)

    w = SCORE_WEIGHTS
    score = w[0] * counts[:, 0] + w[1] * counts[:, 1] + w[2] * counts[:, 2]
    return pd.DataFrame({
        "post_type": post_type, "content_theme": theme, "post_time": ptime,
        "caption_length": cap, "num_hashtags": tags,
        "target_class": (score > SCORE_THRESHOLD).astype(int)})


def prep(cat=CAT, num=NUM):
    return ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), cat),
                              ("n", StandardScaler(), num)])


def evaluate(df, decorrelate=False):
    X, y = df[CAT + NUM], df.target_class
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=SEED, stratify=y)
    if decorrelate:
        subsets = [(["post_type"], ["caption_length"]),
                   (["content_theme"], ["num_hashtags"]),
                   (["post_type", "content_theme", "post_time"], [])]
    else:
        subsets = [(CAT, NUM)] * 3
    algos = [RandomForestClassifier(n_estimators=100, random_state=SEED),
             GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                        random_state=SEED),
             MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=SEED)]
    names = ["rf", "gb", "ann"]
    base = [(nm, Pipeline([("p", prep(c, n)), ("m", a)]))
            for nm, a, (c, n) in zip(names, algos, subsets)]

    accs = {}
    for nm, mdl in base:
        mdl.fit(Xtr, ytr)
        accs[nm] = accuracy_score(yte, mdl.predict(Xte))
    stack = StackingClassifier(estimators=base, final_estimator=LogisticRegression(), cv=5)
    stack.fit(Xtr, ytr)
    acc_stack = accuracy_score(yte, stack.predict(Xte))

    # Bayes rate of the design, from the two informative indicators
    d = df.copy()
    d["V"] = (d.post_type == "Video").astype(int)
    d["S"] = (d.content_theme == "Scholarship").astype(int)
    cells = d.groupby(["V", "S"]).target_class
    bayes = sum(max(p, 1 - p) * k for p, k in zip(cells.mean(), cells.count())) / len(d)
    return max(accs.values()), acc_stack, bayes


CONDITIONS = [
    ("Baseline (as reported)",            dict(noise_scale=1.0, interaction=0.0), False),
    ("Low noise (0.5x)",                  dict(noise_scale=0.5, interaction=0.0), False),
    ("High noise (2x)",                   dict(noise_scale=2.0, interaction=0.0), False),
    ("Interaction on (V,S)",              dict(noise_scale=1.0, interaction=1.0), False),
    ("Interaction + high noise",          dict(noise_scale=2.0, interaction=1.0), False),
    ("Decorrelated base learners",        dict(noise_scale=1.0, interaction=0.0), True),
    ("Decorrelated + high noise",         dict(noise_scale=2.0, interaction=0.0), True),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    a = ap.parse_args()
    print(f"{'Condition':30s} {'Bayes%':>7s} {'Best base%':>11s} {'Stacking%':>10s} {'Delta':>7s}")
    for label, kw, dec in CONDITIONS:
        df = make_corpus(a.n, SEED, **kw)
        best, st, bayes = evaluate(df, decorrelate=dec)
        print(f"{label:30s} {bayes*100:7.2f} {best*100:11.2f} {st*100:10.2f} "
              f"{(st-best)*100:+7.2f}")
