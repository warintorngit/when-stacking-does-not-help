"""
Reproduces every number reported in the Results section of:
"When Stacking Does Not Help: An Evaluative Comparison of Ensemble Architectures
 for Pre-Publication Social Media Effectiveness Prediction"

Usage:
    python generate_dataset.py --out simulated_pr_data.csv
    python run_experiments.py --data simulated_pr_data.csv

Outputs Table 2, the meta-learner coefficients of Equation (6), the confusion
matrix of Figure 5, the five-fold cross-validation comparison with paired
t-tests, the feature importance scores, the ablation study, and the Bayes rate
of the design.
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
CATEGORICAL = ["post_type", "content_theme", "post_time"]
NUMERICAL = ["caption_length", "num_hashtags"]


def preprocessor(one_hot: bool = True, scale: bool = True):
    cat = (
        OneHotEncoder(handle_unknown="ignore")
        if one_hot
        else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    )
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([("cat", cat, CATEGORICAL), ("num", num, NUMERICAL)])


def base_learners():
    return [
        ("rf", RandomForestClassifier(n_estimators=100, random_state=SEED)),
        (
            "gb",
            GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, random_state=SEED
            ),
        ),
        ("ann", MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=SEED)),
    ]


def all_models(one_hot: bool = True, scale: bool = True):
    models = {
        name: Pipeline([("pre", preprocessor(one_hot, scale)), ("clf", clf)])
        for name, clf in base_learners()
    }
    models["xgb"] = Pipeline(
        [
            ("pre", preprocessor(one_hot, scale)),
            (
                "clf",
                XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=4,
                    random_state=SEED,
                    eval_metric="logloss",
                ),
            ),
        ]
    )
    models["hgb"] = Pipeline(
        [
            ("pre", preprocessor(one_hot, scale)),
            (
                "clf",
                HistGradientBoostingClassifier(
                    learning_rate=0.1,
                    max_depth=4,
                    random_state=SEED,
                ),
            ),
        ]
    )
    models["stack"] = Pipeline(
        [
            ("pre", preprocessor(one_hot, scale)),
            (
                "clf",
                StackingClassifier(
                    estimators=base_learners(),
                    final_estimator=LogisticRegression(),
                    cv=5,
                ),
            ),
        ]
    )
    return models


LABELS = {
    "rf": "Random Forest",
    "gb": "Gradient Boosting",
    "ann": "Artificial Neural Networks (ANN)",
    "xgb": "XGBoost",
    "hgb": "Histogram-based Gradient Boosting",
    "stack": "Stacking Ensemble",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="simulated_pr_data.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    X, y = df[CATEGORICAL + NUMERICAL], df["target_class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    print(f"corpus {len(df):,} records | test partition "
          f"{(y_test == 0).sum():,} low / {(y_test == 1).sum():,} high\n")

    # ---- Table 2 -----------------------------------------------------------
    print("Table 2. Performance comparison")
    print(f"{'Model':36s} {'Acc %':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s}")
    fitted = {}
    for key, model in all_models().items():
        model.fit(X_train, y_train)
        fitted[key] = model
        pred = model.predict(X_test)
        print(
            f"{LABELS[key]:36s} {accuracy_score(y_test, pred) * 100:7.2f} "
            f"{precision_score(y_test, pred):7.4f} {recall_score(y_test, pred):7.4f} "
            f"{f1_score(y_test, pred):7.4f}"
        )

    # ---- Figure 5 ----------------------------------------------------------
    tn, fp, fn, tp = confusion_matrix(y_test, fitted["stack"].predict(X_test)).ravel()
    print(f"\nFigure 5 confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")

    # ---- Equation (6) ------------------------------------------------------
    meta = fitted["stack"].named_steps["clf"].final_estimator_
    names = [LABELS[k] for k, _ in base_learners()]
    terms = " + ".join(f"{c:+.4f}*P_{n.split()[0]}" for c, n in zip(meta.coef_[0], names))
    print(f"Equation (6): Z = {terms} {meta.intercept_[0]:+.4f}")

    # ---- cross-validation and paired t-tests -------------------------------
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = {}
    for key in all_models():
        per_fold = []
        for tr, va in folds.split(X_train, y_train):
            m = all_models()[key]
            m.fit(X_train.iloc[tr], y_train.iloc[tr])
            per_fold.append(accuracy_score(y_train.iloc[va], m.predict(X_train.iloc[va])))
        scores[key] = np.array(per_fold)
        print(f"\nCV {LABELS[key]:36s} mean {per_fold and np.mean(per_fold) * 100:6.2f}% "
              f"(SD {np.std(per_fold, ddof=1) * 100:.2f})", end="")

    print("\n\nPaired t-tests, Stacking Ensemble against each base learner")
    for key in ["rf", "gb", "ann", "xgb", "hgb"]:
        diff = (scores["stack"] - scores[key]) * 100
        t_stat, p_value = stats.ttest_rel(scores["stack"], scores[key])
        print(f"  vs {LABELS[key]:36s} {diff.mean():+6.2f} pts  "
              f"t = {t_stat:6.2f}  p = {p_value:.3f}")

    # ---- feature importance ------------------------------------------------
    print("\nFeature importance, aggregated over one-hot columns")
    for key in ["rf", "gb"]:
        pre = fitted[key].named_steps["pre"]
        imp = pd.Series(
            fitted[key].named_steps["clf"].feature_importances_,
            index=pre.get_feature_names_out(),
        )
        grouped = {
            feat: imp[[c for c in imp.index if c.split("__")[1].startswith(feat)]].sum()
            for feat in CATEGORICAL + NUMERICAL
        }
        print(f"  {LABELS[key]:36s} "
              + "  ".join(f"{k}={v:.4f}" for k, v in grouped.items()))

    # ---- ablation ----------------------------------------------------------
    print("\nAblation study")
    gb_only = accuracy_score(y_test, fitted["gb"].predict(X_test))
    print(f"  full pipeline (stacking + one-hot + scaling) "
          f"{accuracy_score(y_test, fitted['stack'].predict(X_test)) * 100:6.2f}%")
    print(f"  stacking layer removed (Gradient Boosting alone) {gb_only * 100:6.2f}%")
    no_fe = all_models(one_hot=False, scale=False)["stack"].fit(X_train, y_train)
    print(f"  feature engineering removed (ordinal, unscaled)  "
          f"{accuracy_score(y_test, no_fe.predict(X_test)) * 100:6.2f}%")

    # ---- Bayes rate --------------------------------------------------------
    train = df.loc[X_train.index].copy()
    test = df.loc[X_test.index].copy()
    for frame in (train, test):
        frame["V"] = (frame.post_type == "Video").astype(int)
        frame["S"] = (frame.content_theme == "Scholarship").astype(int)
    rule = train.groupby(["V", "S"]).target_class.mean()
    bayes_pred = [int(rule[(v, s)] >= 0.5) for v, s in zip(test.V, test.S)]
    print(f"\nBayes rate of the design: "
          f"{np.mean(bayes_pred == test.target_class.values) * 100:.2f}% on the test partition")
    print("Conditional P(high | V, S):")
    for cell, p in rule.items():
        print(f"  V={cell[0]} S={cell[1]}  {p * 100:6.2f}%")


if __name__ == "__main__":
    main()
