# When Stacking Does Not Help

Data generator and experiment code for the article *"When Stacking Does Not
Help: An Evaluative Comparison of Ensemble Architectures for Pre-Publication
Social Media Effectiveness Prediction."*

Everything reported in the article is reproducible from this repository. No
empirical social media records are used at any point: the corpus is simulated
from the design constants declared in `generate_dataset.py`.

## Reproducing the results

```bash
pip install numpy pandas scikit-learn scipy xgboost matplotlib
python generate_dataset.py --out simulated_pr_data.csv --n 10000 --seed 42
python run_experiments.py --data simulated_pr_data.csv
python make_figures.py --data simulated_pr_data.csv
python run_dgp_sweep.py
```

`run_experiments.py` prints Table 2, the Equation (6) meta-learner
coefficients, the Figure 5 confusion matrix, the five-fold cross-validation
comparison with paired t-tests, the feature importance scores, the ablation
study, and the Bayes rate of the design. `make_figures.py` writes Figures 5
and 6.

## Design of the simulation

Five pre-publication features are sampled independently: post type, content
theme, post time, caption length and number of hashtags. Three engagement
counts (likes, comments, shares) are drawn from a trivariate normal whose mean
vector and covariance matrix depend on exactly two of those features — whether
the post is a Video and whether its theme is Scholarship. Post time, caption
length and number of hashtags carry no signal and are retained so that the
learners face irrelevant predictors.

The binary target is a deterministic threshold on a weighted engagement score:

    score = likes + 2*comments + 3*shares
    target_class = 1 if score > 245 else 0

Because the features fix only the mean of the score while the threshold applies
to its noisy realization, the label is stochastic given the feature vector. The
conditional class probability takes four values, one per cell, which caps the
accuracy of any classifier at roughly 91% — the Bayes rate of the design, and
the reason the stacking layer cannot improve on its base learners.

## Files

| File | Purpose |
|---|---|
| `generate_dataset.py` | Simulation; all design constants are declared at the top |
| `run_experiments.py` | Reproduces every number in the Results section |
| `make_figures.py` | Regenerates Figures 5 and 6 |
| `run_dgp_sweep.py` | Reproduces Table 6, the sweep across generating processes |
| `simulated_pr_data.csv` | The corpus used in the article (seed 42) |

## License

Released under the MIT License. Please cite the article if you use this code or data.
