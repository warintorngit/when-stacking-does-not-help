"""
Synthetic corpus generator for:
"When Stacking Does Not Help: An Evaluative Comparison of Ensemble Architectures
 for Pre-Publication Social Media Effectiveness Prediction"

The corpus simulates public-relations posts on an institutional Facebook page.
Every parameter below is a design constant of the simulation: no empirical
social media records are used at any point.

Design
------
Five pre-publication features are sampled independently:

    post_type       Image 0.50, Video 0.20, Carousel 0.20, Link 0.10
    content_theme   uniform over 5 institutional topics
    post_time       uniform over 4 dayparts
    caption_length  uniform integer on [10, 149]
    num_hashtags    uniform integer on [0, 9]

Three post-publication engagement counts (likes, comments, shares) are then
drawn from a trivariate normal whose mean vector and covariance matrix depend
on exactly two of the five features -- whether the post is a Video (V) and
whether its theme is Scholarship (S). The four (V, S) cells and their
parameters are listed in CELL_MEANS and CELL_COV below. Draws are rounded to
integers and clipped at zero.

post_time, caption_length and num_hashtags enter no mean and therefore carry no
signal. They are retained so that the learners face irrelevant predictors.

The binary target is a deterministic threshold on a weighted engagement score:

    score = likes + 2*comments + 3*shares
    target_class = 1 if score > 245 else 0

Because the features fix only the mean of the score while the threshold applies
to its noisy realization, the label is stochastic given the feature vector. The
conditional class probability takes four values (one per V,S cell), which caps
the accuracy of any classifier at roughly 91% -- the Bayes rate of the design.
"""

import argparse

import numpy as np
import pandas as pd

SEED = 42
N_RECORDS = 10_000

POST_TYPES = ["Image", "Video", "Carousel", "Link"]
POST_TYPE_P = [0.50, 0.20, 0.20, 0.10]

CONTENT_THEMES = [
    "Admission_Alert",
    "Alumni_Success",
    "Faculty_Intro",
    "Scholarship",
    "Student_Life",
]

POST_TIMES = ["Morning", "Afternoon", "Evening", "Night"]

CAPTION_LENGTH_RANGE = (10, 149)
NUM_HASHTAGS_RANGE = (0, 9)

# Engagement is governed by exactly two of the five features: whether the post
# is a Video (V) and whether its theme is Scholarship (S). Each of the four
# (V, S) cells has its own mean vector and covariance matrix for
# (likes, comments, shares). Within a cell the three counts are positively
# correlated: a post that does unusually well on likes tends to do unusually
# well on the other two as well.
CELL_MEANS = {
    (0, 0): [98.84, 11.95, 7.39],
    (0, 1): [180.78, 22.18, 14.15],
    (1, 0): [150.84, 18.48, 41.59],
    (1, 1): [230.35, 28.81, 47.20],
}

CELL_COV = {
    (0, 0): [[917.19, 118.08, 73.00], [118.08, 35.32, 9.12], [73.00, 9.12, 23.69]],
    (0, 1): [[849.21, 108.57, 73.22], [108.57, 77.82, 8.05], [73.22, 8.05, 61.96]],
    (1, 0): [[877.42, 110.06, 74.31], [110.06, 59.29, 5.13], [74.31, 5.13, 172.75]],
    (1, 1): [[971.78, 134.41, 81.64], [134.41, 112.53, 20.10], [81.64, 20.10, 240.34]],
}

SCORE_WEIGHTS = (1, 2, 3)  # likes, comments, shares
SCORE_THRESHOLD = 245


def generate(n_records: int = N_RECORDS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    post_type = rng.choice(POST_TYPES, size=n_records, p=POST_TYPE_P)
    content_theme = rng.choice(CONTENT_THEMES, size=n_records)
    post_time = rng.choice(POST_TIMES, size=n_records)
    caption_length = rng.integers(
        CAPTION_LENGTH_RANGE[0], CAPTION_LENGTH_RANGE[1] + 1, size=n_records
    )
    num_hashtags = rng.integers(
        NUM_HASHTAGS_RANGE[0], NUM_HASHTAGS_RANGE[1] + 1, size=n_records
    )

    is_video = (post_type == "Video").astype(int)
    is_scholarship = (content_theme == "Scholarship").astype(int)

    counts = np.empty((n_records, 3))
    for cell in CELL_MEANS:
        mask = (is_video == cell[0]) & (is_scholarship == cell[1])
        counts[mask] = rng.multivariate_normal(
            CELL_MEANS[cell], CELL_COV[cell], size=int(mask.sum())
        )
    counts = np.rint(counts).clip(min=0).astype(int)
    likes, comments, shares = counts[:, 0], counts[:, 1], counts[:, 2]

    w_likes, w_comments, w_shares = SCORE_WEIGHTS
    score = w_likes * likes + w_comments * comments + w_shares * shares
    target_class = (score > SCORE_THRESHOLD).astype(int)

    return pd.DataFrame(
        {
            "post_id": [f"POST_{i + 1:04d}" for i in range(n_records)],
            "post_type": post_type,
            "content_theme": content_theme,
            "post_time": post_time,
            "caption_length": caption_length,
            "num_hashtags": num_hashtags,
            "likes": likes,
            "comments": comments,
            "shares": shares,
            "target_class": target_class,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="simulated_pr_data.csv")
    parser.add_argument("--n", type=int, default=N_RECORDS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    df = generate(args.n, args.seed)
    df.to_csv(args.out, index=False)

    share_high = df.target_class.mean()
    print(f"wrote {len(df):,} records to {args.out}")
    print(f"class balance: {1 - share_high:.4f} low / {share_high:.4f} high")


if __name__ == "__main__":
    main()
