# DCN-v2 Recommendation: Neural Re-Ranking with Deep & Cross Network v2

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Baseline-blue)](https://xgboost.readthedocs.io)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-4285F4)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-Educational-green)](LICENSE)

> This project extends the [two-tower-recsys](https://github.com/nbatra/two-tower-recsys) baseline by replacing the XGBoost LambdaMART re-ranker with DCN-v2 (Deep & Cross Network v2) and adding rigorous statistical evaluation. Retrieval models (Two-Tower, ComiRec, SASRec) and FAISS indexing are unchanged.

## What This Project Does

This project answers a practical question: **Can a neural ranking model (DCN-v2) outperform XGBoost LambdaMART for re-ranking recommendation candidates?**

DCN-v2 (Google, 2021) learns explicit polynomial feature interactions via its Cross Network alongside implicit patterns via a Deep Network. In theory, it should capture complex multi-feature interactions more efficiently than XGBoost's axis-aligned tree splits. We test this hypothesis on MovieLens-25M across three retrieval pipelines with proper statistical rigor.

## Architecture

```
Stage 1: Retrieval (UNCHANGED from baseline)
    Three models: Two-Tower / ComiRec / SASRec
    FAISS inner-product search --> 200 candidates per user

Stage 2: Re-Ranking (THIS PROJECT'S FOCUS)
    Baseline: XGBoost LambdaMART (tree ensemble, rank:ndcg objective)
    New:      DCN-v2 (Cross Network + Deep Network, BCE + BPR loss)
    Features: retrieval_score + user(24) + item(73) + cross(7) = 105/109 dim

Stage 3: Post-processing (UNCHANGED)
    MMR diversity re-ranking, already-seen filtering
```

### DCN-v2 Model

```
Input: x_0 (105 or 109 features)
         |
         +--> Cross Network (3 layers, full-rank W matrices)
         |      x_{l+1} = x_0 * (W_l @ x_l + b_l) + x_l
         |      Learns explicit up-to-4-way feature interactions
         |      Output: 105-dim
         |
         +--> Deep Network (MLP: 256 -> 128, BatchNorm, ReLU, Dropout)
         |      Learns implicit high-order patterns
         |      Output: 128-dim
         |
         +--> Concat = 233-dim --> Linear(233, 1) --> score
```

Parameters: ~94K (Two-Tower/SASRec) or ~99K (ComiRec). Inference: <1ms for 200 candidates on CPU.

## Results

### DCN-v2 vs XGBoost Ranking Quality

| Pipeline | NDCG@10 (DCN-v2) | NDCG@10 (XGBoost) | Relative Delta | Statistically Significant? |
|----------|------------------:|-------------------:|---------------:|:--------------------------:|
| Two-Tower (105-dim) | 0.0502 | 0.0495 | +1.3% | No |
| ComiRec (109-dim) | 0.0562 | 0.0593 | -5.3% | No |
| SASRec (105-dim) | 0.0570 | 0.0574 | -0.7% | No |

**No metric was statistically significant after Benjamini-Hochberg FDR correction across any pipeline.** All Cohen's d effect sizes were < 0.1 (negligible). The two models perform equivalently on this dataset.

### Why This Result is Informative

1. **Feature set limitations**: With 105 features (mostly binary genre indicators and normalized scalars), there may not be enough complex interaction signal for DCN-v2 to exploit beyond what XGBoost's 175 trees already capture.
2. **Dataset scale**: MovieLens-25M has ~3M training interactions. DCN-v2's advantages emerge at web-scale (100M+ interactions, as in Google's original paper).
3. **XGBoost is a strong baseline**: LambdaMART with rank:ndcg objective directly optimizes the evaluation metric. DCN-v2's BCE+BPR loss is a surrogate.
4. **Practical conclusion**: Do not deploy DCN-v2 as a replacement for XGBoost based on ranking quality alone. Consider deployment if you need calibrated probability outputs, end-to-end gradient flow, or incremental fine-tuning.

## Statistical Evaluation Framework

This project uses production-grade statistical methodology:

| Component | Method | Purpose |
|-----------|--------|---------|
| Paired comparison | Both rankers score same FAISS candidates per user | Eliminates retrieval variance |
| Significance test | Wilcoxon signed-rank (non-parametric, paired) | Tests if DCN-v2 consistently ranks differently |
| Multiple testing | Benjamini-Hochberg FDR correction | Controls false discovery rate across 6+ simultaneous tests |
| Confidence intervals | Bootstrap (10,000 resamples) | Distribution-free uncertainty quantification |
| Effect size | Cohen's d | Distinguishes statistical from practical significance |
| A/B testing | Group Sequential Design, O'Brien-Fleming boundaries, CUPED | Production deployment framework |

See Notebook 00, Section 7 for a detailed layman explanation of why FDR correction matters, using this project's actual results as the worked example.

## Project Structure

```
notebooks/
    00_architecture_overview.ipynb    # System design, DCN-v2 theory, FDR explanation
    01_data_loading_and_exploration.ipynb  # EDA (unchanged from baseline)
    02_feature_engineering.ipynb      # Feature pipeline (unchanged)
    03_two_tower_model.ipynb          # Two-Tower retrieval (unchanged)
    04_two_tower_dcn_v2_ranker.ipynb  # DCN-v2 training on Two-Tower features
    05_two_tower_evaluation.ipynb     # Paired evaluation with full statistics
    06_comirec_model.ipynb            # ComiRec retrieval (unchanged)
    07_comirec_dcn_v2_ranker.ipynb    # DCN-v2 training on ComiRec features (109-dim)
    08_comirec_evaluation.ipynb       # ComiRec pipeline evaluation
    09_sasrec_model.ipynb             # SASRec retrieval (unchanged)
    10_sasrec_dcn_v2_ranker.ipynb     # DCN-v2 training on SASRec features
    11_sasrec_evaluation.ipynb        # SASRec pipeline evaluation
    12_production_inference_simulation.ipynb  # Production serving with model routing
    13_ab_testing_framework.ipynb     # Sequential testing, CUPED, guardrails

models/
    dcn_v2_ranker.pt                  # Trained DCN-v2 (Two-Tower pipeline)
    dcn_v2_config.json                # Architecture hyperparameters
    xgboost_ranker.json               # Baseline XGBoost (for comparison)
    comirec/dcn_v2_ranker.pt          # DCN-v2 for ComiRec pipeline
    sasrec/dcn_v2_ranker.pt           # DCN-v2 for SASRec pipeline

data/
    ml-25m/                           # Raw MovieLens 25M dataset
    processed/                        # Engineered features, train/val/test splits
```

## Key Differences from Baseline Project

| Aspect | [Baseline](https://github.com/nbatra/two-tower-recsys) | This Project |
|--------|---------|--------------|
| Ranking model | XGBoost LambdaMART | DCN-v2 (+ XGBoost for comparison) |
| Loss function | rank:ndcg (LambdaMART) | BCE + BPR (combined) |
| Statistical evaluation | Mean comparison | FDR-corrected paired tests, bootstrap CIs, Cohen's d |
| A/B testing | Basic Welch's t-test | Group Sequential Design, CUPED, guardrails |
| Score interpretation | Relative (unbounded) | Calibrated probabilities (sigmoid) |
| Model artifact | .json (tree ensemble) | .pt (neural network state dict) |

## How to Run

```bash
# Clone and setup
git clone <this-repo>
cd DCN-v2_Recommendation

# Create environment and install dependencies
uv venv --python 3.13 .venv
uv pip install torch numpy pandas scikit-learn xgboost faiss-cpu \
    matplotlib jupyter statsmodels pyarrow

# Run notebooks in order (NB00 is reference-only)
.venv/bin/jupyter lab notebooks/
```

Notebooks 01-03, 06, 09 are unchanged from the baseline and require the MovieLens-25M dataset (downloaded automatically by `setup.sh` or manually from [grouplens.org](https://grouplens.org/datasets/movielens/25m/)).

Notebooks 04, 07, 10 train DCN-v2 rankers (~2-5 minutes each on CPU).
Notebooks 05, 08, 11 run paired evaluation (~5-10 minutes each).
Notebooks 12, 13 run production simulations (~2 minutes each).

## Key Takeaways

1. **XGBoost LambdaMART is a remarkably strong baseline for recommendation ranking.** On MovieLens-25M with 105 features, DCN-v2 cannot statistically significantly outperform it. This aligns with industry experience: gradient-boosted trees dominate when features are well-engineered and datasets are moderate-scale.

2. **Statistical rigor prevents false conclusions.** Without FDR correction, we would have reported "significant" results that are actually noise from testing multiple metrics. The correction saves us from deploying a model change that provides no real benefit.

3. **DCN-v2's theoretical advantages require scale to materialize.** The Cross Network's ability to learn arbitrary feature interactions is powerful, but with 105 features and 3M training examples, XGBoost's greedy split search finds the important interactions just as well. At 1000+ features and 100M+ interactions (web-scale), DCN-v2 would likely separate.

4. **The evaluation framework is reusable.** The paired testing, FDR correction, bootstrap CIs, and A/B testing notebooks apply to any model comparison -- not just DCN-v2 vs XGBoost.

---

## Author

Built by **Nipun Batra**

[![GitHub](https://img.shields.io/badge/GitHub-nbatra-181717?logo=github)](https://github.com/nbatra)

---

## References

- Wang et al. (2021). *DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems.* Google Research.
- Covington et al. (2016). *Deep Neural Networks for YouTube Recommendations.* Google.
- Cen et al. (2020). *Controllable Multi-Interest Framework for Recommendation.* KDD.
- Kang & McAuley (2018). *Self-Attentive Sequential Recommendation.* ICDM.
- Benjamini & Hochberg (1995). *Controlling the False Discovery Rate.* JRSS-B.

## License

This project is released for educational and portfolio purposes. The MovieLens 25M dataset is provided by [GroupLens Research](https://grouplens.org/) under their own terms of use.

<details>
<summary>Keywords</summary>

DCN-v2, Deep & Cross Network, Cross Network, Polynomial Feature Interactions, Higher-Order Features, Neural Ranking, Recommendation System, Two-Tower, ComiRec, SASRec, FAISS, Approximate Nearest Neighbor, XGBoost LambdaMART, Learning to Rank, Benjamini-Hochberg, FDR Correction, Bootstrap Confidence Intervals, Cohen's d, A/B Testing, Sequential Testing, MovieLens 25M, Production ML, RecSys, Re-Ranking, Deep Network, Stacked Architecture, PyTorch, NDCG, Google Research

</details>
