# Fairness in LLM-based Recommender Systems

## Overview

This project evaluates the fairness of LLM-based recommender systems across sensitive attributes (gender, age). We prompt LLMs using various bias-aware prompting strategies to generate recommendations for users when sensitive values are implied in the prompt, and measure how recommendations change compared to a neutral baseline. We evaluate on two domains, job recommendations and news recommendations, using multiple fairness and accuracy metrics.

## Repository Structure

```
.
├── main.py                      # Main entry point: runs the full pipeline for both domains
├── run_model.py                 # LLM loading, prompt templates and response parsing
├── collect_results.py           # Orchestrates collecting the recommendations results and fairness scoring
├── compute_fairness_metrics.py  # Code for the various fairness metrics used
├── compute_accuracy_metrics.py  # Code for effectiveness evaluation
├── sensitive_attributes.py      # Defines the sensitive attributes and values
├── preprocess_jobs.py           # Preprocessing and loading of the sampled job recommendation dataset
├── preprocess_news.py           # Preprocessing and loading of the sampled news recommendation dataset
├── visualize_results.py         # Visualization and plotting of fairness/accuracy results
└── data/
    ├── jobs/
    │   ├── sampled_users_jobs_train.csv
    │   └── sampled_users_jobs_test.csv
    └── news/
        └── sampled_users_news_final.csv
```

## Usage

1. Set the HuggingFace API key in `run_model.py`.
2. Select the desired model by changing `MODEL_NAME` in `run_model.py`.
3. Run the full pipeline:

```bash
python main.py
```

This will generate CSV files with per-user fairness scores and accuracy metrics for both the jobs and news domains.