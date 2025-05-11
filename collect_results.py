import numpy as np
import pandas as pd

from compute_fairness_metrics import compare_with_neutral, calc_mean_metric_k
from preprocess_jobs import get_job_titles_str
from preprocess_news import get_browsing_history_str
from run_model import *


def collect_similarity_scores_for_all_sensitive_values(
    user_id,
    sensitive_values_dict,
    metric="jaccard",
    recommendation_type="news",
    num_recommendations=5
):
  results = []

  history_str = get_browsing_history_str(user_id) if (
      recommendation_type == "news") else get_job_titles_str(user_id)

  extra_instructions = " and their predicted category and subcategory in parentheses" \
    if (recommendation_type == "news") else ""

  # get neutral prompt & results once
  neutral_prompt = get_prompt(user_id, history_str, recommendation_type,
                              num_recommendations,
                              sensitive_attribute="this user",
                              extra_instructions=extra_instructions)
  neutral_response = generate_recommendations(neutral_prompt)
  neutral_list = response_to_list(neutral_response)

  # for debugging in case of failure
  print("\n[Neutral Prompt Response]")
  print(neutral_response)

  results.append({
    "UserID": user_id,
    "RecommendationType": recommendation_type,
    "Metric": metric,
    "SensitiveAttribute": "neutral",
    "SensitiveValue": "this user",
    "SimilarityScore": 1.0,
    "Response": neutral_list
  })

  # loop over sensitive attributes
  for sensitive_key, value_list in sensitive_values_dict.items():
    for sensitive_value in value_list:
      # 1. Generate sensitive prompt
      sensitive_prompt = get_prompt(user_id, history_str, recommendation_type,
                                    num_recommendations,
                                    sensitive_attribute=sensitive_value,
                                    extra_instructions=extra_instructions)

      # 2. Generate LLM output
      sensitive_response = generate_recommendations(sensitive_prompt)
      sensitive_list = response_to_list(sensitive_response)

      print(f"\nPrompt for '{sensitive_value}':")
      print("\nResponse:")
      print(sensitive_response)

      # 3. Compute similarity to neutral results
      similarity_at_k = compare_with_neutral(sensitive_list, neutral_list,
                                             top_k=num_recommendations,
                                             metric=metric)
      similarity_score = calc_mean_metric_k(similarity_at_k, top_k=num_recommendations)[-1]  # use score at top-K

      # 4. Store
      results.append({
        "UserID": user_id,
        "RecommendationType": recommendation_type,
        "Metric": metric,
        "SensitiveAttribute": sensitive_key,
        "SensitiveValue": sensitive_value,
        "SimilarityScore": similarity_score,
        "Response": sensitive_list
      })

  return results


def collect_fairness_results_for_users(
    user_ids,
    sensitive_values_dict,
    recommendation_type="news",
    metrics=None,
    num_recommendations=5
):
  """
  Collect similarity scores for multiple users across metrics, sensitive attributes, and recommendation types
  """
  if metrics is None:
    metrics = ["jaccard", "serp_ms", "prag", "bertscore"]
  all_results = []

  # for debugging purposes
  for idx, user_id in enumerate(user_ids, start=1):
    if idx % 20 == 0:
      print(f"Processing user {idx}/{len(user_ids)}: {user_id}")
    for metric in metrics:
      user_results = collect_similarity_scores_for_all_sensitive_values(
          user_id=user_id,
          sensitive_values_dict=sensitive_values_dict,
          metric=metric,
          recommendation_type=recommendation_type,
          num_recommendations=num_recommendations
      )

      all_results.extend(user_results)

  return pd.DataFrame(all_results)


def summarize_fairness_metrics_with_snsr_snsv(fairness_df):
  """
  Compute SNSR and SNSV for each user, sensitive attribute, metric, and recommendation type
  """

  summary_rows = []

  fairness_filtered = fairness_df[fairness_df["SimilarityScore"] != "#NULL"].copy()

  grouped = fairness_filtered.groupby(
      ["UserID", "RecommendationType", "Metric", "SensitiveAttribute"])

  for (user_id, rec_type, metric, attr), group in grouped:
    scores = group["SimilarityScore"].values
    snsr = np.max(scores) - np.min(scores)
    snsv = np.std(scores)
    summary_rows.append({
      "UserID": user_id,
      "RecommendationType": rec_type,
      "Metric": metric,
      "SensitiveAttribute": attr,
      "SNSR": snsr,
      "SNSV": snsv,
      "Max": np.max(scores),
      "Min": np.min(scores),
      "Mean": np.mean(scores)
    })

  return pd.DataFrame(summary_rows)