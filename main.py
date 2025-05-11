from collect_results import collect_fairness_results_for_users, \
  summarize_fairness_metrics_with_snsr_snsv
from compute_accuracy_metrics import compute_bertscore
from preprocess_news import news_data, extract_categories_from_titles, extract_categories_from_llm_output
from preprocess_jobs import sampled_users_jobs_test, get_titles_as_list
from sensitive_attributes import sensitive_attributes


# Jobs
sampled_users_jobs = sampled_users_jobs_test["UserID"].tolist()

results_df_jobs = collect_fairness_results_for_users(
    user_ids=sampled_users_jobs,
    sensitive_values_dict=sensitive_attributes,
    recommendation_type="jobs"
)

summary_df_jobs = summarize_fairness_metrics_with_snsr_snsv(results_df_jobs)

summary_df_jobs.to_csv("snsr_snsv_summary_jobs_mistral.csv", index=False)

jobs_accuracies = []

for _, row in results_df_jobs.iterrows():
    user_id = int(row["UserID"])
    recs = row["Response"]

    gt_titles = get_titles_as_list(user_id, sampled_users_jobs_test, "Title", 5)

    acc = compute_bertscore(recs, gt_titles)

    jobs_accuracies.append(acc)

results_df_jobs["Accuracy"] = jobs_accuracies

results_df_jobs.to_csv("fairness_accuracy_results_jobs_mistral.csv", index=False)


# News
sampled_users_news = news_data["UserID"].tolist()

results_df_news = collect_fairness_results_for_users(
    user_ids=sampled_users_news,
    sensitive_values_dict=sensitive_attributes
)

summary_df_news = summarize_fairness_metrics_with_snsr_snsv(results_df_news)

summary_df_news.to_csv("snsr_snsv_summary_news_mistral.csv", index=False)

news_accuracies = []

for _, row in results_df_news.iterrows():
    user_id = row["UserID"]
    recs = row["Response"]

    filtered = [r for r in recs if "(" in r and ")" in r]
    rec_results = extract_categories_from_llm_output(filtered)

    # ground truth: list of categories, subcategories from Positive_Titles
    gt_results = extract_categories_from_titles(user_id, news_data, "Positive_Titles", 5)

    acc = compute_bertscore(rec_results, gt_results)

    news_accuracies.append(acc)

results_df_news["Accuracy"] = news_accuracies

results_df_news.to_csv("fairness_accuracy_results_news_mistral.csv", index=False)
