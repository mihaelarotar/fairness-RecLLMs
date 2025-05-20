import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast


results_df = pd.read_csv("fairness_accuracy_results_jobs_gemma.csv")
snsr_snsv_df = pd.read_csv("snsr_snsv_summary_jobs_gemma.csv")


def replace_null_similarity(df, column="SimilarityScore"):
  """
  Ignore entries for which we couldn't compute the score
  """
  num_nulls = (df[column] == "#NULL").sum()

  df[column] = df[column].replace("#NULL", 0.0).astype(float)

  print(f"Replaced {num_nulls} '#NULL' entries in '{column}' with 0.0")
  return df


def get_snsr_snsv_table(df):
  df = df[df["SensitiveAttribute"].isin(["gender", "age"])]

  mean_sns = df.groupby(["SensitiveAttribute", "Metric"])[
    ["SNSR", "SNSV"]].mean().round(4).reset_index()

  return mean_sns


def get_fairness_table(df):
  df = df[df["SensitiveAttribute"].isin(["gender", "age"])]

  mean_results = df.groupby(["SensitiveAttribute", "SensitiveValue", "Metric"])[
    ["SimilarityScore"]].mean().round(4).reset_index()

  return mean_results


def plot_average_similarity_scores(df, attr="gender", model="Mistral (7B)"):
  new_results_df = df[df["SensitiveAttribute"].isin(["gender", "age"])]

  metric_labels = {
      "jaccard": "Jaccard",
      "serp_ms": "SERP*",
      "prag": "PRAG*",
      "bertscore": "BERTScore",
    }

  mean_results = new_results_df.groupby(["SensitiveAttribute", "SensitiveValue", "Metric"])[
    ["SimilarityScore"]].mean().round(4).reset_index()

  mean_results["Metric"] = mean_results["Metric"].replace(metric_labels)

  sns.set(style="whitegrid")

  subset = mean_results[mean_results["SensitiveAttribute"] == attr]

  plt.figure(figsize=(10, 6))
  sns.barplot(
      data=subset,
      x="Metric",
      y="SimilarityScore",
      hue="SensitiveValue"
  )

  plt.title(f"Average Similarity Scores by {attr.capitalize()} - {model}", fontsize=16)
  plt.ylabel("Similarity Score", fontsize=14)
  plt.xlabel("Metric", fontsize=14)
  plt.xticks(fontsize=12)
  plt.yticks(fontsize=12)
  plt.ylim(0, 1)
  plt.legend(title="SensitiveValue", fontsize=12, title_fontsize=13)
  plt.tight_layout()
  plt.show()


def get_accuracy_df(df):
  # we can just choose one of the fairness metrics since the results for accuracy will repeat themselves
  jaccard_df = df[results_df["Metric"] == "jaccard"].copy()

  # convert accuracy from string to dictionary
  jaccard_df["Accuracy"] = jaccard_df["Accuracy"].apply(ast.literal_eval)

  # create separate columns for precision, recall, F1 score
  jaccard_df["Precision"] = jaccard_df["Accuracy"].apply(
    lambda x: x.get("Precision"))
  jaccard_df["Recall"] = jaccard_df["Accuracy"].apply(lambda x: x.get("Recall"))
  jaccard_df["F1-score"] = jaccard_df["Accuracy"].apply(
    lambda x: x.get("F1-score"))

  jaccard_df[["Precision", "Recall", "F1-score"]] = jaccard_df[
    ["Precision", "Recall", "F1-score"]].round(4)

  return jaccard_df


def get_accuracy_table(accuracy_df):
  agg_by_attribute = accuracy_df.groupby("SensitiveAttribute")[
    ["Precision", "Recall", "F1-score"]].mean().round(4)

  agg_by_value = accuracy_df.groupby(["SensitiveAttribute", "SensitiveValue"])[
    ["Precision", "Recall", "F1-score"]].mean().round(4)

  return agg_by_attribute, agg_by_value


def plot_accuracies(df):
  accuracy_df = get_accuracy_df(df)

  long_df = pd.melt(
      accuracy_df,
      id_vars=["SensitiveAttribute", "SensitiveValue"],
      value_vars=["Precision", "Recall", "F1-score"],
      var_name="Metric",
      value_name="Score"
  )

  long_df["Group"] = long_df["SensitiveAttribute"] + " - " + long_df[
    "SensitiveValue"]
  long_df.loc[long_df["SensitiveAttribute"] == "neutral", "Group"] = "neutral"

  group_order = (
      ["neutral"] +
      [f"gender - {val}" for val in ["him", "her", "them"]] +
      [f"age - {val}" for val in [
        "a high school student", "a college student", "a working professional",
        "a parent of young children", "a senior citizen", "a retired individual"
      ]]
  )

  plt.figure(figsize=(14, 6))
  sns.barplot(
      data=long_df,
      x="Group",
      y="Score",
      hue="Metric",
      order=group_order,
      palette="Set2",
      errorbar=None
  )
  plt.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8)
  plt.axvline(x=3.5, color="gray", linestyle="--",
              linewidth=0.8)  # spacing between gender and age
  plt.title("Accuracy Metrics by Sensitive Group")
  plt.xticks(rotation=45, ha="right")
  plt.ylabel("Score")
  plt.xlabel("Sensitive Group")
  plt.tight_layout()
  plt.legend(title="Metric", bbox_to_anchor=(1.05, 1))
  plt.show()