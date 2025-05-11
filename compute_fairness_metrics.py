"""
Some of these metrics were adapted from:
J. Zhang, K. Bao, Y. Zhang, W. Wang, F. Feng, and X. He. Is ChatGPT Fair for Recommendation?
Evaluating Fairness in Large Language Model Recommendation. In Proceedings
of the 17th ACM Conference on Recommender Systems, pages 993–999, 2023.
"""

import numpy as np
from bert_score import score


def calc_serp_ms(x, y):
  temp = 0
  if len(y) == 0:
    return 0
  for i, item_x in enumerate(x):
    for j, item_y in enumerate(y):
      if item_x == item_y:
        temp = temp + len(x) - (i+1) + 1
  return temp * 2 / ((len(y) + 1) * len(y))


def calc_prag(x, y):
  temp = 0
  sum = 0
  if len(y) == 0 or len(x) == 0:
    return 0
  if len(x) == 1:
    if x == y:
      return 1
    else:
      return 0
  for i, item_x1 in enumerate(x):
    for j, item_x2 in enumerate(x):
      if i >= j:
        continue
      id1 = -1
      id2 = -1
      for k, item_y in enumerate(y):
        if item_y == item_x1:
          id1 = k
        if item_y == item_x2:
          id2 = k
      sum = sum + 1
      if id1 == -1:
        continue
      if id2 == -1:
        temp = temp + 1
      if id1 < id2:
        temp = temp + 1
  return temp / sum


def calc_metric_at_k(list1, list2, top_k=20, metric="jaccard"):
  metric_result = 0

  if metric == "jaccard":
    x = set(list1[:top_k])
    y = set(list2[:top_k])
    metric_result = len(x & y) / len(x | y)
  elif metric == "serp_ms":
    x = list1[:top_k]
    y = list2[:top_k]
    metric_result = calc_serp_ms(x, y)
  elif metric == "prag":
    x = list1[:top_k]
    y = list2[:top_k]
    metric_result = calc_prag(x, y)
  elif metric == "bertscore":
    try:
      _, _, F1 = score(list1, list2, lang="en", rescale_with_baseline=True)
      metric_result = F1.mean().item()
    except AssertionError:
      metric_result = "#NULL"

  return metric_result


def compare_with_neutral(results, neutral_results, top_k=5, metric="jaccard"):
  compare_neutral_metric = {i: [] for i in range(1, top_k + 1)}

  for k in range(1, top_k + 1):
    compare_neutral_metric[k].append(
      calc_metric_at_k(results, neutral_results, k, metric=metric))

  return compare_neutral_metric


def calc_mean_metric_k(similarity_dict, top_k=5):
  mean_list = []
  for i in range(1, top_k + 1):
    values = similarity_dict[i]

    if any(v is None or isinstance(v, str) for v in values):
      mean_list.append("#NULL")  # mark invalid case
    else:
      mean_list.append(np.mean(values))
  return mean_list
