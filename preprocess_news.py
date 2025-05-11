import re

import pandas as pd


news_data = pd.read_csv('data/news/sampled_users_news_final.csv')


def get_news_items_list(user_id, dataset=news_data,
    column_name="History_Titles", num_items=10):
  """
  Parses the list of (title, category, subcategory) tuples into actual tuples,
  accounting for nested or double quotes.
  Returns a list of tuples: (title, category, subcategory)
  """
  user_row = dataset[dataset["UserID"] == user_id]

  if user_row.empty:
    return []

  raw_str = user_row[column_name].values[0].strip()

  if raw_str.startswith("[") and raw_str.endswith("]"):
    raw_str = raw_str[1:-1]  # remove outer brackets

  # clean up double quotes
  raw_str = re.sub(r'""', '"', raw_str)

  # replace double quotes with single quotes
  raw_str = re.sub(r'"([^"]*?)"', r"'\1'", raw_str)

  # extract tuples: (title, category, subcategory)
  pattern = r"\(\s*'(.+?)'\s*,\s*'(.+?)'\s*,\s*'(.+?)'\s*\)"
  matches = re.findall(pattern, raw_str)

  # up to num_items items
  return matches[:num_items]


def get_browsing_history_str(user_id, dataset=news_data, column_name="History_Titles",
    num_items=10):
  """
  Uses 'get_news_items_list' to format the user's browsing history as a string
  """
  items = get_news_items_list(user_id, dataset, column_name, num_items)

  if not items:
    return "No valid history entries found."

  formatted_items = [f"{title} ({category}, {subcategory})" for
                     title, category, subcategory in items]

  return "; ".join(formatted_items)


def extract_categories_from_titles(user_id, dataset,
    column_name="Positive_Titles", num_items=5):
  """
  Extracts category and subcategory from each (title, category, subcategory) tuple
  and returns them as strings in the format "category, subcategory"
  """
  items = get_news_items_list(user_id, dataset, column_name, num_items)

  return [f"{category}, {subcategory}" for _, category, subcategory in items]


def extract_categories_from_llm_output(recommendations):
  """
  Extracts category and subcategory strings from LLM-generated recommendation strings
  Assumes each string ends with: '(category, subcategory)'
  """
  extracted = []

  for rec in recommendations:
    if "(" in rec and ")" in rec:
      match = re.search(r'\((.*?)\)', rec)
      if match:
        parts = [p.strip() for p in match.group(1).split(",")]
        if len(parts) >= 2:
          extracted.append(f"{parts[0]}, {parts[1]}")  # get category and subcategory

  return extracted