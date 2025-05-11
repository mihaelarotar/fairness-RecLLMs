import pandas as pd


sampled_users_jobs_train = pd.read_csv('data/jobs/sampled_users_jobs_train.csv')
sampled_users_jobs_test = pd.read_csv('data/jobs/sampled_users_jobs_test.csv')


def get_titles_as_list(user_id, dataset=sampled_users_jobs_train,
    column_name="JobTitleHistory", num_items=10):
  """
  Parses the string in the dataset and returns the extracted job titles as a list
  """
  user_jobs = dataset[dataset["UserID"] == user_id]

  if user_jobs.empty:
    return []

  job_titles_str = user_jobs[column_name].values[0]
  job_titles = job_titles_str.strip("[]").replace("'", "").split(", ")

  return [job.strip() for job in job_titles[:num_items]]


def get_job_titles_str(user_id, dataset=sampled_users_jobs_train, column_name="JobTitleHistory",
    num_items=10):
  """
  Uses the 'get_titles_as_list' function to get the job title list,
  and formats them into a single string separated by semicolons.
  """
  job_titles = get_titles_as_list(user_id, dataset, column_name, num_items)

  if not job_titles:
    return "No job history found for user."

  return "; ".join(job_titles)
