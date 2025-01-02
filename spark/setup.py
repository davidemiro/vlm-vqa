from pyspark.sql import SparkSession

def install_requirements_and_clone_repo(spark):
  """
  Installs libraries from requirements.txt and clones a GitHub repository
  within a Dataproc cluster.

  Args:
    spark: SparkSession object.

  Raises:
    Exception: If any of the commands fail.
  """

  # Clone the GitHub repository
  try:
    !git clone https://davidemiro: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@github.com/davidemiro/vlm-gemma-2-2b.git
    %cd vlm-gemma-2-2b
  except Exception as e:
    raise Exception(f"Error cloning repository: {e}")

  # Install libraries from requirements.txt
  try:
    !pip install -r requirements.txt
  except Exception as e:
    raise Exception(f"Error installing libraries: {e}")

if __name__ == "__main__":
  spark = SparkSession.builder.appName("InstallLibsAndCloneRepo").getOrCreate()
  try:
    install_requirements_and_clone_repo(spark)
    print("Installation and cloning completed successfully!")
  except Exception as e:
    print(f"An error occurred: {e}")
  finally:
    spark.stop()