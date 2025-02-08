from pyspark.sql import SparkSession
from datasets import Dataset
from configs import configs
from pyspark.sql import functions as F
from PIL import Image
from vlm.configuration_vlm import VLMConfig
from vlm.processing_vlm import VLMVQAProcessor
import os
import zipfile

def unzip_file(zip_file_path, output_dir=None):
  """
  Unzips a given zip file.

  Args:
    zip_file_path: Path to the zip file.
    output_dir: Optional path to the output directory.
                 If None, extracts to the current directory.

  Raises:
    FileNotFoundError: If the zip file does not exist.
  """


  try:
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
      zip_ref.extractall(output_dir)
  except FileNotFoundError:
    print("Error: Zip file {} not found.".format(zip_file_path))


def _load_image(path, split, image_id):
    image_id = "0" * (12 - len(str(image_id))) + str(image_id)
    img = Image.open(os.path.join(path, "COCO_{}2014_{}.jpg".format(split, image_id)))
    return img


def map_df(row):
    image = _load_image(row['img_path'], row['split'], row['image_id'])
    return processor(text=row['question'], image=image, label=row['answer'], return_tensors="np")


def process(spark, annotations_path, questions_path, save_path, split, img_path, is_zip):

    if is_zip:
        img_zip_path = img_path + ".zip"
        print("Extracting zip file {}".format(img_zip_path))
        unzip_file(img_zip_path, None)


    annotations_df = spark.read.json(annotations_path)
    annotations_df = annotations_df.select(annotations_df.annotations)
    annotations_df = annotations_df.drop(columns=['question_type', 'answers', 'answer_type', 'image_id'])
    annotations_df.withColumnRenamed('multiple_choice_answer', 'answer')

    questions_df = spark.read.json(questions_path)
    questions_df = questions_df.select(questions_df.questions)

    df = annotations_df.join(questions_df, on="question_id", how="inner")
    df = df.withColumn("split", F.lit(split))
    df = df.withColumn("img_path", F.lit(img_path))

    df = df.map(map_df)

    # Convert PySpark DataFrame to Pandas DataFrame
    pandas_df = df.toPandas()

    # Convert Pandas DataFrame to Hugging Face Dataset
    hf_dataset = Dataset.from_pandas(pandas_df)

    hf_dataset.save_to_disk(save_path)


config = configs.load_configs()["PROCESS"]

# Spark session
spark = SparkSession.builder \
    .appName("Create VQA dataset for VLM") \
    .getOrCreate()

vlm_config = VLMConfig(text_length=int(config["text_length"]), num_patches=int(config["num_patches"]), visual_embed_dim=int(config["visual_embed_dim"]))
processor = VLMVQAProcessor(vlm_config)

process(spark, config['train_annotations_path'], config['train_questions_path'], config['train_path'], config['train_img_path'],bool(int(config["zip"])))

process(spark, config['val_annotations_path'], config['val_questions_path'], config['val_path'], config['val_img_path'],bool(int(config["zip"])))


spark.stop()