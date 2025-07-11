import datasets
import pandas
from PIL import Image
import os
from vlm.processing_vlm import VLMProcessor

from transformers import DefaultDataCollator


def get_dataset(annotations_path, questions_path, prompt, p):
    annotations = datasets.load_dataset("json", data_files=annotations_path, field="annotations")
    annotations = annotations["train"]
    annotations = pandas.DataFrame(annotations)
    annotations = annotations.drop(columns=['question_type', 'answers', 'answer_type', 'image_id'])
    annotations.rename(columns={'multiple_choice_answer': 'answer'}, inplace=True)

    questions = datasets.load_dataset("json", data_files=questions_path, field="questions")
    questions = questions["train"]
    questions = pandas.DataFrame(questions)

    dataset = pandas.merge(questions, annotations, on="question_id")
    dataset['question'] = dataset['question'].apply(lambda x: prompt.format(x))


    dataset = datasets.Dataset.from_pandas(dataset)
    dataset = dataset.shuffle()


    return dataset.select(range(int(p * len(dataset))))


class RawDataCollator(DefaultDataCollator):
    def __init__(self, processor: VLMProcessor, img_path, split='train') -> None:
        self.img_path = img_path
        self.split = split
        self.processor = processor

    def _load_image(self, path, split, image_id):
        image_id = "0" * (12 - len(str(image_id))) + str(image_id)
        img = Image.open(os.path.join(path, "COCO_{}2014_{}.jpg".format(split, image_id)))

        return img

    def __call__(self, row):
        image = self._load_image(self.img_path, self.split, row['image_id'])
        return self.processor(text=row['question'], image=image, label=row['answer'], return_tensors="np")


