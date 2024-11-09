import datasets
import pandas
import torch


def get_dataset(annotations_path, questions_path):
    annotations = datasets.load_dataset("json", data_files=annotations_path, field="annotations")
    annotations = annotations["train"]
    annotations = pandas.DataFrame(annotations).iloc[:100]
    annotations = annotations.drop(columns=['question_type', 'answers', 'answer_type', 'image_id'])
    annotations.rename(columns={'multiple_choice_answer': 'answer'}, inplace=True)

    questions = datasets.load_dataset("json", data_files=questions_path, field="questions")
    questions = questions["train"]
    questions = pandas.DataFrame(questions).iloc[:100]

    dataset = pandas.merge(questions, annotations, on="question_id")

    return datasets.Dataset.from_pandas(dataset)


