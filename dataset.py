import datasets
import pandas


def get_dataset(annotations_path, questions_path):
    annotations = datasets.load_dataset("json", data_files=annotations_path, field="annotations")
    annotations = annotations["train"].select(range(10))
    annotations = pandas.DataFrame(annotations)
    annotations = annotations.drop(columns=['question_type', 'answers', 'answer_type', 'image_id'])
    annotations.rename(columns={'multiple_choice_answer': 'answer'}, inplace=True)

    questions = datasets.load_dataset("json", data_files=questions_path, field="questions")
    questions = questions["train"].select(range(10))
    questions = pandas.DataFrame(questions)

    dataset = pandas.merge(questions, annotations, on="question_id")

    return datasets.Dataset.from_pandas(dataset)


