import os
from PIL import Image
import vlm
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling, AutoModel, AutoTokenizer, Gemma2Config

import configs
from dataset import get_dataset


config = configs.load_configs()["TRAIN"]
os.environ['HF_HOME'] = config['cache']

dataset_train = get_dataset(config['train_annotations_path'], config['train_questions_path'])
dataset_val = get_dataset(config['val_annotations_path'], config['val_questions_path'])


tokenizer = AutoTokenizer.from_pretrained(config["name"], token=config['token'])
tokenizer = vlm.vlm_tokenizer(tokenizer)


vlm = vlm.VLMGemma2ForCausalLM.from_pretrained(config["name"], token=config['token'])
vlm_preprocessing = vlm.VLMPreprocessing.from_pretrained(config["name"])


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


training_args = TrainingArguments(
    output_dir=config["output_dir"],
    eval_strategy="epoch",
    learning_rate=float(config["learning_rate"]),
    weight_decay=float(config["weight_decay"]),
    push_to_hub=True,
)

trainer = Trainer(
    model=vlm,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    data_collator=data_collator,
    processing_class=vlm_preprocessing
)


trainer.train()


