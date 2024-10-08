import os
import datasets
from PIL import Image
import vlm
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling, AutoModelForCausalLM, AutoTokenizer
import configs




config = configs.load_configs()["TRAIN"]
os.environ['HF_HOME'] = config['cache']


dataset = datasets.load_dataset("json", data_files=config["path"], field="questions")
dataset_val = datasets.load_dataset("json", data_files=config["path"], field="questions")


tokenizer = AutoTokenizer.from_pretrained(config["name"], token=config['token'])
model = AutoModelForCausalLM.from_pretrained(config["name"], token=config['token'])

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


training_args = TrainingArguments(
    output_dir=config["output_dir"],
    eval_strategy="epoch",
    learning_rate=float(config["learning_rate"]),
    weight_decay=float(config["weight_decay"]),
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset_val,
    data_collator=data_collator,
)

trainer.train()


