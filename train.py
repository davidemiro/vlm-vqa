import os
import datasets
from PIL import Image
import vlm
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling, AutoModel, AutoTokenizer, Gemma2Config

import configs




config = configs.load_configs()["TRAIN"]
os.environ['HF_HOME'] = config['cache']


dataset = datasets.load_dataset("json", data_files=config["path"], field="questions")
dataset_val = datasets.load_dataset("json", data_files=config["path"], field="questions")

vit = AutoModel.from_pretrained('facebook/dinov2-base')

tokenizer = AutoTokenizer.from_pretrained(config["name"], token=config['token'])
tokenizer = vlm.vlm_tokenizer(tokenizer)

gemma_config = Gemma2Config()
gemma_config.visual_embed_dim = vit.config.hidden_size
llm = vlm.VLMGemma2ForCausalLM.from_pretrained(config["name"], token=config['token'])


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


training_args = TrainingArguments(
    output_dir=config["output_dir"],
    eval_strategy="epoch",
    learning_rate=float(config["learning_rate"]),
    weight_decay=float(config["weight_decay"]),
    push_to_hub=True,
)

trainer = Trainer(
    model=llm,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset_val,
    data_collator=data_collator,
)

trainer.train()


