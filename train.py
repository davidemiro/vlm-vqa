import os
import vlm
from transformers import Trainer, TrainingArguments, AutoConfig, AutoModel
from transformers import AutoTokenizer
import multiprocessing
import configs
import datasets


config = configs.load_configs()["TRAIN"]

dataset_train = datasets.load_from_disk(config['train_path'])
dataset_val = datasets.load_from_disk(config['val_path'])

tokenizer = AutoTokenizer.from_pretrained(config["name"], token=config['token'])
tokenizer = vlm.vlm_tokenizer(tokenizer)

vlm_config = AutoConfig.from_pretrained(config['name'], token=config['token'])
vlm_config.visual_embed_dim = 768
vlm_config.pad_token_id = tokenizer.pad_token_id
vlm_config.image_token_id = tokenizer.convert_tokens_to_ids('<image>')
vlm_config.vit_name = config["vit_name"]


vlm_model = vlm.VLMGemma2ForCausalLM.from_pretrained(config['name'], config=vlm_config, token=config['token'])
num_patches = (vlm_model.vit.config.patch_size + 2) ** 2 + 1

data_collator_batch = vlm.VLMBatchDataCollator()

training_args = TrainingArguments(
    output_dir=config["output_dir"],
    eval_strategy="epoch",
    learning_rate=float(config["learning_rate"]),
    weight_decay=float(config["weight_decay"]),
    per_device_train_batch_size=int(config["batch_size"]),
    num_train_epochs=int(config["num_train_epochs"]),
    optim=config["optim"],
    push_to_hub=True,

)

trainer = Trainer(
    model=vlm_model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    data_collator=data_collator_batch,
    tokenizer=tokenizer,
)


trainer.train()



