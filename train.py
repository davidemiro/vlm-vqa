import os
import vlm
from transformers import Trainer, TrainingArguments, AutoConfig, AutoModel
from transformers import AutoTokenizer

import configs
from dataset import get_dataset


config = configs.load_configs()["TRAIN"]
os.environ['HF_HOME'] = config['cache']

dataset_train = get_dataset(config['train_annotations_path'], config['train_questions_path'])
dataset_val = get_dataset(config['val_annotations_path'], config['val_questions_path'])

tokenizer = AutoTokenizer.from_pretrained(config["name"], token=config['token'])
tokenizer = vlm.vlm_tokenizer(tokenizer)


vlm_config = AutoConfig.from_pretrained(config['name'], token=config['token'])
vlm_config.visual_embed_dim = 768
vlm_model = vlm.VLMGemma2ForCausalLM.from_pretrained(config['name'], config=vlm_config, token=config['token'])
num_patches = (vlm_model.vit.config.patch_size + 2) ** 2 + 1


data_collator_train = vlm.VLMDataCollator(tokenizer, vlm_config.max_position_embeddings, num_patches, config['train_img_path'], split='train')
data_collator_val = vlm.VLMDataCollator(tokenizer, vlm_config.max_position_embeddings, num_patches, config['val_img_path'], split='val')
data_collator_batch = vlm.VLMBatchDataCollator()

dataset_train.map(data_collator_train, batched=False)
dataset_val.map(data_collator_val, batched=False)


training_args = TrainingArguments(
    output_dir=config["output_dir"],
    eval_strategy="epoch",
    learning_rate=float(config["learning_rate"]),
    weight_decay=float(config["weight_decay"]),
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
    batch_size=config["batch_size"],
    num_train_epochs=config["num_train_epochs"],
)


trainer.train()


