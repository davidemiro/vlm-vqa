from transformers import Trainer, TrainingArguments
from configs import configs
from data.raw import get_dataset
import torch
from peft import LoraConfig, get_peft_model
from evaluation.metrics import compute_accuracy

from vlm.utils_vlm import BatchDataCollator, get_vlm



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = configs.load_configs()["TRAIN"]

dataset_train = get_dataset(config['train_annotations_path'], config['train_questions_path'])
dataset_val = get_dataset(config['val_annotations_path'], config['val_questions_path'])


dataset_train = dataset_train.add_column("split", ["train"] * len(dataset_train))
dataset_val = dataset_val.add_column("split", ["val"] * len(dataset_val))

dataset_train = dataset_train.add_column("img_path", [config['train_img_path']] * len(dataset_train))
dataset_val = dataset_val.add_column("img_path", [config['val_img_path']] * len(dataset_val))


processor, vlm_model, vlm_config = get_vlm(config)
processor.push_to_hub(config["output_dir"])
vlm_config.push_to_hub(config["output_dir"])

vlm_model.to(device)

data_collator_batch = BatchDataCollator(processor)

training_args = TrainingArguments(
    output_dir=config["output_dir"],
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",
    learning_rate=float(config["learning_rate"]),
    weight_decay=float(config["weight_decay"]),
    per_device_train_batch_size=int(config["batch_size"]),
    num_train_epochs=int(config["num_train_epochs"]),
    gradient_accumulation_steps=2,
    push_to_hub=True,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=10,
    logging_dir="./logs",
    gradient_checkpointing=True,
    fp16=True,
    fp16_full_eval=True,
    deepspeed="deepspeed/ds_config.json",



)

trainer = Trainer(
    model=vlm_model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    data_collator=data_collator_batch,
    compute_metrics=compute_accuracy,
)


trainer.train()






