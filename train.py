from vlm.modelling_vlm import VLMForCausalLM
from vlm.configuration_vlm import VLMConfig
from vlm.utils_vlm import BatchDataCollator
from vlm.processing_vlm import VLMProcessor
from transformers import Trainer, TrainingArguments
from configs import configs
from data.raw import get_dataset
import torch
import multiprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(1)
torch.cuda.current_device()

config = configs.load_configs()["TRAIN"]

dataset_train = get_dataset(config['train_annotations_path'], config['train_questions_path'])
dataset_val = get_dataset(config['val_annotations_path'], config['val_questions_path'])


dataset_train = dataset_train.add_column("split", ["train"] * len(dataset_train))
dataset_val = dataset_val.add_column("split", ["val"] * len(dataset_val))

dataset_train = dataset_train.add_column("img_path", [config['train_img_path']] * len(dataset_train))
dataset_val = dataset_val.add_column("img_path", [config['val_img_path']] * len(dataset_val))


vlm_config = VLMConfig()
processor = VLMProcessor(vlm_config)
vlm_model = VLMForCausalLM.from_pretrained("google/gemma-2-2b-it", config=vlm_config, token=config['token'])
vlm_model.to(device)

data_collator_batch = BatchDataCollator(processor)

training_args = TrainingArguments(
    output_dir=config["output_dir"],
    eval_strategy="epoch",
    learning_rate=float(config["learning_rate"]),
    weight_decay=float(config["weight_decay"]),
    per_device_train_batch_size=int(config["batch_size"]),
    num_train_epochs=int(config["num_train_epochs"]),
    optim=config["optim"],
    push_to_hub=True,
    remove_unused_columns=False,

)

print(training_args.device)

trainer = Trainer(
    model=vlm_model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    data_collator=data_collator_batch,
)


trainer.train()




