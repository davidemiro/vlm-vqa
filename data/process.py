from raw import get_dataset
from configs import configs
from vlm.configuration_vlm import VLMConfig
from vlm.processing_vlm import VLMProcessor
from raw import RawDataCollator
import multiprocessing

config = configs.load_configs()["TRAIN"]

dataset_train = get_dataset(config['train_annotations_path'], config['train_questions_path'])
dataset_val = get_dataset(config['val_annotations_path'], config['val_questions_path'])

vlm_config = VLMConfig()
processor = VLMProcessor(vlm_config)

data_collator_train = RawDataCollator(processor, config['train_img_path'], split='train')
data_collator_val = RawDataCollator(processor, config['val_img_path'], split='val')

dataset_train = dataset_train.map(data_collator_train, batched=False, num_proc=multiprocessing.cpu_count())
dataset_val = dataset_val.map(data_collator_val, batched=False, num_proc=multiprocessing.cpu_count())

dataset_train.save_to_disk(config['train_path'])
dataset_val.save_to_disk(config['val_path'])