import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np

#TOD0: remove
import os
os.environ['TRANSFORMERS_CACHE'] = "/Volumes/TOSHIBA/HuggingFaceCache"

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def load_image(path, dir, image_id):
    image_id = "0" * (12 - len(image_id)) + image_id
    img = Image.open(os.path.join(path, dir, "COCO_{}_{}.jpg".format(dir, image_id)))
    img = img.resize((224, 224))
    img = np.array(img)

    img = img.transpose((2, 0, 1))  # (channel, height, width)

    img = img / 255.0  # normalization
    for i in range(3):
        img[i, :, :] = (img[i, :, :] - IMAGENET_DEFAULT_MEAN[i]) / IMAGENET_DEFAULT_STD[i]

    img = img[None, ...]  # add batch dimension

    return torch.Tensor(img)





















