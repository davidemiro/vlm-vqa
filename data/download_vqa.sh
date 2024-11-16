.#!/bin/sh

sudo wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
sudo unzip v2_Annotations_Train_mscoco.zip

sudo wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
sudo unzip v2_Annotations_Val_mscoco.zip

sudo wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
sudo unzip v2_Questions_Train_mscoco.zip

sudo wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
sudo unzip v2_Questions_Val_mscoco.zip

sudo wget http://images.cocodataset.org/zips/train2014.zip
sudo unzip train2014.zip

sudo wget http://images.cocodataset.org/zips/val2014.zip
sudo unzip val2014.zip