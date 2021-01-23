import os
import torch
import cv2
import warnings
import pandas as pd
import numpy as np
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from glob import glob
from PIL import Image,ImageFile
from config import configs
# from models.model import get_model
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms, models
from utils.misc import get_files
from IPython import embed
import time

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = configs.gpu_id
len_data = 0

class WeatherTTADataset(Dataset):
    def __init__(self,labels_file,aug):
        imgs = []
        for index, row in labels_file.iterrows():
            imgs.append((row["filename"],row["label"]))
        self.imgs = imgs
        self.length = len(imgs)
        global len_data
        len_data = self.length
        self.aug = aug
        self.Hflip = transforms.RandomHorizontalFlip(p=1)
        self.Vflip = transforms.RandomVerticalFlip(p=1)
        self.Rotate = transforms.functional.rotate
        self.resize = transforms.Resize((configs.input_size,configs.input_size))
        self.randomCrop = transforms.Compose([transforms.Resize(int(configs.input_size * 1.2)),
                                            transforms.CenterCrop(configs.input_size),
                                            ])
    def __getitem__(self,index):
        filename,label_tmp = self.imgs[index]
        img = Image.open(os.sep + filename).convert('RGB')
        img = self.transform_(img,self.aug)
        return img, filename

    def __len__(self):
        return self.length
    def transform_(self,data_torch,aug):
        if aug == 'Ori':
            data_torch = data_torch
            data_torch = self.resize(data_torch)
        if aug == 'Ori_Hflip':
            data_torch = self.Hflip(data_torch)
            data_torch = self.resize(data_torch)
        if aug == 'Ori_Vflip':
            data_torch = self.Vflip(data_torch)
            data_torch = self.resize(data_torch)
        if aug == 'Ori_Rotate_90':
            data_torch = self.Rotate(data_torch, 90)
            data_torch = self.resize(data_torch)
        if aug == 'Ori_Rotate_180':
            data_torch = self.Rotate(data_torch, 180)
            data_torch = self.resize(data_torch)
        if aug == 'Ori_Rotate_270':
            data_torch = self.Rotate(data_torch, 270)
            data_torch = self.resize(data_torch)
        if aug == 'Crop':
            # print(data_torch.size)
            data_torch = self.randomCrop(data_torch)
            data_torch = data_torch
        if aug == 'Crop_Hflip':
            data_torch = self.randomCrop(data_torch)
            data_torch = self.Hflip(data_torch)
        if aug == 'Crop_Vflip':
            data_torch = self.randomCrop(data_torch)
            data_torch = self.Vflip(data_torch)
        if aug == 'Crop_Rotate_90':
            data_torch = self.randomCrop(data_torch)
            data_torch = self.Rotate(data_torch, 90)
        if aug == 'Crop_Rotate_180':
            data_torch = self.randomCrop(data_torch)
            data_torch = self.Rotate(data_torch, 180)
        if aug == 'Crop_Rotate_270':
            data_torch = self.randomCrop(data_torch)
            data_torch = self.Rotate(data_torch, 270)
        data_torch = transforms.ToTensor()(data_torch)
        data_torch = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(data_torch)
        return data_torch


aug = ['Ori_Hflip']

# best_cpk = './checkpoints/efficient-best_loss.pth.tar'
best_cpk = './checkpoints/best.pth'
checkpoint = torch.load(best_cpk)
cudnn.benchmark = True
# model = get_model()
model = models.mobilenet_v2()
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.BatchNorm1d(in_features),
    nn.Dropout(0.5),
    nn.Linear(in_features, configs.num_classes),
)
model.load_state_dict(checkpoint)
model.cuda().eval()
# test_files = pd.read_csv(configs.submit_example)
data_root = './dataset/1'
test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
i = 0
with torch.no_grad():
    for root, dir, files in os.walk(data_root):
        files = [os.path.join(root, file) for file in files]
        for file in files:
            inputs = cv2.imread(file)
            inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB)
            inputs = cv2.resize(inputs, (112, 112))
            t1 = time.time()
            inputs = test_transform(inputs).unsqueeze(0).cuda()
            outputs = model(inputs)
            # print(outputs)
            outputs = torch.nn.functional.softmax(outputs, dim=1).data.cpu().numpy()[0][0]
            # print('tttttt',time.time() - t1)
            # print(outputs)
            if outputs > 0.5:
                mask = 0
                i += 1
                print('no mask', file, i)
            else:
                mask = 1
                # i += 1
                # print('with mask', file, i)
