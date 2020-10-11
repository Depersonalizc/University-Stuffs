import os
import random
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class DataBuilder:
    def __init__(self, img_size=256, img_dir="./data/JPEGImages", mask_dir="./data/JPEGImages"):
        self.img_size = img_size
        self.imgs = img_dir
        self.masks = mask_dir
        self.count = 0
        self.training_data = []

    def build_training_data(self, save=False, name='training_data'):
        self.count = 0
        self.training_data = []
        s = self.img_size

        # file names
        img_list = os.listdir(self.imgs)
        mask_list = os.listdir(self.masks)

        # augmentation transforms
        img_aug = T.Compose([
            T.ToTensor(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # normalized to [-1, 1]
            T.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
        ])
        mask_aug = T.Compose([T.ToTensor(), ])

        for mask in tqdm(mask_list):
            try:
                img = mask.replace('-removebg-preview.png', '.jpg')
                if img in img_list:
                    # fetch RGB images
                    img_path = os.path.join(self.imgs, img)
                    img = Image.open(img_path).convert("RGB")
                    h, w = img.size

                    # fetch binary masks
                    mask_path = os.path.join(self.masks, mask)
                    mask = Image.open(mask_path).split()[-1]  # alpha channel
                    mask = mask.resize((h, w), Image.BILINEAR)

                    # resizing
                    new_size = (s, s * w // h) if h > w else (s * h // w, s)
                    img = img.resize(new_size, Image.BILINEAR)
                    mask = mask.resize(new_size, Image.BILINEAR)

                    # augmentation & padding
                    img = img_aug(img)
                    mask = mask_aug(mask)
                    dh, dw = s - new_size[0], s - new_size[1]
                    padding = (dh // 2, dh - (dh // 2), dw // 2, dw - (dw // 2))
                    pad = nn.ConstantPad2d(padding, 0)
                    img = pad(img)
                    mask = pad(mask)
                    if random.random() < 0.5:
                        img = T.functional.hflip(img)
                        mask = T.functional.hflip(mask)

                    # append data point
                    data = [img, mask]
                    self.training_data.append(data)
                    self.count += 1

            except Exception as e:
                pass

        print("Built", self.count, "data points!")
        if save: self.save_training_data(name)

    def save_training_data(self, name='training_data'):
        torch.save(self.training_data, name + '.pt')

    def load_training_data(self, data='./data/training_data.pt'):
        self.training_data = torch.load(data)
        self.img_size = self.training_data[0][1].size()[-1]
        self.count = len(self.training_data)
        print("Loaded", self.count, "data points!")

    def show_data_point(self, i):
        if i < len(self):
            data = self.training_data
            f = plt.figure()
            f.add_subplot(1, 2, 1)
            plt.imshow(data[i][0].permute(1, 2, 0) * .5 + .5)
            f.add_subplot(1, 2, 2)
            plt.imshow(data[i][1].permute(1, 2, 0))
            plt.show()
        else:
            print('Index {} out of range ([0:{}]).'.format(i, len(self)-1))

    def __len__(self):
        return self.count


if __name__ == "__main__":
    data_builder = DataBuilder(256)
    # data_builder.build_training_data(save=True, name='./data/example')
    # data_builder.show_data_point(15)
    # print(len(data_builder))

    data_builder.load_training_data('H:/data/example.pt')
    data_builder.show_data_point(50)
    print(len(data_builder))
