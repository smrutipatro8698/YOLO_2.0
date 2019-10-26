import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
import torchvision.transforms as transforms

##Data augmentation function goes here##
def flip_horizontal(images, [-1]):
	images = torch.flip(images,[-1])
	targets[:, 2] = 1-targets[:,2]
	return images, targets


#########Dataset Loader helper functions#############

##To convert rectangular image to square image first pad the image##
def square_pad(image):
    channels, height, width = image.shape
    diff = np.abs(height - width)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = diff // 2, diff - diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if height <= width else (pad1, pad2, 0, 0)
    # Add padding
    image = F.pad(image, pad, "constant", value=0)


    return image, pad

##Resize image to match YOLO's input dimension##
def resize(image, img_size):
    image = F.interpolate(image.unsqueeze(0), size=img_size, mode="nearest").squeeze(0)
    return image

##Left 2 functions, add later on.... ###
class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        img = transforms.ToTensor()(Image.open(img_path))
        img, _ = square_pad(img)
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)

##Helper class to get dataset##
class GetDataset(dataset):
	##Check parameters once##
	def __init__(self, images_path, img_size=416, augment=True, multiscale=True, normalised_labels=False):
		##Read image file names##
		with open(images_path,'r') as fp:
			self.image_files = fp.readlines()

		self.label_files = [path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt") for path in self.image_files]

		self.image_size = img_size
		self.augment = augment
		self.multiscale = multiscale
		self.normalised_labels = normalised_labels
		self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0


	def __getitem__(self, idx):
		##Get the image file path##
		imgfile_name = self.image_files[idx % len(image_files)].rstrip()

		image = transforms.ToTensor()(Image.open(imgfile_name).convert('RGB'))

		c, h, w = image.shape
		h_factor, w_factor = (h, w) if self.normalised_labels else (1, 1)

		image, pad = square_pad(image)
		pad_c, pad_w, pad_h = image.shape

		labelfile_name = self.label_files[idx % len(self.image_files)].rstrip()

		targets = None
		if os.path.exists(labelfile_name):
			box = torch.from_numpy(np.loadtxt(labelfile_name).reshape(-1,5))

			x1 = w_factor * (box[:,1] - (box[:,3]/2))
			y1 = h_factor * (box[:,2] - (box[:,4]/2))
			x2 = w_factor * (box[:,1] + (box[:,3]/2))
			y2 = h_factor * (box[:,2] + (box[:,4]/2))

			x1 += pad[0]
			y1 += pad[2]
			x2 += pad[1]
			y2 += pad[3]

			##Check this!!!##
			box[:,1] = ((x1+x2)/2) / pad_w
			box[:,1] = ((x1+x2)/2) / pad_w
			box[:, 3] *= w_factor / pad_w
            box[:, 4] *= h_factor / pad_h

            targets = torch.zeros((len(box), 6))
            targets[:,1:] = box

            if (self.augment):
            	if np.random.random() < 0.5:
            		image, targets = flip_horizontal(image, targets)

            return image, targets

    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))
        targets = [boxes for boxes in targets if boxes is not None]
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        if self.multiscale and self.batch_count % 10 == 0:
            self.image_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        imgs = torch.stack([resize(img, self.image_size) for img in imgs])
        self.batch_count += 1
        return imgs, targets

    def __len__(self):
        return len(self.image_files)




####Parsing config files helper functions####

##Differentt!!!!###
def parse_cfg_model(config_path):

	file = open(config_path, "r")

	##Different!!##
	content = file.read().split('\n')
    content = [x for x in content if len(x) > 0]
    content = [x for x in content if x[0] != '#']
    content = [x.rstrip().lstrip() for x in content]
    content = [x for x in content if len(x) > 0]               #Stripping one more time to get rid of empty strings

    blocks = []
    for line in content:
    	if line.startswith('['):
    		blocks.append({})
    		blocks[-1]['block_type'] = line[1:-1].rstrip()
    		if (blocks[-1]['block_type'] == 'convolutional'):
    			blocks[-1]['batch_normalize'] = 0
    	else:
    		key, value = line.split("=")
            value = value.strip()
            blocks[-1][key.rstrip()] = value.strip()
    
    return blocks

def parse_cfg_data(config_path):
    data = dict()
    data['gpus'] = '0'
    data['num_workers'] = '4'
    with open(config_path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        data[key.strip()] = value.strip()
    return data

