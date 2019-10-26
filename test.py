from __future__ import division

from model import *
from utils.util import *
import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

def evaluate(model, file, iou_thres, conf_thres, nms_thres, img_size, batch_size):
	##Set model in evaluation mode##
	model.eval()

	##Get dataset from given file##
	dataset = GetDataset(file, img_size=img_size, augment=False, multiscale=False)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)

	Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

	labels = []
	metrics = []

	for idx,(images, targets) in enumerate(tqdm.tqdm(dataloader)):
		labels += targets[:,1].tolist()

