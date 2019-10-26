from __future__ import division

from model import *
from utils.util import *
from test import evaluate
import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--accum_grad", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--cfg_model", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--cfg_data", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--save_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    args = parser.parse_args()

    ##Check if GPU available##
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   	##Make directories for saving output and state of the models##
    os.makedirs("output",exist_ok=True)
    os.makedirs("savemodels",exist_ok=True)

    data_cfg = parse_cfg_data(args.cfg_data)
    train_file = data_cfg("train")
    validation_file = data_cfg("validation")

    model = YOLOModel(args.cfg_model).to(device)
    model.apply(initialise_weights)

    ##Load specified states or load pre-trained weights##
    if (args.pretrained_weights):
    	if args.pretrained_weights.endswith(".pth"):
    		model.load_state_dict(torch.load(args.pretrained_weights)) ##Loading saved model
    	else:
    		model.load_YOLO_weights(args.pretrained_weights) ##Load pre-trained weights

    dataset = GetDataset(train_file, augment=True, multiscale=args.multiscale_training)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.n_cpu, pin_memory=True, collate_fn=dataset.collate_fn)

    optimiser = torch.optim.Adam(model.parameters())

    for epoch in range(args.epoch):
    	##Set the model to train##
    	model.train()
    	start_time = time.time()
    	for idx, (images, labels) in enumerate(dataloader):
    		##See number of completed batches##
    		j = epoch * len(dataloader) + idx

    		images = Variable(images.to(device))
    		labels = Variable(labels.to(device), requires_grad=False)

    		##Get loss and backpropagate##
    		loss, outputs = model(images, labels)
    		loss.backward()

    		##Optimise every 2 batches##
    		if (j%args.accum_grad):
    			optimiser.step()
    			optimiser.zero_grad()

    		##Removed metrics logging##
    		model.seen+=images.size(0)

    	print(" Training completed for epoch %d \n" %(epoch))
    	print("Total loss in this epoch %d \n" %(loss.item()))
    		##Logging the metrics removed##

    	if(epoch%args.interval == 0):
    		print("Getting training metrics on validation data...\n")
    		precision, recall, AP, f1 = evaluate(model, path=validation_file,iou_thres=0.5, conf_thers=0.5, nms_thres=0.5, img_size=args.img_size, batch_size=4)

    		print("Evaluation metrics \n")
    		print("Precision = %.4f \n" %precision)
    		print("Recall = %.4f \n" %recall)
    		print("mAP = %.4f \n" %AP.mean())
    		print("f1 score %.4f \n" f1.mean())

    	##Save model##
    	if (epoch % args.save_interval == 0):
    		torch.save(model.state_dict(),f"savemodels/yolov3_ckpt_%d.pth" % epoch)










