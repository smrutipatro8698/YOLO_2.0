from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

##Different##

import matplotlib.pyplot as plt
import matplotlib.patches as patches

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class Upsample(nn.Module):
    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

class YOLO_layer(nn.Module):
	def __init__(self, anchors, no_of_classes, image_dim=416):
		super(YOLO_layer, self).__init__()
		self.anchors = anchors
		self.no_of_anchors = len(anchors)
		self.no_of_classes = no_of_classes
		self.threshold = 0.5
		self.mse_loss = nn.MSELoss()
		self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.grid_size = 0  # grid size
		self.image_dim = image_dim

	def grid_offset(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.image_dim / self.grid_size
        
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.no_of_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.no_of_anchors, 1, 1))

    def forward(self, x, targets=None, image_dim=None):
    	FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.image_dim = image_dim
        no_of_samples = x.size(0)
        grid_size = x.size(2)

        pred = (x.view(no_of_samples, self.no_of_anchors, self.no_of_classes+5, grid_size, grid_size).permute(0,1,3,4,2).contiguous())

        # Get outputs
        x = torch.sigmoid(pred[..., 0])  # Center x
        y = torch.sigmoid(pred[..., 1])  # Center y
        w = pred[..., 2]  # Width
        h = pred[..., 3]  # Height
        pred_conf = torch.sigmoid(pred[..., 4])  # Conf
        pred_cls = torch.sigmoid(pred[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.grid_offset(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat((pred_boxes.view(no_of_samples, -1, 4) * self.stride, pred_conf.view(no_of_samples, -1, 1), pred_cls.view(no_of_samples, -1, self.no_of_classes),),-1,)

        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(pred_boxes=pred_boxes,pred_cls=pred_cls,target=targets,anchors=self.scaled_anchors,ignore_thres=self.threshold,)

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }



##add##
def build_modules(blocks):

	params_net = blocks.pop(0)

	##Find the number of output channels##
	output_filters = [int(params_net["channels"])]

	##Maintain a list of modules##
	list_of_modules = nn.ModuleList()

	for idx, block in enumerate(blocks):
		modules = nn.Sequential()

		if(blocks["block_type"] == "convolutional"):
			##Find batch normalise there or not##
			batch_norm = int(blocks["batch_normalize"])
			filters = int(blocks["filters"])
            kernel_size = int(blocks["size"])
            stride = int(blocks["stride"])
            pad = (kernel_size - 1) // 2

            conv = nn.Conv2d(in_channels = output_filters[-1], out_channels=filters, kernel_size=kernel_size, stride=,padding=pad,bias=not batch_norm)
            modules.add_module("conv_{0}".format(idx), conv)

            if batch_norm:
            	batchNorm = nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5)
            	modules.add_module("batch_norm_{0}".format(idx), batchNorm)
            if (blocks["activation"] == "leaky"):
            	actvn = nn.LeakyReLU(0.1)
            	modules.add_module("leaky_{0}".format(idx), actvn)

        elif (blocks["block_type"] == "maxpool"):
        	kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            # if kernel_size == 2 and stride == 1:
            #     modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module("maxpool_{0}".format(idx), maxpool)

        elif (blocks["block_type"] == "upsample"):
        	stride = int(module_def["stride"])
            upsample = Upsample(scale_factor=stride, mode="nearest")
            modules.add_module("upsample_{0}".format(idx), upsample)

        elif (blocks["block_type"] == "route"):
            layers = [int(x) for x in blocks["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])

            route = EmptyLayer()
            modules.add_module("route_{0}".format(idx), route)

        elif (blocks["block_type"] == "shortcut"):
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module("shortcut_{}".format(idx), EmptyLayer())

        elif (blocks["block_type"] == "yolo"):
        	#Get masks/ indices of anchor boxes to use
        	indices = blocks["mask"].split(",")

        	#parse into integers
        	indices = [int(a) for a in indices]

        	#Get anchor sizes
        	anchor = blocks["anchors"].split(",")
        	#Typecast strings into int
        	anchors = [int(a) for a in anchor]

        	#Pair width and height into one
        	anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
        	#Now take only those pairs as required in the mask
            anchors = [anchors[i] for i in indices]

            no_of_classes = int(blocks["classes"])
            image_height = int(params_net["height"])
            # Define detection layer
            yolo_layer = YOLO_layer(anchors, no_of_classes, image_height)
            modules.add_module("yolo_{0}".format(idx), yolo_layer)

        list_of_modules.append(modules)
        output_filters.append(filters)

    return params_net, list_of_modules

class YOLOModel(nn.Module):
    def __init__(self, cfg_file, img_size=416):
        super(YOLOModel, self).__init__()
        self.blocks = parse_model_config(cfg_file)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None):
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                x, layer_loss = module[0](x, targets, img_dim)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_YOLO_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w




