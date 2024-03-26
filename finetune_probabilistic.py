import os
os.environ["CUDA_VISIBLE_DEVICES"]="GPU-f16670b1-ac4a-8a32-02cb-60439c6bf799"

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from dataset.iou_detector_dataset import iou_prediction_dataset
from dataset.transforms import Resize_with_pad
from models.iou_predictor.iou_predictor import iou_predictor
from models.iou_predictor.iou_predictor_vit import iou_predictor_vit
from utils.losses import gaussian_log_likelihood_loss, gaussian_beta_log_likelihood_loss
import torchvision.transforms as tvt
import statistics
import wandb
from tqdm import tqdm
import numpy as np

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True

def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight)

def train():
    model.train()
    overall_loss = []
    for i, batch in enumerate(tqdm(train_loader)):
        image, mask, predicted_mask, predicted_iou = batch['image'].cuda(), batch['target_mask'].cuda(), batch['predicted_mask'].cuda(), batch['predicted_iou'].cuda().unsqueeze(dim = -1)
        inps = torch.cat([image, predicted_mask], dim = 1)
        with torch.autocast(device_type="cuda"):
            mean, var = model(inps)
            if betanll:
                loss = gaussian_beta_log_likelihood_loss(mean, var, predicted_iou, beta = 0.5)
            else:
                loss = gaussian_log_likelihood_loss(mean=mean, var=var, target=predicted_iou)
        if torch.isnan(loss):
            print(f"nan loss in batch number {i+1}")
            pass
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            overall_loss.append(loss.item())
        if not i%1000:
            print(f"epoch iter {i} : {statistics.mean(overall_loss)}")
    return statistics.mean(overall_loss)


def validation():
    model.eval()
    overall_loss = []
    probabilistic_mean = 0
    for i, batch in enumerate(tqdm(validation_loader)):
        image, mask, predicted_mask, predicted_iou = batch['image'].cuda(), batch['target_mask'].cuda(), batch['predicted_mask'].cuda(), batch['predicted_iou'].cuda().unsqueeze(dim = -1)
        inps = torch.cat([image, predicted_mask], dim = 1)
        with torch.no_grad():
            mean, var = model(inps)
            if betanll:
                loss = gaussian_beta_log_likelihood_loss(mean, var, predicted_iou, beta = 0.5)
            else:
                loss = gaussian_log_likelihood_loss(mean=mean, var=var, target=predicted_iou)
            probabilistic_mean += criterion(input = mean, target = predicted_iou)
        overall_loss.append(loss.item())
    return statistics.mean(overall_loss), probabilistic_mean/len(validation_dataset)


checkpoint_path = "checkpoint"
batch_size = 32
EPOCHS = 100
datadir = "DATA/iou_prediction"
training_name = "iou_prediction_resnet50_Probabilistic_gnll_finetune_variance"
image_width = 512
image_height = 512
patch_size = 8
transform = tvt.Compose([Resize_with_pad(w=image_width, h = image_height)])
backbone = "resnet50"
betanll = False
probabilistic = True

model = iou_predictor(model_name = backbone, probabilistic=probabilistic).cuda()

for name, params in model.named_parameters():
    if not "regressor_std" in name:
        params.requires_grad = False