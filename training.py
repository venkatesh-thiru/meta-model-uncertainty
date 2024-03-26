import os
os.environ["CUDA_VISIBLE_DEVICES"]=""

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
            if probabilistic:
                mean, var = model(inps)
                if betanll:
                    loss = gaussian_beta_log_likelihood_loss(mean, var, predicted_iou, beta = 0.5)
                else:
                    loss = gaussian_log_likelihood_loss(mean=mean, var=var, target=predicted_iou)
            else:
                logit = model(inps)
                loss = criterion(input = logit, target=predicted_iou)
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
            if probabilistic:
                mean, var = model(inps)
                if betanll:
                    loss = gaussian_beta_log_likelihood_loss(mean, var, predicted_iou, beta = 0.5)
                else:
                    loss = gaussian_log_likelihood_loss(mean=mean, var=var, target=predicted_iou)
                probabilistic_mean += criterion(input = mean, target = predicted_iou)
            else:
                logit = model(inps)
                loss = criterion(input = logit, target=predicted_iou)
        overall_loss.append(loss.item())
    return statistics.mean(overall_loss), probabilistic_mean/len(validation_dataset)


checkpoint_path = "checkpoint"
batch_size = 7
EPOCHS = 100
datadir = "DATA/iou_prediction"
training_name = "iou_prediction_resnet101_MSE"
# training_name = "iou_prediction_resnet50_Probabilistic_gnll_v2"
image_width = 512
image_height = 512
patch_size = 8
transform = tvt.Compose([Resize_with_pad(w=image_width, h = image_height)])
# backbone = "convnext_base.fb_in22k_ft_in1k"
# backbone = "convnext_xlarge.fb_in22k_ft_in1k"
# backbone = "efficientnet_b5.sw_in12k"
backbone = "resnet101"
probabilistic = False
betanll = False


wand_config={
    "training_name":training_name,
    "EPOCHS":EPOCHS,
    "image_width, image_height":(image_width, image_height),
    "backbone":backbone,
    "patch size":patch_size
    }

wandb.init(project = f"iou_prediction_CSEK", name = training_name, entity="v3nkyc0d3z", config=wand_config)

train_dataset = iou_prediction_dataset(os.path.join(datadir, "Train"), transform=transform)
train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size, num_workers=4)

validation_dataset = iou_prediction_dataset(os.path.join(datadir, "Validation"), transform=transform)
validation_loader = DataLoader(dataset = validation_dataset, batch_size=batch_size, num_workers = 4)

model = iou_predictor(model_name = backbone, probabilistic=probabilistic).cuda()
# model = iou_predictor_vit(image_size = 512, patch_size=8).cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(params = model.parameters(), lr = 0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor = 0.75, patience=10, threshold=0.000001, min_lr=1e-6)
scaler = torch.cuda.amp.GradScaler()

wandb.watch(model)

old_validation_loss = 0

for epoch in range(EPOCHS):
    training_loss = train()
    validation_loss, probabilistic_mean_loss = validation()
    print(f"Validation Loss : {validation_loss} | Probabilistic Mean: {probabilistic_mean_loss}")

    scheduler.step(validation_loss)

    wandb.log(
        {
            "training loss":training_loss,
            "validation loss":validation_loss,
            "learning rate":optimizer.param_groups[0]['lr']
        }
    )

    if (old_validation_loss == 0) or (old_validation_loss > validation_loss):
        torch.save({
            "epoch":epoch,
            "model_state_dict":model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(),
            "loss":validation_loss
        }, os.path.join(checkpoint_path, f"{training_name}.pth"))
        old_validation_loss=validation_loss
        print("model saved.")
