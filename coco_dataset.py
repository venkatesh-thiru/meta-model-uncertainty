import torch
from torch.utils.data import Dataset

import torchvision.transforms as tvt
import torchvision.transforms.functional as tvtf

from pycocotools.coco import COCO

from PIL import Image
import os

class coco_dataset(Dataset):
    def __init__(self, annotations_file, img_dir):

        self.coco = COCO(annotations_file)
        self.ids = [str(k) for k in self.coco.imgs]
        self.classes = {k: v["name"] for k, v in self.coco.cats.items()}
        self.img_dir = img_dir

    def get_image(self, image_id):
        img_id = int(image_id)
        img_info = self.coco.imgs[img_id]
        image = Image.open(os.path.join(self.img_dir, img_info['file_name']))
        return image.convert('RGB')
    
    def convert_to_xyxy(boxes): # box format: (xmin, ymin, w, h)
        x, y, w, h = boxes.T
        return torch.stack((x, y, x + w, y + h), dim=1) # new_box format: (xmin, ymin, xmax, ymax)

    def get_target(self, img_id):
        img_id = int(img_id)
        ann_ids = self.coco.getAnnIds(img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        labels = []
        masks = []

        if len(anns)>0:
            for ann in anns:
                boxes.append(ann['bbox'])
                labels.append(ann['category_id'])
                mask = self.coco.annToMask(ann)
                mask = torch.tensor(mask, dtype=torch.uint8)
                masks.append(mask)
            
            boxes = torch.tensor(boxes, dtype=torch.float32)
            boxes = self.convert_to_xyxy(boxes)
            labels = torch.tensor(labels)
            masks = torch.stack(masks)
        target = dict(image_id=torch.tensor([img_id]), boxes=boxes, labels=labels, masks=masks)
        return target
    
    def __getitem__(self, index):
        image_id = self.ids[index]
        image = self.get_image(image_id)
        target = self.get_target(image_id)
        return image, target


    def __len__(self):
        return len(self.ids)