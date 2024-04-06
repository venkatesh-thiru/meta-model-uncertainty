from torch.utils.data import Dataset, DataLoader
import glob
import h5py
import numpy as np
import torch
import os

class iou_prediction_dataset(Dataset):
    def __init__(self, root, transform = None, return_file_name = False):
        self.root = root
        self.instances = glob.glob(os.path.join(self.root, "*.h5"))
        self.transform = transform
        self.return_file_name = return_file_name
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        file_path = self.instances[index]
        file = h5py.File(file_path, "r")


        image = torch.from_numpy(np.array(file.get("image"))).type(torch.float32)
        target_mask = torch.from_numpy(np.array(file.get("gt_mask"))).unsqueeze(dim = 0).type(torch.float32)
        predicted_mask = torch.from_numpy(np.array(file.get("pred_mask"))).unsqueeze(dim = 0).type(torch.float32)
        iou_predicted = torch.from_numpy(np.array(file.get("predicted_iou"))).type(torch.float32)

        if self.return_file_name:
            score =  torch.from_numpy(np.array(file.get("score"))).type(torch.float32)
        else:
            score = torch.zeros_like(image)
        file.close()

        if 0 in image.shape:
            image = torch.zeros([3, 10, 10])
            target_mask = torch.zeros([1, 10, 10])
            predicted_mask = torch.zeros([1, 10, 10])
            iou_predicted = torch.zeros_like(iou_predicted)
            score = 0.


        if self.transform:
            image = self.transform(image)
            target_mask = self.transform(target_mask)
            predicted_mask = self.transform(predicted_mask)

        if image.shape[0] == 1:
            image = torch.zeros([3, image.shape[-2], image.shape[-1]])
            target_mask = torch.zeros([1, image.shape[-2], image.shape[-1]])
            predicted_mask = torch.zeros([1, image.shape[-2], image.shape[-1]])
            iou_predicted = torch.zeros_like(iou_predicted)
            score = 0.

        if torch.isnan(iou_predicted):
            iou_predicted = torch.zeros_like(iou_predicted)

        # print(image.shape, target_mask.shape, predicted_mask.shape, iou_predicted)
        if self.return_file_name:
            return {"image":image, 
                "target_mask":target_mask,
                "predicted_mask":predicted_mask,
                "predicted_iou":iou_predicted,
                "score":score,
                "fname":os.path.split(file_path)[-1]}
    
        else:
            return {"image":image, 
                    "target_mask":target_mask,
                    "predicted_mask":predicted_mask,
                    "predicted_iou":iou_predicted}
        

if __name__ == "__main__":
    from transforms import Resize_with_pad
    from torch.utils.data import DataLoader
    ds = iou_prediction_dataset(root = "/data/fast/venkatesh/iou_prediction/DATA/iou_prediction/Train", transform = Resize_with_pad(512, 512))
    dl = DataLoader(ds, batch_size= 32, num_workers=4)

    for i, data in enumerate(dl):
        print(data['predicted_iou'])
        break
