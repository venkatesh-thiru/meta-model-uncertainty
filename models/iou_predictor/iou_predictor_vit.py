import vit_pytorch
import torch
import torch.nn as nn
from vit_pytorch.extractor import Extractor


class iou_predictor_vit(nn.Module):
    def __init__(self, image_size, patch_size, dim = 1024):
        super().__init__()
        self.model = vit_pytorch.ViT(image_size = image_size, patch_size=patch_size, channels=4, num_classes=1, dim = dim, depth = 6, heads=16, mlp_dim=2048)
        self.model = Extractor(self.model)

        self.to_latent = nn.Identity()

        self.linear_head = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.model(x,return_embeddings_only=True)

        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)
    



# if __name__ == "__main__":
#     x = torch.randn([1, 4, 512, 512])

#     model = iou_predictor_vit(image_size=512, patch_size=8)
