import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class iou_predictor(nn.Module):
    def __init__(self, model_name, probabilistic=False):
        super(iou_predictor, self).__init__()
        self.model = timm.create_model(model_name, in_chans = 4, pretrained=True,  features_only=True)
        self.nfeatures = self.model.feature_info.channels()[-1]
        self.probabilistic = probabilistic

        self.regressor = nn.Sequential(
                                        nn.Linear(self.nfeatures, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 1),
                                        nn.Sigmoid()
                                        )
        if self.probabilistic:
            self.regressor_std = nn.Sequential(nn.Linear(self.nfeatures, 1024),
                                               nn.ReLU(),
                                               nn.Linear(1024, 1),
                                               nn.Sigmoid())

    def forward(self, x):
        features = self.model(x)[-1]
        features = F.adaptive_avg_pool2d(features, (1, 1)).view(-1, self.nfeatures)
        if self.probabilistic:
            out_mean = self.regressor(features)
            out_var = self.regressor_std(features)
            out_var = torch.where(out_var==0, 1e-6, out_var)
            return out_mean, out_var
            # return torch.distributions.Normal(out_mean, out_sd)
        else:
            out = self.regressor(features)
            return out
