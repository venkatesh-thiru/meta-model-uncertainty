import math

import numpy as np
import torch
from torch.nn import init

_LOG_2PI = math.log(2 * math.pi)
_LOG_PI = math.log(math.pi)

def gaussian_log_likelihood_loss(mean, var, target):

    ll = -0.5 * ((target - mean) ** 2 / var + torch.log(var) + _LOG_2PI)

    return -torch.sum(ll, axis=0)

def gaussian_beta_log_likelihood_loss(mean, var, target, beta=1):
    ll = -0.5 * ((target - mean) ** 2 / var + torch.log(var) + _LOG_2PI)
    weight = var.detach() ** beta

    return -torch.sum(ll * weight, axis=0)