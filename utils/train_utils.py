"""Training helpers: device selection, logging and baseline factories."""

import logging
import sys

import torch

from dataloader.loader import Antm2cLoader, AmazonLoader
import modules.baseline_models as baseline_models


def getModel(model: str, opt):
    """Build a single-domain baseline model by name."""
    model = model.lower()
    if model == "dnn":
        return baseline_models.DNN(opt)
    elif model == "deepfm":
        return baseline_models.DeepFM(opt)
    elif model == "dcn":
        return baseline_models.DeepCrossNet(opt)
    else:
        raise ValueError("Invalid model type: {}".format(model))


def getOptim(network, optim, lr, l2):
    """Build a single optimizer for a baseline model."""
    params = network.parameters()
    optim = optim.lower()
    if optim == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=l2)
    elif optim == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=l2)
    else:
        raise ValueError("Invalid optimizer type: {}".format(optim))


def getDevice(device_id):
    """Return a CUDA device when ``device_id != -1``, otherwise CPU."""
    if device_id != -1:
        assert torch.cuda.is_available(), "CUDA is not available"
        return torch.device("cuda")
    return torch.device("cpu")


def getDataLoader(dataset: str, path):
    """Build the TFRecord loader for the requested dataset."""
    dataset = dataset.lower()
    if dataset == "antm2c":
        return Antm2cLoader(path)
    elif dataset == "amazon":
        return AmazonLoader(path)
    else:
        raise ValueError("Invalid dataset type: {}".format(dataset))


def get_log(name=""):
    """Create a stdout logger with a compact timestamped format."""
    formatter = logging.Formatter(fmt="[{asctime}]:{message}", style="{")
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
