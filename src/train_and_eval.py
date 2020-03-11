import functools
import math
from collections import defaultdict

from PIL import Image
import tqdm
import numpy as np
import torch
import glob
import time
from loss import ContentLoss
from metrics import psnr_fn, ssim_fn
from model import ZSSRModel, ZSSRModelWithBackbone
from data import ZSSRDataset, ZSSRSampler, VDSRDataset
from torchvision.transforms import transforms
from torch import nn
from transforms import  ToTensor, RandomCrop
import torch.nn.functional as F
from torch.utils.data import DataLoader
from config import config


def train_single_img(lr_img,
                     model,
                    upsample,
                    data_sampler,
                    trans,
                    optimizer,
                    num_batches,
                    device="cuda"):

    sr_factor = config["sr_factor"] if config is not None else 2
    model.train()
    avg_loss = 0
    loss_type = config["loss_type"] if config is not None else "l1"
    l1_loss = F.l1_loss

    if loss_type == "content" or loss_type == "hybrid":
        content_loss = ContentLoss()
        lr_downsampled = lr_img.resize([lr_img.size[0] // sr_factor, lr_img.size[1] // sr_factor],
                       resample=Image.BICUBIC)
        lr_upsampled = lr_downsampled.resize([lr_downsampled.size[0] * sr_factor, lr_downsampled.size[1] * sr_factor],
                       resample=Image.BICUBIC)
        lr_upsampled = transforms.ToTensor()(lr_upsampled).unsqueeze(0).to(device)
        lr_img_tensor = transforms.ToTensor()(lr_img).unsqueeze(0).to(device)

    for iter, (hr, lr) in enumerate(data_sampler):
        optimizer.zero_grad()
        scale = hr.size[0] // lr.size[0]
        lr = upsample(lr, scale)
        hr, lr = trans((hr, lr))
        hr, lr = hr.unsqueeze(0).to(device), lr.unsqueeze(0).to(device)
        hr_pred = model(lr)
        if loss_type == "content":
            loss = content_loss(model(lr_upsampled), lr_img_tensor)
        elif loss_type == "hybrid":
            loss = config["l1_loss_coff"] * l1_loss(hr_pred, hr)
            loss += config["content_loss_coff"] * content_loss(model(lr_upsampled), lr_img_tensor)
        else:
            loss = l1_loss(hr_pred, hr)


        avg_loss += loss.item()
        loss.backward()
        optimizer.step()

        if iter > num_batches:
            print('Done training.')
            avg_loss /= iter
            print(f'Avg training loss is {avg_loss}')
            break

def test_single_img(model,
                    upsample,
                    downsampled_img,
                    original_img,
                    sr_factor):
    model.eval()
    with torch.set_grad_enabled(False):
        lr_upsampled = upsample(downsampled_img, sr_factor)
        hr_pred = model(lr_upsampled.unsqueeze(0).cuda()).squeeze(0)
        hr_pred = hr_pred.cpu().numpy()
        hr_pred[np.where(hr_pred < 0)] = 0.0
        hr_pred[np.where(hr_pred > 1)] = 1.0
        original_img = np.array(original_img) / 255.0
        original_img = np.transpose(original_img, (2, 0, 1))
        metrics = {}
        if "psnr" in config["metrics"]:
            metrics["psnr"] = psnr_fn(hr_pred, original_img)
        if "ssim" in config["metrics"]:
            metrics["ssim"] = ssim_fn(hr_pred, original_img)
        return metrics

def train_test_single_img(hr_img_path, lr_img_path, config):
    sr_factor = config["sr_factor"]
    img = Image.open(hr_img_path)
    lr_img = Image.open(lr_img_path)
    print(f"Starting training on {hr_img_path} with resolution factor {sr_factor}")
    dataset = ZSSRDataset.from_image(lr_img, config["sr_factor"])
    data_sampler = ZSSRSampler(dataset)
    if config["model"] == "zssr":
        model = ZSSRModel()
    else:
        vdsr_backbone = torch.load("./models/model_epoch_100.pth")
        vdsr_backbone.to(config["device"])
        model = ZSSRModelWithBackbone(vdsr_backbone, True)
    model.to(config["device"])
    model.train()
    all_models = None
    trans = transforms.Compose([
        ToTensor(),
        RandomCrop(config["crop_size"])
    ])
    if config["upsample"] == "pixelshuffle":
        trans = transforms.Compose([
            ToTensor(),
            # RandomCrop(config["crop_size"])
        ])
        upsamples_layers = []
        for _ in range(int(math.log(config["sr_factor"], 2))):
            upsamples_layers += [nn.Conv2d(3, 3 * 4, kernel_size=3, padding=3 // 2, bias=True),
                                 nn.ReLU(),
                                 nn.PixelShuffle(2)]
        upsample_model = nn.Sequential(*upsamples_layers)
        upsample_model.to(config["device"])
        upsample_model.train()
        upsample_fn = functools.partial(pixelshuffle_upsample, upsample_model=upsample_model)
        # we do some of the computation in the upsample phase
        model = ZSSRModel(layers_num=6 - int(math.log(sr_factor)))
        model.to(config["device"])
        model.train()
        all_models = nn.ModuleList([upsample_model, model]).to(config["device"])
    elif config["upsample"] == "cubic":
        all_models = model
        upsample_fn = bicubic_upsample

    optimizer = torch.optim.Adam(all_models.parameters(), lr=config["learning_rate"])

    num_batches = config["number_of_iterations"]

    train_single_img(lr_img,
                     model,
                     upsample_fn,
                     data_sampler,
                     trans,
                     optimizer,
                     num_batches)
    model.eval()
    if config["upsample"] != "cubic":
        upsample_model.eval()

    metrics = test_single_img(model,
                    upsample_fn,
                    lr_img,
                    img,
                    config["sr_factor"])
    return metrics

def pixelshuffle_upsample(lr, scale, upsample_model):
    lr = transforms.ToTensor()(lr).unsqueeze(0)
    lr = lr.cuda()
    lr = upsample_model(lr)
    return lr.squeeze(0)

def bicubic_upsample(lr, scale):
    lr = lr.resize([lr.size[0] * scale, lr.size[1] * scale],
                   resample=Image.BICUBIC)
    return transforms.ToTensor()(lr)

def train_test_all(hr_folder_name, lr_folder_name,  config):
    hr_images = sorted(glob.glob(f"{hr_folder_name}/img_*.png"))
    lr_images = sorted(glob.glob(f"{lr_folder_name}/img_*.png"))

    all_metrics = {}
    for metric_name in config["metrics"]:
        all_metrics[metric_name] = []

    for hr_path, lr_path in zip(hr_images, lr_images):
        metrics = train_test_single_img(hr_path, lr_path, config)
        print(f"Finish evaluation")
        for k, v in metrics.items():
            print(f"Average {k}: {v}")
            all_metrics[k].append(v)
    for k, metric in all_metrics.items():
        avg_metric = sum(metric) / len(metric)
        print(f"Average {k} on all images: {avg_metric}")

if __name__ == "__main__":
    from PIL import Image
    config["sr_factor"] = 2
    config["upsample"] = "cubic"
    config["loss_type"]="content"
    train_test_all("./data/BSD100", "bicubic_kernel_sr2", config)