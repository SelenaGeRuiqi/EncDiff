from typing import List
import torch
import os
from torch.nn import MSELoss
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from einops import rearrange, repeat
from lfw_src.lfw_attribute import LFWAttribute
from torch.utils.data import Subset, DataLoader
from torchvision import transforms
import torch.nn as nn
from torch.optim import SGD, Adam
from tqdm import tqdm
from lfw_src.eval_utils import eval_regression
# config = OmegaConf.load("configs/latent-diffusion/shapes3d-vq-4-16-enc-sin.yaml")
config = OmegaConf.load("configs/latent-diffusion/celeba-vq-4-16-enc.yaml")
model = instantiate_from_config(config.model.params.cond_stage_config)
# op_path = "shapes3d_exp_aba/2023-12-20T15-15-04_shapes3d-vq-4-16-enc-sin363"
# op_path = "celeba_exp_aba/2023-12-23T14-37-03_celeba-vq-4-16-enc-sin779"
# constructing dataloaders
size = 64
train_transform = test_transform = transforms.Compose([
    transforms.Resize(int(size * 1.1)),
    transforms.CenterCrop(size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,), inplace=True)
])
# default funnel set is already loose crop
dataset_root = "../../diti/data/lfw"
train_set = LFWAttribute("train", dataset_root, split='train', transform=train_transform, download=True)
test_set = LFWAttribute("data", dataset_root, split='test', transform=test_transform, download=True)

train_loader = DataLoader(
    dataset=train_set,
    pin_memory=False,
    num_workers=4,
    batch_size=64,
    shuffle=True,
)
test_loader = DataLoader(
    dataset=test_set,
    pin_memory=False,
    num_workers=4,
    batch_size=64
)
# define loss function
loss_fn = MSELoss()
for op_p in os.listdir("celeba_exp"):
    op_path = os.path.join("celeba_exp",op_p)
    ckpt = torch.load(f"{op_path}/checkpoints/last.ckpt")['state_dict']
    new_ckpt = {}
    for k in ckpt.keys():
        if "cond_stage_model" in k:
            new_ckpt[k.replace("cond_stage_model.","")] = ckpt[k]
    u,m = model.load_state_dict(new_ckpt)
    print(u)
    print(m)
    # prepare logging directory
    log_dir_name = "regression"
    log_dir = os.path.join(op_path, log_dir_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    classifier = nn.Linear(model.latent_dim, train_set.num_attributes).cuda()
    optimizer = Adam(classifier.parameters(), lr=0.001)

    import numpy as np
    import torch
    model.cuda()
    model.eval()

    # training
    best_r = 0.0
    step = 0
    test_results = []
    test_mse = []
    epochs = 15
    for epoch in range(epochs):
        classifier.train()
        model.eval()
        pbar = tqdm(train_loader)
        # imgs_list = []
        # y_gt = []

        for i, batch in enumerate(pbar):
            optimizer.zero_grad()
            imgs = batch[0].cuda()
            labels = batch[2].cuda().float()
            # imgs_list.append(imgs.detach().cpu())
            # y_gt.append(labels.detach().cpu())
            
            z = model(imgs)
            preds = classifier(z)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            # writer.add_scalar("multitask/loss", loss, step)
            pbar.set_description(f"Step {step} loss {loss:.3f}")
            step += 1

        # images_gt = torch.cat(imgs_list, dim=0).numpy()
        # y_gt = torch.cat(y_gt, dim=0).numpy()

        # np.savez("/home/t-taoyang/cloud/2021_4_7/temp/diti/data/lfw/lfw-py/lfw_train.npz",label=y_gt, images=images_gt)
        pearson_r, mse_per_attribute = eval_regression(test_loader, model, classifier)
        avg_r = sum(pearson_r)/len(pearson_r)
        avg_mse = mse_per_attribute.mean()
        test_results.append(pearson_r)
        test_mse.append(mse_per_attribute)
        print(f"Epoch {epoch} test avg pearson r: {avg_r:.3f}; avg MSE: {avg_mse:.3f}")
        if avg_r > best_r:
            print(f"New best @Epoch {epoch} with val ap: {avg_r:.3f}")
            best_r = avg_r
            torch.save(classifier.state_dict(), os.path.join(log_dir, "best.pt"))

    # Write results
    with open(os.path.join(log_dir, "lfw2.txt"), 'a') as file:
        for epoch, result in enumerate(test_results):
            file.write(f"Test pearson r @Epoch{epoch}: {sum(result) / len(result):.3f}.\n")
            file.write(', '.join([str(r) for r in result]))
            file.write(f"Test MSE @Epoch{epoch}: {test_mse[epoch].mean()}.\n")
            file.write(', '.join([str(mse) for mse in test_mse[epoch]]))
            file.write('\n\n')