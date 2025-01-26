from typing import List
import torch
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import json
import os
from einops import rearrange, repeat
config = OmegaConf.load("configs/latent-diffusion/celeba-vq-4-16-enc-sin.yaml")
model = instantiate_from_config(config.model)
# model.load_state_dict(torch.load("celeba_exp_aba/2023-12-23T14-31-52_celeba-vq-4-16-enc-sin606/checkpoints/last.ckpt")['state_dict'])
model.load_state_dict(torch.load("celeba_exp_aba/2023-12-23T14-32-17_celeba-vq-4-16-enc-sin855/checkpoints/last.ckpt")['state_dict'])
run_files = ["celeba_exp_aba/2023-12-23T14-32-17_celeba-vq-4-16-enc-sin855/checkpoints/last.ckpt"]

new_eval_path = run_files[0]
file = new_eval_path.split("/")[-1]
os.makedirs(new_eval_path.replace(f"checkpoints/{file}","tad_metrics"),exist_ok=True)
sim_logdir = os.path.join(*new_eval_path.split("/")[:-2])
# model.init_from_ckpt(new_eval_path,only_model=False)

model.cuda()


seed = 30 # FIXED AT 30 FOR ALL EXPERIMENTS
import random
random.seed(seed)
import numpy as np
import matplotlib.pyplot as plt
from ae_utils_exp import multi_t, LatentClass, aurocs_search, tags
from torchvision.transforms import Compose

np.random.seed(seed)
torch.manual_seed(seed)
from torchvision.datasets import CelebA
import torchvision.transforms as tforms

tform = tforms.Compose([tforms.Resize(96), tforms.CenterCrop(64), tforms.ToTensor()])

eval_bs = 1000
# set up dataset for eval
# dataset_eval = CelebA(root='../../../guided-diffusion/datasets/CelebA', split='all', target_type='attr', download=False, transform=tform)

# dataloader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=eval_bs, shuffle=True, drop_last=False,num_workers=8)
file = np.load("/home/t-taoyang/new_cloud/2021_4_7/temp/diffusion_disen/latent-diffusion/test_celeba.npz")
data = torch.from_numpy(file['data'])
targ = torch.from_numpy(file['targ'])
au_result, base_rates_raw, targ = aurocs_search(data, targ, model)
base_rates = base_rates_raw.where(base_rates_raw <= 0.5, 1. - base_rates_raw)

# fig, ax = plt.subplots(8, 5, figsize=(16, 16))
# print the ind, tag, max auroc, arg max auroc, norm_diff
max_aur, argmax_aur = torch.max(au_result.clone(), dim=1)
norm_diffs = torch.zeros(40).cuda()
aurs_diffs = torch.zeros(40).cuda()
for ind, tag, max_a, argmax_a, aurs in zip(range(40), tags, max_aur.clone(), argmax_aur.clone(), au_result.clone()):
    norm_aurs = (aurs.clone() - 0.5) / (aurs.clone()[argmax_a] - 0.5)
    aurs_next = aurs.clone()
    aurs_next[argmax_a] = 0.0
    aurs_diff = max_a - aurs_next.max()
    aurs_diffs[ind] = aurs_diff
    norm_aurs[argmax_a] = 0.0
    norm_diff = 1. - norm_aurs.max()
    norm_diffs[ind] = norm_diff
    print("{}\t\t Lat: {}\t Max: {:1.3f}\t ND: {:1.3f}".format(tag, argmax_a.item(), max_a.item(), norm_diff.item()))
    plt_ind = ind//5, ind%5

    assert aurs.max() == max_a



# calculate mutual information shared between attributes
# determine which share a lot of information with each other
with torch.no_grad():
    not_targ = 1 - targ
    j_prob = lambda x, y: torch.logical_and(x, y).sum() / x.numel()
    mi = lambda jp, px, py: 0. if jp == 0. or px == 0. or py == 0. else jp*torch.log(jp/(px*py))

    # Compute the Mutual Information (MI) between the labels
    mi_mat = torch.zeros((40, 40)).cuda()
    for i in range(40):
        # get the marginal of i
        i_mp = targ[:, i].sum() / targ.shape[0]
        for j in range(40):
            j_mp = targ[:, j].sum() / targ.shape[0]
            # get the joint probabilities of FF, FT, TF, TT
            # FF
            jp = j_prob(not_targ[:, i], not_targ[:, j])
            pi = 1. - i_mp
            pj = 1. - j_mp
            mi_mat[i][j] += mi(jp, pi, pj)
            # FT
            jp = j_prob(not_targ[:, i], targ[:, j])
            pi = 1. - i_mp
            pj = j_mp
            mi_mat[i][j] += mi(jp, pi, pj)
            # TF
            jp = j_prob(targ[:, i], not_targ[:, j])
            pi = i_mp
            pj = 1. - j_mp
            mi_mat[i][j] += mi(jp, pi, pj)
            # TT
            jp = j_prob(targ[:, i], targ[:, j])
            pi = i_mp
            pj = j_mp
            mi_mat[i][j] += mi(jp, pi, pj)

    mi_maxes, mi_inds = (mi_mat * (1 - torch.eye(40).cuda())).max(dim=1)
    ent_red_prop = 1. - (mi_mat.diag() - mi_maxes) / mi_mat.diag()
    print(mi_mat.diag())

thresh = 0.75
ent_red_thresh = 0.2

# calculate Average Norm AUROC Diff when best detector score is at a certain threshold
filt = (max_aur >= thresh).logical_and(ent_red_prop <= ent_red_thresh)
# calculate Average Norm AUROC Diff when best detector score is at a certain threshold
aurs_diffs_filt = aurs_diffs[filt]
print(len(aurs_diffs_filt))


print("TAD SCORE: ", aurs_diffs_filt.sum().item(), "Attributes Captured: ", len(aurs_diffs_filt))
save_file = new_eval_path.replace(f"checkpoints","tad_metrics").replace("ckpt","json")
with open(save_file,"w+") as f:
    json.dump({"TAD SCORE: ":aurs_diffs_filt.sum().item(), "Attributes Captured: ":len(aurs_diffs_filt)},f)
