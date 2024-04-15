
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
import os.path as osp

import torch
import torch.nn.functional as F
import clip
import h5py
from PIL import Image
from tqdm import tqdm

model, preprocess = clip.load("ViT-L/14", jit=False)
model.cuda().eval()

image_dir = "data/mm_bench/images"
image_paths = [osp.join(image_dir, p) for p in os.listdir(image_dir)]
images = []
image_ids = []
bs = 0
image_feats = {}

for idx, im_p in enumerate(tqdm(image_paths)):
    image = Image.open(im_p).convert("RGB")
    image = preprocess(image)
    images.append(image)
    image_ids.append(osp.basename(im_p).replace(".png", ""))
    bs += 1

    if bs == 32 or idx == len(image_paths) - 1:
        with torch.no_grad():
            images = torch.stack(images).cuda()
            image_feature = model.encode_image(images).to(torch.float32)
            image_feature = F.normalize(image_feature, dim =1)
            for im_id, feat in zip(image_ids, image_feature):
                image_feats[im_id] = feat.cpu()
        images = []
        image_ids = []
        bs = 0

torch.save(image_feats, "data/mm_bench/object_feats_eva.pt")
