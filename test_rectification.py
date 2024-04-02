"""
Code references:
# >>> https://github.com/cvlab-stonybrook/DewarpNet
# >>> https://github.com/fh2019ustc/DocGeoNet
"""
import argparse
import time

import cv2
import glob
import numpy as np
import os

import torch
import torch.nn.functional as F
import pdf2image

from PIL import Image
from pathlib import Path


from preprocessor.model import DewarpTextlineMaskGuide


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def pdf2imgs(path):
  path = Path(path).expanduser().resolve()
  imgs = pdf2image.convert_from_path(str(path))
  imgs = [np.array(x) for x in imgs] if isinstance(imgs, list) else [imgs]
  return imgs

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default=224, help='image size')
    parser.add_argument('--model_path', type=str, default='weights/rectification/30.pt', help='model path')
    parser.add_argument('--file_path', type=str, default='data/rectification/',
                        help='file path or path to folder containing files')

    parser.add_argument('--save_rectif_path', type=str, default='prediction/rectification/', help='save path')
    parser.add_argument('--save_ocr_path', type=str, default='prediction/ocr/', help='save path')

    return parser.parse_args()


def predict(img_intput, save_path, filename, recti_model):

    if not os.path.exists(save_path): 
        print('Create non-existed Save Path')
        os.makedirs(save_path)

    img_size = parser.input_size

    img = np.array(img_intput)[:, :, :3] / 255.
    img_h, img_w, _ = img.shape
    input_img = cv2.resize(img, (img_size, img_size))

    with torch.no_grad():
        recti_model.eval()
        input_ = torch.from_numpy(input_img).permute(2, 0, 1).cuda()
        input_ = input_.unsqueeze(0)
        start = time.time()

        bm = recti_model(input_.float())
        bm = (2 * (bm / 223.) - 1) * 0.99
        ps_time = time.time() - start

    bm = bm.detach().cpu()
    bm0 = cv2.resize(bm[0, 0].numpy(), (img_w, img_h))  # x flow
    bm1 = cv2.resize(bm[0, 1].numpy(), (img_w, img_h))  # y flow
    bm0 = cv2.blur(bm0, (3, 3))
    bm1 = cv2.blur(bm1, (3, 3))
    lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0).float()  # h * w * 2

    out = F.grid_sample(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float(), lbl, align_corners=True)
    img_geo = ((out[0] * 255.).permute(1, 2, 0).numpy()).astype(np.uint8)

    cv2.imwrite(filename, img_geo[:, :, ::-1])  # save

    return ps_time


if __name__ == '__main__':
    parser = get_args()

    ########
    # Rerctification
    ########
    recti_model = DewarpTextlineMaskGuide(image_size=parser.input_size)
    recti_model = torch.nn.DataParallel(recti_model)
    state_dict = torch.load(parser.model_path, map_location='cpu')

    recti_model.load_state_dict(state_dict)
    recti_model.cuda()
    print('model loaded')

    images = pdf2imgs(parser.file_path)
    save_path = parser.save_rectif_path
    total_time = 0.0

    start = time.time()
    img_num = 0.0
    for idx, image in enumerate(images):  # img_names:  #
        filename = (save_path + f"{idx}.png")
        total_time += predict(image, save_path, filename, recti_model)
        img_num += 1

    print('FPS: %.1f' % (1.0 / (total_time / img_num)))

    ########
    # OCR
    ########

    lsd = cv2.createLineSegmentDetector()
    #
    out_dir = Path(parser.save_ocr_path).expanduser().resolve()
    os.makedirs(out_dir, exist_ok=True)
    #