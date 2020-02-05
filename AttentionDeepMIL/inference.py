from __future__ import print_function
import os
import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from skimage.io import imread, imsave

from ckpt import load_ckpt


def run(args):
    # resume checkpoint
    checkpoint = load_ckpt()
    if checkpoint is None:
        print('Failed to load ckpt')
        exit(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = checkpoint.model
    model.eval()
    model = model.to(device)

    imgs = [
        imread(f) for f in sorted(Path(args.img_folder).iterdir())
        if f.is_file()
    ]
    np_image = np.stack(imgs)
    np_image = np.expand_dims(np_image, -1)  # expand to [Batch, H, W, C=1]
    batchsize, h, w, c = np_image.shape
    tensor = [transforms.ToTensor()(np_image[i]) for i in range(batchsize)]
    tensor = torch.stack(tensor, dim=0) if batchsize > 1 else tensor[0]
    if tensor.ndim < 4:
        tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        tensor = tensor.to(device)
        Y_logits = model(tensor)
        print(f'Pred: {Y_logits.detach().argmax(-1).cpu().numpy()}\
                \nLogits:\n{Y_logits.detach().cpu().numpy()}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
    parser.add_argument('--img_folder',
                        type=str,
                        help='specify an image folder to inference',
                        required=True)
    args = parser.parse_args()
    run(args)
