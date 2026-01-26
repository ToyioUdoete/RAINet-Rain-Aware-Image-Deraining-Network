import cv2
import os
import argparse
import numpy as np
import torch
from torch.autograd import Variable
import time

from utils import *
from Network import RAINet
from ptflops import get_model_complexity_info
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


# --------------------------------------------------
# Arguments
# --------------------------------------------------
parser = argparse.ArgumentParser(description="RAINet Test (Single GPU)")

parser.add_argument("--model_dir", type=str, default="./checkpoints/Rain100L_single",
                    help="path to model files")
parser.add_argument("--data_path", type=str,
                    default="/home/magecliff/rainnet/dataset/Rain100L/rain",
                    help="path to testing data")
parser.add_argument("--gt_path", type=str,
                    default="/home/magecliff/rainnet/dataset/Rain100L/norain",
                    help="path to testing gt data")

parser.add_argument("--num_M", type=int, default=32)
parser.add_argument("--num_Z", type=int, default=32)
parser.add_argument("--T", type=int, default=4)
parser.add_argument("--S", type=int, default=20)

parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--save_path", type=str,
                    default="./results/Rain100L_single",
                    help="path to save results")

opt = parser.parse_args()


# --------------------------------------------------
# Setup
# --------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)
device = torch.device("cuda:0")

torch.cuda.empty_cache()
os.makedirs(opt.save_path, exist_ok=True)


# --------------------------------------------------
# Utils
# --------------------------------------------------
def print_network(net):
    num_params = sum(p.numel() for p in net.parameters())
    print("Total number of parameters:", num_params)


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    print("Loading model...")

    model = RAINet(opt).to(device)
    model.eval()

    macs, params = get_model_complexity_info(
        model, (3, 64, 64),
        as_strings=True,
        flops_units="GMac",
        param_units="M",
        print_per_layer_stat=False
    )

    print(f"Computational complexity: {macs}")
    print(f"Number of parameters: {params}")
    print_network(model)

    psnrs_all_epochs = []

    for epoch in range(80, 20, -5):
        torch.cuda.empty_cache()

        epoch_save_dir = os.path.join(opt.save_path, f"epoch_{epoch}")
        os.makedirs(epoch_save_dir, exist_ok=True)

        ckpt_path = os.path.join(opt.model_dir, f"DerainNet_state_{epoch}.pt")
        print(f"\nLoading checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        psnrs = []
        total_time = 0
        count = 0

        with torch.no_grad():
            for img_name in os.listdir(opt.data_path):
                if not is_image(img_name):
                    continue

                img_path = os.path.join(opt.data_path, img_name)
                gt_path = os.path.join(opt.gt_path, img_name)

                O = cv2.imread(img_path)
                gt = cv2.imread(gt_path)

                if O is None or gt is None:
                    print(f"[WARN] Skipping {img_name} (missing file)")
                    continue

                # BGR -> RGB
                b, g, r = cv2.split(O)
                O = cv2.merge([r, g, b])

                O = O.transpose(2, 0, 1)[None, ...]
                O = torch.from_numpy(O).float().to(device)

                torch.cuda.synchronize()
                start = time.time()
                outputs, _ = model(O)
                torch.cuda.synchronize()
                total_time += time.time() - start

                out = torch.clamp(outputs[-1], 0., 255.)
                out = out.cpu().numpy().squeeze().astype(np.uint8)
                out = out.transpose(1, 2, 0)

                # RGB -> BGR
                b, g, r = cv2.split(out)
                out = cv2.merge([r, g, b])

                psnr = compare_psnr(out, gt, data_range=255)
                psnrs.append(psnr)

                cv2.imwrite(os.path.join(epoch_save_dir, img_name), out)
                count += 1

        avg_psnr = sum(psnrs) / len(psnrs)
        avg_time = total_time / max(count, 1)

        print(f"Epoch {epoch} | Avg PSNR: {avg_psnr:.4f} | Avg Time: {avg_time:.4f}s")
        psnrs_all_epochs.append(avg_psnr)

    print("\nMAX PSNR:", max(psnrs_all_epochs),
          "at epoch index", np.argmax(psnrs_all_epochs))


if __name__ == "__main__":
    main()
