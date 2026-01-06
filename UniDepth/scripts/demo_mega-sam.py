import argparse
import glob
import os

import cv2
import imageio
import numpy as np
from PIL import Image
import torch
import tqdm
from unidepth.models import UniDepthV2
from unidepth.utils import colorize, image_grid

LONG_DIM = 640

def demo(model, args):
  outdir = args.outdir  # "./outputs"
  # os.makedirs(outdir, exist_ok=True)

  # for scene_name in scene_names:
  scene_name = args.scene_name
  outdir_scene = os.path.join(outdir, scene_name)
  os.makedirs(outdir_scene, exist_ok=True)
  # img_path_list = sorted(glob.glob("/home/zhengqili/filestore/DAVIS/DAVIS/JPEGImages/480p/%s/*.jpg"%scene_name))
  img_path_list = sorted(glob.glob(os.path.join(args.img_path, "*.jpg")))
  img_path_list += sorted(glob.glob(os.path.join(args.img_path, "*.png")))
  img_path_list += sorted(glob.glob(os.path.join(args.img_path, "*.jpeg")))

  fovs = []
  for img_path in tqdm.tqdm(img_path_list):
    rgb = np.array(Image.open(img_path))[..., :3]
    if rgb.shape[1] > rgb.shape[0]:
      final_w, final_h = LONG_DIM, int(
          round(LONG_DIM * rgb.shape[0] / rgb.shape[1])
      )
    else:
      final_w, final_h = (
          int(round(LONG_DIM * rgb.shape[1] / rgb.shape[0])),
          LONG_DIM,
      )
    rgb = cv2.resize(
        rgb, (final_w, final_h), cv2.INTER_AREA
    )  # .transpose(2, 0, 1)

    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
    # intrinsics_torch = torch.from_numpy(np.load("assets/demo/intrinsics.npy"))
    # predict
    predictions = model.infer(rgb_torch)
    fov_ = np.rad2deg(
        2
        * np.arctan(
            predictions["depth"].shape[-1]
            / (2 * predictions["intrinsics"][0, 0, 0].cpu().numpy())
        )
    )
    depth = predictions["depth"][0, 0].cpu().numpy()
    print(fov_)
    fovs.append(fov_)
    # breakpoint()
    np.savez(
        os.path.join(outdir_scene, img_path.split("/")[-1][:-4] + ".npz"),
        depth=np.float32(depth),
        fov=fov_,
    )


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--img-path", type=str)
  parser.add_argument("--outdir", type=str, default="./vis_depth")
  parser.add_argument("--scene-name", type=str)

  args = parser.parse_args()

  print("Torch version:", torch.__version__)
  # model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")
  # model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
  model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14", revision="1d0d3c52f60b5164629d279bb9a7546458e6dcc4")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  demo(model, args)
