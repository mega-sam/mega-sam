# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Test camera tracking on a single scene."""

# pylint: disable=invalid-name
# pylint: disable=g-importing-member
# pylint: disable=g-bad-import-order
# pylint: disable=g-import-not-at-top
# pylint: disable=redefined-outer-name
# pylint: disable=undefined-variable
# pylint: disable=undefined-loop-variable

import sys

sys.path.append("base/droid_slam")

from tqdm import tqdm
import numpy as np
import torch
import cv2
import os
import glob
import argparse
from lietorch import SE3

import torch.nn.functional as F
from droid import Droid

def show_image(img):
  img  = img.numpy()
  c,w,h = img.shape
  img = img.reshape((w,h,c))
  import pdb; pdb.set_trace()
  cv2.imshow("img", img)
  cv2.waitKey()

def image_stream(
    image_list,
    mono_disp_list,
    scene_name,
    use_depth=False,
    aligns=None,
    K=None,
    stride=1,
):
  """image generator."""
  del scene_name, stride

  fx, fy, cx, cy = (
      K[0, 0],
      K[1, 1],
      K[0, 2],
      K[1, 2],
  )  # np.loadtxt(os.path.join(datapath, 'calibration.txt')).tolist()

  for t, (image_file) in enumerate(image_list):
    image = cv2.imread(image_file)
    # depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH) / 5000.
    # depth = np.float32(np.load(depth_file)) / 300.0
    # depth =  1. / pt_data["depth"]

    mono_disp = mono_disp_list[t]
    # mono_disp = np.float32(np.load(disp_file)) #/ 300.0
    depth = np.clip(
        1.0 / ((1.0 / aligns[2]) * (aligns[0] * mono_disp + aligns[1])),
        1e-4,
        1e4,
    )
    depth[depth < 1e-2] = 0.0

    # breakpoint()
    h0, w0, _ = image.shape
    h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
    w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

    image = cv2.resize(image, (w1, h1), interpolation=cv2.INTER_AREA)
    image = image[: h1 - h1 % 8, : w1 - w1 % 8]

    # if t == 4 or t == 29:
    # imageio.imwrite("debug/camel_%d.png"%t, image[..., ::-1])

    image = torch.as_tensor(image).permute(2, 0, 1)
    # print("image ", image.shape)
    # breakpoint()

    depth = torch.as_tensor(depth)
    depth = F.interpolate(
        depth[None, None], (h1, w1), mode="nearest-exact"
    ).squeeze()
    depth = depth[: h1 - h1 % 8, : w1 - w1 % 8]

    mask = torch.ones_like(depth)

    intrinsics = torch.as_tensor([fx, fy, cx, cy])
    intrinsics[0::2] *= w1 / w0
    intrinsics[1::2] *= h1 / h0

    if use_depth:
      yield t, image[None], depth, intrinsics, mask
    else:
      yield t, image[None], intrinsics, mask


def save_full_reconstruction(
    droid, full_traj, rgb_list, senor_depth_list, motion_prob, scene_name
):
  """Save full reconstruction."""
  from pathlib import Path
  t = full_traj.shape[0]
  images = np.array(rgb_list[:t])  # droid.video.images[:t].cpu().numpy()
  disps = 1.0 / (np.array(senor_depth_list[:t]) + 1e-6)

  poses = full_traj  # .cpu().numpy()
  intrinsics = droid.video.intrinsics[:t].cpu().numpy()

  Path("reconstructions/{}".format(scene_name)).mkdir(
      parents=True, exist_ok=True
  )
  np.save("reconstructions/{}/images.npy".format(scene_name), images)
  np.save("reconstructions/{}/disps.npy".format(scene_name), disps)
  np.save("reconstructions/{}/poses.npy".format(scene_name), poses)
  np.save(
      "reconstructions/{}/intrinsics.npy".format(scene_name), intrinsics * 8.0
  )
  np.save("reconstructions/{}/motion_prob.npy".format(scene_name), motion_prob)

  intrinsics = intrinsics[0] * 8.0
  poses_th = torch.as_tensor(poses, device="cpu")
  cam_c2w = SE3(poses_th).inv().matrix().numpy()

  K = np.eye(3)
  K[0, 0] = intrinsics[0]
  K[1, 1] = intrinsics[1]
  K[0, 2] = intrinsics[2]
  K[1, 2] = intrinsics[3]
  print("K ", K)
  print("img_data ", images.shape)
  print("disp_data ", disps.shape)

  max_frames = min(1000, images.shape[0])
  print("outputs/%s_droid.npz" % scene_name)
  Path("outputs").mkdir(parents=True, exist_ok=True)

  np.savez(
      "outputs/%s_droid.npz" % scene_name,
      images=np.uint8(images[:max_frames, ::-1, ...].transpose(0, 2, 3, 1)),
      depths=np.float32(1.0 / disps[:max_frames, ...]),
      intrinsic=K,
      cam_c2w=cam_c2w[:max_frames],
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--datapath")
  parser.add_argument("--weights", default="droid.pth")
  parser.add_argument("--buffer", type=int, default=1024)
  parser.add_argument("--image_size", default=[240, 320])
  parser.add_argument("--disable_vis", action="store_true")

  parser.add_argument("--beta", type=float, default=0.3)
  parser.add_argument(
      "--filter_thresh", type=float, default=2.0
  )  # motion threhold for keyframe
  parser.add_argument("--warmup", type=int, default=8)
  parser.add_argument("--keyframe_thresh", type=float, default=2.0)
  parser.add_argument("--frontend_thresh", type=float, default=12.0)
  parser.add_argument("--frontend_window", type=int, default=25)
  parser.add_argument("--frontend_radius", type=int, default=2)
  parser.add_argument("--frontend_nms", type=int, default=1)

  parser.add_argument("--stereo", action="store_true")
  parser.add_argument("--depth", action="store_true")
  parser.add_argument("--upsample", action="store_true")
  parser.add_argument("--scene_name", help="scene_name")

  parser.add_argument("--backend_thresh", type=float, default=16.0)
  parser.add_argument("--backend_radius", type=int, default=2)
  parser.add_argument("--backend_nms", type=int, default=3)

  parser.add_argument(
      "--mono_depth_path", default="Depth-Anything/video_visualization"
  )
  parser.add_argument("--metric_depth_path", default="UniDepth/outputs ")
  args = parser.parse_args()

  print("Running evaluation on {}".format(args.datapath))
  print(args)

  scene_name = args.scene_name.split("/")[-1]

  tstamps = []
  rgb_list = []
  senor_depth_list = []

  image_list = sorted(glob.glob(os.path.join("%s" % (args.datapath), "*.jpg")))
  image_list += sorted(glob.glob(os.path.join("%s" % (args.datapath), "*.png")))
  image_list += sorted(glob.glob(os.path.join("%s" % (args.datapath), "*.jpeg")))

  # NOTE Mono is inverse depth, but metric-depth is depth!
  mono_disp_paths = sorted(
      glob.glob(
          os.path.join("%s/%s" % (args.mono_depth_path, scene_name), "*.npy")
      )
  )
  metric_depth_paths = sorted(
      glob.glob(
          os.path.join("%s/%s" % (args.metric_depth_path, scene_name), "*.npz")
      )
  )

  img_0 = cv2.imread(image_list[0])
  scales = []
  shifts = []
  mono_disp_list = []
  fovs = []
  for t, (mono_disp_file, metric_depth_file) in enumerate(
      zip(mono_disp_paths, metric_depth_paths)
  ):
    da_disp = np.float32(np.load(mono_disp_file))  # / 300.0
    uni_data = np.load(metric_depth_file)
    metric_depth = uni_data["depth"]

    fovs.append(uni_data["fov"])

    da_disp = cv2.resize(
        da_disp,
        (metric_depth.shape[1], metric_depth.shape[0]),
        interpolation=cv2.INTER_NEAREST_EXACT,
    )
    mono_disp_list.append(da_disp)
    gt_disp = 1.0 / (metric_depth + 1e-8)

    # avoid some bug from UniDepth
    valid_mask = (metric_depth < 2.0) & (da_disp < 0.02)
    gt_disp[valid_mask] = 1e-2

    # avoid cases sky dominate entire video
    sky_ratio = np.sum(da_disp < 0.01) / (da_disp.shape[0] * da_disp.shape[1])
    if sky_ratio > 0.5:
      non_sky_mask = da_disp > 0.01
      gt_disp_ms = (
          gt_disp[non_sky_mask] - np.median(gt_disp[non_sky_mask]) + 1e-8
      )
      da_disp_ms = (
          da_disp[non_sky_mask] - np.median(da_disp[non_sky_mask]) + 1e-8
      )
      scale = np.median(gt_disp_ms / da_disp_ms)
      shift = np.median(gt_disp[non_sky_mask] - scale * da_disp[non_sky_mask])
    else:
      gt_disp_ms = gt_disp - np.median(gt_disp) + 1e-8
      da_disp_ms = da_disp - np.median(da_disp) + 1e-8
      scale = np.median(gt_disp_ms / da_disp_ms)
      shift = np.median(gt_disp - scale * da_disp)

    gt_disp_ms = gt_disp - np.median(gt_disp) + 1e-8
    da_disp_ms = da_disp - np.median(da_disp) + 1e-8

    scale = np.median(gt_disp_ms / da_disp_ms)
    shift = np.median(gt_disp - scale * da_disp)

    scales.append(scale)
    shifts.append(shift)

  print("************** UNIDEPTH FOV ", np.median(fovs))
  ff = img_0.shape[1] / (2 * np.tan(np.radians(np.median(fovs) / 2.0)))
  K = np.eye(3)
  K[0, 0] = (
      ff * 1.0
  )  # pp_intrinsic[0]  * (img_0.shape[1] / (pp_intrinsic[1] * 2))
  K[1, 1] = (
      ff * 1.0
  )  # pp_intrinsic[0]  * (img_0.shape[0] / (pp_intrinsic[2] * 2))
  K[0, 2] = (
      img_0.shape[1] / 2.0
  )  # pp_intrinsic[1]) * (img_0.shape[1] / (pp_intrinsic[1] * 2))
  K[1, 2] = (
      img_0.shape[0] / 2.0
  )  # (pp_intrinsic[2]) * (img_0.shape[0] / (pp_intrinsic[2] * 2))

  ss_product = np.array(scales) * np.array(shifts)
  med_idx = np.argmin(np.abs(ss_product - np.median(ss_product)))

  align_scale = scales[med_idx]  # np.median(np.array(scales))
  align_shift = shifts[med_idx]  # np.median(np.array(shifts))
  normalize_scale = (
      np.percentile((align_scale * np.array(mono_disp_list) + align_shift), 98)
      / 2.0
  )

  aligns = (align_scale, align_shift, normalize_scale)

  for t, image, depth, intrinsics, mask in tqdm(
      image_stream(
          image_list,
          mono_disp_list,
          scene_name,
          use_depth=True,
          aligns=aligns,
          K=K,
      )
  ):
    # if not args.disable_vis:
    #   show_image(image[0])

    rgb_list.append(image[0])
    senor_depth_list.append(depth)
    # breakpoint()
    if t == 0:
      args.image_size = [image.shape[2], image.shape[3]]
      droid = Droid(args)

    droid.track(t, image, depth, intrinsics=intrinsics, mask=mask)

  # last frame
  droid.track_final(t, image, depth, intrinsics=intrinsics, mask=mask)

  traj_est, depth_est, motion_prob = droid.terminate(
      image_stream(
          image_list,
          mono_disp_list,
          scene_name,
          use_depth=True,
          aligns=aligns,
          K=K,
      ),
      _opt_intr=True,
      full_ba=True,
      scene_name=scene_name,
  )

  if args.scene_name is not None:
    save_full_reconstruction(
        droid,
        traj_est,
        rgb_list,
        senor_depth_list,
        motion_prob,
        args.scene_name,
    )
