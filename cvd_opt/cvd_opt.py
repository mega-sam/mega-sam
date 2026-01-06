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

"""Consistent video depth optimization."""

# pylint: disable=invalid-name
# pylint: disable=g-importing-member
# pylint: disable=redefined-outer-name

import argparse
import os
from pathlib import Path

from geometry_utils import NormalGenerator
import kornia
from lietorch import SE3
import numpy as np
import torch


def gradient_loss(gt, pred, u):
  """Gradient loss."""
  del u
  diff = pred - gt
  v_gradient = torch.abs(
      diff[..., 0:-2, 1:-1] - diff[..., 2:, 1:-1]
  )  # * mask_v
  h_gradient = torch.abs(
      diff[..., 1:-1, 0:-2] - diff[..., 1:-1, 2:]
  )  # * mask_h

  pred_grad = torch.abs(
      pred[..., 0:-2, 1:-1] - (pred[..., 2:, 1:-1])
  ) + torch.abs(pred[..., 1:-1, 0:-2] - pred[..., 1:-1, 2:])
  gt_grad = torch.abs(gt[..., 0:-2, 1:-1] - (gt[..., 2:, 1:-1])) + torch.abs(
      gt[..., 1:-1, 0:-2] - gt[..., 1:-1, 2:]
  )

  grad_diff = torch.abs(pred_grad - gt_grad)
  nearby_mask = (torch.exp(gt[..., 1:-1, 1:-1]) > 1.0).float().detach()
  # weight = (1. - torch.exp(-(grad_diff * 5.)).detach())
  weight = 1.0 - torch.exp(-(grad_diff * 5.0)).detach()
  weight *= nearby_mask

  g_loss = torch.mean(h_gradient * weight) + torch.mean(v_gradient * weight)
  return g_loss


def si_loss(gt, pred):
  log_gt = torch.log(torch.clamp(gt, 1e-3, 1e3)).view(gt.shape[0], -1)
  log_pred = torch.log(torch.clamp(pred, 1e-3, 1e3)).view(pred.shape[0], -1)
  log_diff = log_gt - log_pred
  num_pixels = gt.shape[-2] * gt.shape[-1]
  data_loss = torch.sum(log_diff**2, dim=-1) / num_pixels - torch.sum(
      log_diff, dim=-1
  ) ** 2 / (num_pixels**2)
  return torch.mean(data_loss)


def sobel_fg_alpha(disp, mode="sobel", beta=10.0):
  sobel_grad = kornia.filters.spatial_gradient(
      disp, mode=mode, normalized=False
  )
  sobel_mag = torch.sqrt(
      sobel_grad[:, :, 0, Ellipsis] ** 2 + sobel_grad[:, :, 1, Ellipsis] ** 2
  )
  alpha = torch.exp(-1.0 * beta * sobel_mag).detach()

  return alpha


ALPHA_MOTION = 0.25
RESIZE_FACTOR = 0.5


def consistency_loss(
    cam_c2w,
    K,
    K_inv,
    disp_data,
    init_disp,
    uncertainty,
    flows,
    flow_masks,
    ii,
    jj,
    compute_normals,
    fg_alpha,
    w_ratio=1.0,
    w_flow=0.2,
    w_si=1.0,
    w_grad=2.0,
    w_normal=4.0,
):
  """Consistency loss."""
  _, H, W = disp_data.shape
  # mesh grid
  xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
  yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
  xx = xx.view(1, 1, H, W)  # .repeat(B ,1 ,1 ,1)
  yy = yy.view(1, 1, H, W)  # .repeat(B ,1 ,1 ,1)
  grid = (
      torch.cat((xx, yy), 1).float().cuda().permute(0, 2, 3, 1)
  )  # [None, ...]

  loss_flow = 0.0  # flow reprojection loss
  loss_d_ratio = 0.0  # depth consistency loss

  flows_step = flows.permute(0, 2, 3, 1)
  flow_masks_step = flow_masks.permute(0, 2, 3, 1).squeeze(-1)

  cam_1to2 = torch.bmm(
      torch.linalg.inv(torch.index_select(cam_c2w, dim=0, index=jj)),
      torch.index_select(cam_c2w, dim=0, index=ii),
  )

  # warp disp from target time
  pixel_locations = grid + flows_step
  resize_factor = torch.tensor([W - 1.0, H - 1.0]).cuda()[None, None, None, ...]
  normalized_pixel_locations = 2 * (pixel_locations / resize_factor) - 1.0

  disp_sampled = torch.nn.functional.grid_sample(
      torch.index_select(disp_data, dim=0, index=jj)[:, None, ...],
      normalized_pixel_locations,
      align_corners=True,
  )

  uu = torch.index_select(uncertainty, dim=0, index=ii).squeeze(1)

  grid_h = torch.cat([grid, torch.ones_like(grid[..., 0:1])], dim=-1).unsqueeze(
      -1
  )
  # depth of reference view
  ref_depth = 1.0 / torch.clamp(
      torch.index_select(disp_data, dim=0, index=ii), 1e-3, 1e3
  )

  pts_3d_ref = ref_depth[..., None, None] * (K_inv[None, None, None] @ grid_h)
  rot = cam_1to2[:, None, None, :3, :3]
  trans = cam_1to2[:, None, None, :3, 3:4]

  pts_3d_tgt = (rot @ pts_3d_ref) + trans  # [:, None, None, :, None]
  depth_tgt = pts_3d_tgt[:, :, :, 2:3, 0]
  disp_tgt = 1.0 / torch.clamp(depth_tgt, 0.1, 1e3)

  # flow consistency loss
  pts_2D_tgt = K[None, None, None] @ pts_3d_tgt

  flow_masks_step_ = flow_masks_step * (pts_2D_tgt[:, :, :, 2, 0] > 0.1)
  pts_2D_tgt = pts_2D_tgt[:, :, :, :2, 0] / torch.clamp(
      pts_2D_tgt[:, :, :, 2:, 0], 1e-3, 1e3
  )

  disp_sampled = torch.clamp(disp_sampled, 1e-3, 1e2)
  disp_tgt = torch.clamp(disp_tgt, 1e-3, 1e2)

  ratio = torch.maximum(
      disp_sampled.squeeze() / disp_tgt.squeeze(),
      disp_tgt.squeeze() / disp_sampled.squeeze(),
  )
  ratio_error = torch.abs(ratio - 1.0)  #

  loss_d_ratio += torch.sum(
      (ratio_error * uu + ALPHA_MOTION * torch.log(1.0 / uu)) * flow_masks_step_
  ) / (torch.sum(flow_masks_step_) + 1e-8)

  flow_error = torch.abs(pts_2D_tgt - pixel_locations)
  loss_flow += torch.sum(
      (
          flow_error * uu[..., None]
          + ALPHA_MOTION * torch.log(1.0 / uu[..., None])
      )
      * flow_masks_step_[..., None]
  ) / (torch.sum(flow_masks_step_) * 2.0 + 1e-8)

  # prior mono-depth reg loss
  loss_prior = si_loss(init_disp, disp_data)
  KK = torch.inverse(K_inv)

  # multi gradient consistency
  disp_data_ds = disp_data[:, None, ...]
  init_disp_ds = init_disp[:, None, ...]
  K_rescale = KK.clone()
  K_inv_rescale = torch.inverse(K_rescale)
  pred_normal = compute_normals[0](
      1.0 / torch.clamp(disp_data_ds, 1e-3, 1e3), K_inv_rescale[None]
  )
  init_normal = compute_normals[0](
      1.0 / torch.clamp(init_disp_ds, 1e-3, 1e3), K_inv_rescale[None]
  )

  loss_normal = torch.mean(
      fg_alpha * (1.0 - torch.sum(pred_normal * init_normal, dim=1))
  )  # / (1e-8 + torch.sum(fg_alpha))

  loss_grad = 0.0
  for scale in range(4):
    interval = 2**scale
    disp_data_ds = torch.nn.functional.interpolate(
        disp_data[:, None, ...],
        scale_factor=(1.0 / interval, 1.0 / interval),
        mode="nearest-exact",
    )
    init_disp_ds = torch.nn.functional.interpolate(
        init_disp[:, None, ...],
        scale_factor=(1.0 / interval, 1.0 / interval),
        mode="nearest-exact",
    )
    uncertainty_rs = torch.nn.functional.interpolate(
        uncertainty,
        scale_factor=(1.0 / interval, 1.0 / interval),
        mode="nearest-exact",
    )
    loss_grad += gradient_loss(
        torch.log(disp_data_ds), torch.log(init_disp_ds), uncertainty_rs
    )

  return (
      w_ratio * loss_d_ratio
      + w_si * loss_prior
      + w_flow * loss_flow
      + w_normal * loss_normal
      + loss_grad * w_grad
  )

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--w_grad", type=float, default=2.0, help="w_grad")
  parser.add_argument("--w_normal", type=float, default=6.0, help="w_normal")
  parser.add_argument(
      "--output_dir", type=str, default="outputs_cvd", help="outputs direcotry"
  )
  parser.add_argument("--scene_name", type=str, help="scene name")

  args = parser.parse_args()

  cache_dir = "./cache_flow"
  rootdir = os.getcwd() + "/reconstructions"

  output_dir = args.output_dir
  scene_name = args.scene_name
  print("***************************** ", scene_name)
  img_data = np.load(os.path.join(rootdir, scene_name, "images.npy"))[
      :, ::-1, ...
  ]
  disp_data = (
      np.load(
          os.path.join(rootdir, scene_name.replace("_opt", ""), "disps.npy")
      )
      + 1e-6
  )
  intrinsics = np.load(os.path.join(rootdir, scene_name, "intrinsics.npy"))
  poses = np.load(os.path.join(rootdir, scene_name, "poses.npy"))
  mot_prob = np.load(os.path.join(rootdir, scene_name, "motion_prob.npy"))

  flows = np.load(
      "%s/%s/flows.npy" % (cache_dir, scene_name), allow_pickle=True
  )
  flow_masks = np.load(
      "%s/%s/flows_masks.npy" % (cache_dir, scene_name), allow_pickle=True
  )
  flow_masks = np.float32(flow_masks)
  iijj = np.load("%s/%s/ii-jj.npy" % (cache_dir, scene_name), allow_pickle=True)

  intrinsics = intrinsics[0]
  poses_th = torch.as_tensor(poses, device="cpu").float().cuda()

  K = np.eye(3)
  K[0, 0] = intrinsics[0]
  K[1, 1] = intrinsics[1]
  K[0, 2] = intrinsics[2]
  K[1, 2] = intrinsics[3]

  img_data_pt = (
      torch.from_numpy(np.ascontiguousarray(img_data)).half().cuda() / 255.0
  )
  flows = torch.from_numpy(np.ascontiguousarray(flows)).half().cuda()
  flow_masks = (
      torch.from_numpy(np.ascontiguousarray(flow_masks)).half().cuda()
  )  # .unsqueeze(1)
  iijj = torch.from_numpy(np.ascontiguousarray(iijj)).float().cuda()
  ii = iijj[0, ...].long()
  jj = iijj[1, ...].long()
  K = torch.from_numpy(K).float().cuda()

  init_disp = torch.from_numpy(disp_data).half().cuda()
  disp_data = torch.from_numpy(disp_data).half().cuda()

  assert init_disp.shape == disp_data.shape

  init_disp = torch.nn.functional.interpolate(
      init_disp.unsqueeze(1),
      scale_factor=(RESIZE_FACTOR, RESIZE_FACTOR),
      mode="bilinear",
  ).squeeze(1)
  disp_data = torch.nn.functional.interpolate(
      disp_data.unsqueeze(1),
      scale_factor=(RESIZE_FACTOR, RESIZE_FACTOR),
      mode="bilinear",
  ).squeeze(1)

  fg_alpha = sobel_fg_alpha(init_disp[:, None, ...]) > 0.2
  fg_alpha = fg_alpha.squeeze(1).float() + 0.2

  cvd_prob = torch.nn.functional.interpolate(
      torch.from_numpy(mot_prob).unsqueeze(1).cuda(),
      scale_factor=(4, 4),
      mode="bilinear",
  )
  cvd_prob[cvd_prob > 0.5] = 0.5
  cvd_prob = torch.clamp(cvd_prob, 1e-3, 1.0)

  # rescale intrinsic matrix to small resolution
  K_o = K.clone()
  K[0:2, ...] *= RESIZE_FACTOR
  K_inv = torch.linalg.inv(K)

  disp_data.requires_grad = False
  poses_th.requires_grad = False

  uncertainty = cvd_prob

  # First optimize scale and shift to align them
  log_scale_ = torch.log(torch.ones(init_disp.shape[0]).to(disp_data.device))
  shift_ = torch.zeros(init_disp.shape[0]).to(disp_data.device)
  log_scale_.requires_grad = True
  shift_.requires_grad = True
  uncertainty.requires_grad = True

  optim = torch.optim.Adam([
      {"params": log_scale_, "lr": 1e-2},
      {"params": shift_, "lr": 1e-2},
      {"params": uncertainty, "lr": 1e-2},
  ])

  compute_normals = []
  compute_normals.append(
      NormalGenerator(disp_data.shape[-2], disp_data.shape[-1])
  )
  init_disp = torch.clamp(init_disp, 1e-3, 1e3)

  for i in range(100):
    optim.zero_grad()
    cam_c2w = SE3(poses_th).inv().matrix()
    scale_ = torch.exp(log_scale_)

    loss = consistency_loss(
        cam_c2w,
        K,
        K_inv,
        torch.clamp(
            disp_data * scale_[..., None, None] + shift_[..., None, None],
            1e-3,
            1e3,
        ),
        init_disp,
        torch.clamp(uncertainty, 1e-4, 1e3),
        flows,
        flow_masks,
        ii,
        jj,
        compute_normals,
        fg_alpha,
    )

    loss.backward()
    uncertainty.grad = torch.nan_to_num(uncertainty.grad, nan=0.0)
    log_scale_.grad = torch.nan_to_num(log_scale_.grad, nan=0.0)
    shift_.grad = torch.nan_to_num(shift_.grad, nan=0.0)

    optim.step()
    print("step ", i, loss.item())

  # Then optimize depth and uncertainty
  disp_data = (
      disp_data * torch.exp(log_scale_)[..., None, None].detach()
      + shift_[..., None, None].detach()
  )
  init_disp = (
      init_disp * torch.exp(log_scale_)[..., None, None].detach()
      + shift_[..., None, None].detach()
  )
  init_disp = torch.clamp(init_disp, 1e-3, 1e3)

  disp_data.requires_grad = True
  uncertainty.requires_grad = True
  poses_th.requires_grad = False  # True

  optim = torch.optim.Adam([
      {"params": disp_data, "lr": 5e-3},
      {"params": uncertainty, "lr": 5e-3},
  ])

  losses = []
  for i in range(400):
    optim.zero_grad()
    cam_c2w = SE3(poses_th).inv().matrix()
    loss = consistency_loss(
        cam_c2w,
        K,
        K_inv,
        torch.clamp(disp_data, 1e-3, 1e3),
        init_disp,
        torch.clamp(uncertainty, 1e-4, 1e3),
        flows,
        flow_masks,
        ii,
        jj,
        compute_normals,
        fg_alpha,
        w_ratio=1.0,
        w_flow=0.2,
        w_si=1,
        w_grad=args.w_grad,
        w_normal=args.w_normal,
    )

    loss.backward()
    disp_data.grad = torch.nan_to_num(disp_data.grad, nan=0.0)
    uncertainty.grad = torch.nan_to_num(uncertainty.grad, nan=0.0)

    optim.step()
    print("step ", i, loss.item())
    losses.append(loss)

  disp_data_opt = (
      torch.nn.functional.interpolate(
          disp_data.unsqueeze(1), scale_factor=(2, 2), mode="bilinear"
      )
      .squeeze(1)
      .detach()
      .cpu()
      .numpy()
  )

  # poses_ = poses_th.detach().cpu().numpy()

  Path(output_dir).mkdir(parents=True, exist_ok=True)
  np.savez(
      "%s/%s_sgd_cvd_hr.npz" % (output_dir, scene_name),
      images=np.uint8(img_data_pt.cpu().numpy().transpose(0, 2, 3, 1) * 255.0),
      depths=np.clip(np.float16(1.0 / disp_data_opt), 1e-3, 1e2),
      intrinsic=K_o.detach().cpu().numpy(),
      cam_c2w=cam_c2w.detach().cpu().numpy(),
  )
