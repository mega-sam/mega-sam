import argparse
import asyncio
from pathlib import Path

import numpy as np
import math
import viser


def load_cvd_output(path: Path):
    data = np.load(path)
    images = data["images"]
    depths = data["depths"]
    intrinsic = data["intrinsic"]
    cam_c2w = data["cam_c2w"]
    return images, depths, intrinsic, cam_c2w


def depth_to_pointcloud(depth: np.ndarray, rgb: np.ndarray, K: np.ndarray, c2w: np.ndarray):
    h, w = depth.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    pixels = np.stack([xs, ys, np.ones_like(xs)], axis=-1).reshape(-1, 3)
    invK = np.linalg.inv(K)
    cam = (invK @ pixels.T) * depth.reshape(1, -1)
    cam_h = np.vstack([cam, np.ones((1, cam.shape[1]))])
    world = (c2w @ cam_h)[:3].T
    colors = rgb.reshape(-1, 3) / 255.0
    return world.astype(np.float32), colors.astype(np.float32)


def mat2quat(R: np.ndarray):
    # returns (w,x,y,z)
    q = np.empty(4, dtype=float)
    t = np.trace(R)
    if t > 0:
        s = 0.5 / math.sqrt(t + 1.0)
        q[0] = 0.25 / s
        q[1] = (R[2,1] - R[1,2]) * s
        q[2] = (R[0,2] - R[2,0]) * s
        q[3] = (R[1,0] - R[0,1]) * s
    else:
        # pick the largest diagonal element for stability
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            q[0] = (R[2,1] - R[1,2]) / s
            q[1] = 0.25 * s
            q[2] = (R[0,1] + R[1,0]) / s
            q[3] = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            q[0] = (R[0,2] - R[2,0]) / s
            q[1] = (R[0,1] + R[1,0]) / s
            q[2] = 0.25 * s
            q[3] = (R[1,2] + R[2,1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            q[0] = (R[1,0] - R[0,1]) / s
            q[1] = (R[0,2] + R[2,0]) / s
            q[2] = (R[1,2] + R[2,1]) / s
            q[3] = 0.25 * s
    return tuple(q)

async def run_server(args):
    images, depths, K, cam_c2w = load_cvd_output(args.input)
    n_frames = len(images)

    # pre‑compute pointclouds…
    pcds, cols = zip(*(depth_to_pointcloud(d, im, K, c) 
                       for d, im, c in zip(depths, images, cam_c2w)))

    server = viser.ViserServer()

    # ⚠️ new GUI API:
    slider      = server.gui.add_slider("frame", 0, n_frames-1, initial_value=0, step=1)
    play_button = server.gui.add_button("Play/Pause")
    size_slider = server.gui.add_slider(
        "Point Size",    # label
        0.001,             # min size
        0.02,            # max size
        initial_value=0.005,     # starting size
        step=0.001
    )

    point_cloud = server.scene.add_point_cloud("pc", points=pcds[0], colors=cols[0])

    # compute fov, aspect from your intrinsics & image size:
    h, w = depths[0].shape
    fy = K[1,1]
    fov = 2.0 * math.atan(h / (2 * fy))
    aspect = w / h

    # extract initial quat & pos
    R0 = cam_c2w[0][:3,:3]
    p0 = cam_c2w[0][:3, 3]
    quat0 = mat2quat(R0)
    pos0  = tuple(p0)

    camera = server.scene.add_camera_frustum(
      "camera",
      fov,
      aspect,
      wxyz=quat0,
      position=pos0
    )

    playing = False
    idx = 0

    @slider.on_update
    def _on_slider(evt):
        nonlocal idx
        idx = int(evt.target.value)
        point_cloud.points   = pcds[idx]
        point_cloud.colors   = cols[idx]

        R = cam_c2w[idx][:3,:3]
        p = cam_c2w[idx][:3,3]
        camera.wxyz     = mat2quat(R)
        camera.position = tuple(p)

    @play_button.on_click
    async def _on_play(evt):
        nonlocal playing
        playing = not playing

    @size_slider.on_update
    def _on_size(evt):
        # evt.target.value is your new slider value
        point_cloud.point_size = float(evt.target.value)

    while True:
        await asyncio.sleep(0.1)
        if playing:
            idx = (idx + 1) % n_frames
            slider.value = idx    # triggers on_update for you


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize cvd_opt output")
    parser.add_argument("input", type=Path, help="npz file produced by cvd_opt.py")
    args = parser.parse_args()

    asyncio.run(run_server(args))
