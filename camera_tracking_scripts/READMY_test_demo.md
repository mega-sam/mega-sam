# Guide to `test_demo.py`

This document describes the workflow of `camera_tracking_scripts/test_demo.py`.
It first explains the high level procedure and then dives into important
details, parameters, and implementation nuances.

## High level workflow

1. **Parse command line arguments** – configuration such as data path,
   checkpoint weights, and algorithm thresholds are read from the command
   line.
2. **Gather input data** – the script collects RGB images for a scene
   alongside mono‐depth predictions and metric depth files computed by
   external models.
3. **Prepare calibration and alignment** – intrinsic parameters and a scale
   alignment between mono‐depth and metric depth are computed so that both
   depth sources are consistent.
4. **Stream images to the tracker** – frames are loaded one by one and fed to
   a `Droid` tracker instance along with depth estimates and camera
   intrinsics.
5. **Run camera tracking** – the tracker processes each frame, and the final
   optimisation is triggered at the end of the stream.
6. **Save results** – if a scene name is provided, poses, depths and related
   data are written to disk for later use.

## Detailed steps and parameters

### 1. Argument parsing
`test_demo.py` relies on many options. Important ones include:

- `--datapath`: directory containing input frames.
- `--weights`: checkpoint for the underlying tracking network.
- `--image_size`: resolution to which images are resized before tracking.
- `--beta`, `--filter_thresh`, `--keyframe_thresh`, etc.: various
  thresholds controlling key‑frame selection and optimisation windows.
- `--mono_depth_path` and `--metric_depth_path`: locations of
  pre‑computed mono‑depth and metric depth files.
- `--scene_name`: name used when saving reconstruction outputs.

### 2. Input collection
RGB frames are searched under `args.datapath` with typical image
extensions. Depth predictions are read from `args.mono_depth_path` and
`args.metric_depth_path` for the specified scene. The script expects one
mono inverse‑depth `.npy` per frame and one metric depth `.npz` per
frame, matching the image order.

### 3. Calibration and depth alignment
For each frame, the inverse depths from Depth‑Anything and metric depths
from UniDepth are compared to estimate a scale `align_scale` and shift
`align_shift`. These parameters are used with a normalisation scale to
convert mono inverse depth to metric depth. They also yield a focal
length estimate `ff` derived from the median field‑of‑view reported by
UniDepth. The final 3×3 intrinsic matrix `K` is constructed from `ff` and
image dimensions.

### 4. Image streaming
`image_stream` is a generator that yields one frame at a time. Each image
is resized to keep roughly `384×512` pixels in total and cropped to
multiples of eight. The corresponding depth map is interpolated to the
same size, and an all‑ones mask is produced. If `use_depth` is true,
`image_stream` yields `(t, image, depth, intrinsics, mask)` per frame.
Otherwise depth is omitted.

### 5. Tracking loop
During the main loop, a `Droid` object is created at the first frame.
Every subsequent frame is fed to `droid.track`, passing the current
intrinsics and mask. After the last frame, `droid.track_final` is called
before invoking `droid.terminate` with options `_opt_intr=True` and
`full_ba=True` to refine intrinsics and run bundle adjustment.

### 6. Saving the reconstruction
If `--scene_name` was specified, `save_full_reconstruction` writes the
following to `reconstructions/<scene_name>/`:

- RGB images and disparity maps used for tracking.
- Camera poses in `full_traj` (world‑to‑camera SE3 transforms).
- Per‑frame intrinsics.
- Motion probabilities estimated by the tracker.

A helper `outputs/<scene>_droid.npz` is also produced containing image
arrays, depth maps, the intrinsic matrix and camera pose matrices for the
first thousand frames.

## Nuances

- The script assumes depth predictions are already available. Running
  `Depth-Anything` and `UniDepth` prior to `test_demo.py` is mandatory.
- The intrinsic matrix `K` is derived from field‑of‑view estimates and
  may be refined during optimisation. Providing accurate inputs helps
  convergence.
- Key‑frame and frontend/backend thresholds influence tracking stability
  significantly. Adjust them for challenging scenes.
- If `--disable_vis` is not set, `show_image` will open an interactive
  window displaying each image, which pauses execution until a key is
  pressed.
- The final optimisation with `droid.terminate` may take additional time
  as it runs full bundle adjustment on the collected trajectory.

