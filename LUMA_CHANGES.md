# Luma Production Changes

This branch documents all modifications made to the upstream [mega-sam/mega-sam](https://github.com/mega-sam/mega-sam) repository for Luma's production usage.

## Base Upstream

- **Repo:** `https://github.com/mega-sam/mega-sam`
- **Commit:** `a27b4e6` (pinned in `main` branch)
- **DROID-SLAM submodule:** `ee9ac6a` (unchanged, kept as submodule)

## Changes from Upstream

### 1. CUDA Extension Build Patches (`patches/setup.py`)

The patched `setup.py` replaces `base/setup.py` at compile time. Changes:

- **GPU architecture targets** — sm_70 through sm_90 covering A10 (sm_86, production) up to H100 (sm_90, forward-compat)
- **`CUDA_TARGETS_INCLUDE` path** — CUDA 12+ moves headers (e.g. `cusparse.h`) to `targets/x86_64-linux/include`
- **`nvcc` warning suppression** — `-diag-suppress=20014,177` silences Eigen alignment and unused variable warnings

The patch is applied by a compile script during the build step.

### 2. Pre-compiled CUDA Extensions

DROID-SLAM and lietorch require compiled CUDA extensions for GPU operations (correlation volumes, lie group math). Built and stored per-cluster on internal object storage:

```
megasam/extensions/<cluster>/
├── droid_backends.so
├── lietorch_backends.so
└── abi_manifest.json
```

**ABI manifest** records PyTorch version, CUDA version, glibc version, and Python version at build time. Extensions must be recompiled after a PyTorch version bump.

### 3. Model Checkpoints

Downloaded from the upstream release and stored unmodified on internal object storage:

```
megasam/checkpoints/
├── droid.pth              (DROID-SLAM weights)
├── raft_things.pth        (RAFT optical flow)
└── depth_anything_v2_vitl.pth  (Depth Anything V2)
```

### 4. Cached Model Dependencies

Pre-cached to avoid internet downloads during jobs:

- **DINOv2**: PyTorch Hub cache for `dinov2_vitl14` (used by UniDepth backbone)
- **UniDepth**: V2 ViT-L14 checkpoint + config (metric depth / FOV)

### 5. Repo Tarball

A tarball of this repo (minus checkpoints/extensions) used by the production pipeline at runtime.

## Syncing Upstream

```bash
git fetch upstream
git merge upstream/main
# Re-apply patches/setup.py if base/setup.py changed upstream
```

## Recompiling Extensions

See the compile script in the internal repository for instructions.
