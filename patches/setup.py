"""
Patched setup.py for DROID-SLAM CUDA extensions.

This replaces base/setup.py at build time (via compile_megasam_extensions.py).
Changes from upstream:
  - sm_86 (A10, production) + sm_90 (H100, forward-compat)
  - CUDA_TARGETS_INCLUDE path for CUDA 12+ header layout
  - nvcc -diag-suppress=20014,177 to silence Eigen/unused warnings
"""
import os.path as osp

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))

CUDA_TARGETS_INCLUDE = '/usr/local/cuda/targets/x86_64-linux/include'

setup(
    name='droid_backends',
    ext_modules=[
        CUDAExtension(
            'droid_backends',
            include_dirs=[
                osp.join(ROOT, 'thirdparty/eigen'),
                CUDA_TARGETS_INCLUDE,
            ],
            sources=[
                'src/droid.cpp',
                'src/droid_kernels.cu',
                'src/correlation_kernels.cu',
                'src/altcorr_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-diag-suppress=20014',
                    '-diag-suppress=177',
                    '-gencode=arch=compute_70,code=sm_70',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                    '-gencode=arch=compute_90,code=sm_90',
                ],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)

setup(
    name='lietorch',
    version='0.2',
    description='Lie Groups for PyTorch',
    packages=['lietorch'],
    package_dir={'': 'thirdparty/lietorch'},
    ext_modules=[
        CUDAExtension(
            'lietorch_backends',
            include_dirs=[
                osp.join(ROOT, 'thirdparty/lietorch/lietorch/include'),
                osp.join(ROOT, 'thirdparty/eigen'),
                CUDA_TARGETS_INCLUDE,
            ],
            sources=[
                'thirdparty/lietorch/lietorch/src/lietorch.cpp',
                'thirdparty/lietorch/lietorch/src/lietorch_gpu.cu',
                'thirdparty/lietorch/lietorch/src/lietorch_cpu.cpp',
            ],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': [
                    '-O2',
                    '-diag-suppress=20014',
                    '-diag-suppress=177',
                    '-gencode=arch=compute_70,code=sm_70',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                    '-gencode=arch=compute_90,code=sm_90',
                ],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
