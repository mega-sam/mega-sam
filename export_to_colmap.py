import argparse
import numpy as np
import shutil
from pathlib import Path

from colmap_read_model import rotmat2qvec


def export(npz_path: Path, frames_dir: Path, out_dir: Path) -> None:
    data = np.load(npz_path)
    images = data['images']
    K = data['intrinsic']
    cam_c2w = data['cam_c2w']

    h, w = images.shape[1:3]
    out_dir = Path(out_dir)
    sparse_dir = out_dir / 'sparse'
    img_dir = out_dir / 'images'
    sparse_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = []
    for ext in ('*.jpg', '*.png', '*.jpeg'):
        frame_paths.extend(sorted(frames_dir.glob(ext)))
    if len(frame_paths) < cam_c2w.shape[0]:
        raise ValueError('Not enough frames in frames_dir')

    for i in range(cam_c2w.shape[0]):
        dst = img_dir / frame_paths[i].name
        shutil.copy(frame_paths[i], dst)

    with open(sparse_dir / 'cameras.txt', 'w') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        f.write(f'1 PINHOLE {w} {h} {K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]}\n')

    with open(sparse_dir / 'images.txt', 'w') as f:
        f.write('# Image list with two lines of data per image:\n')
        f.write('#   IMAGE_ID, QW QX QY QZ, TX TY TZ, CAMERA_ID, IMAGE_NAME\n')
        f.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
        for i in range(cam_c2w.shape[0]):
            w2c = np.linalg.inv(cam_c2w[i])
            qvec = rotmat2qvec(w2c[:3, :3])
            tvec = w2c[:3, 3]
            fname = frame_paths[i].name
            f.write(f'{i+1} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} '
                    f'{tvec[0]} {tvec[1]} {tvec[2]} 1 {fname}\n')
            f.write('\n')

    with open(sparse_dir / 'points3D.txt', 'w') as f:
        f.write('# Empty point cloud\n')


def main():
    parser = argparse.ArgumentParser(description='Convert MegaSaM output to COLMAP format')
    parser.add_argument('--npz', required=True, type=Path, help='Path to .npz result file')
    parser.add_argument('--frames', required=True, type=Path, help='Directory with extracted frames')
    parser.add_argument('--outdir', required=True, type=Path, help='Destination directory for COLMAP files')
    args = parser.parse_args()
    export(args.npz, args.frames, args.outdir)


if __name__ == '__main__':
    main()


