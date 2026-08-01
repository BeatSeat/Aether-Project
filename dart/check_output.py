import pickle
import numpy as np

pkl_path = '/root/DART/mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000/rollout/wave_hello*5_guidance5.0_seed0/sample_0.pkl'
npz_path = '/root/DART/mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000/rollout/wave_hello*5_guidance5.0_seed0/sample_0_smplx.npz'

print('=== sample_0.pkl ===')
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print('Keys:', list(data.keys()))
for key, val in data.items():
    if hasattr(val, 'shape'):
        print(f'  {key}: shape={val.shape}, dtype={val.dtype}')
    elif isinstance(val, str):
        print(f'  {key}: str = "{val}"')
    elif isinstance(val, (int, float, bool)):
        print(f'  {key}: {type(val).__name__} = {val}')
    else:
        print(f'  {key}: {type(val).__name__}')

print()
print('=== sample_0_smplx.npz ===')
npz = np.load(npz_path, allow_pickle=True)
print('Keys:', list(npz.keys()))
for key in npz.keys():
    arr = npz[key]
    print(f'  {key}: shape={arr.shape}, dtype={arr.dtype}')
    if arr.ndim == 0:
        print(f'    value: {arr.item()}')
    elif arr.ndim == 1 and arr.shape[0] <= 20:
        print(f'    values: {arr}')

print()
print('=== Analysis ===')
poses = npz['poses']
print(f'poses shape: {poses.shape}')
print(f'  Total pose params per frame: {poses.shape[1]}')
print(f'  Body (22 joints * 3): {22*3} = first 66 values')
print(f'  Hand+face padding: {poses.shape[1] - 66} values')

# Check if hand/face portion is all zeros
hand_face = poses[:, 66:]
print(f'  Hand+face portion all zeros: {np.allclose(hand_face, 0)}')
print(f'  Hand+face max abs value: {np.abs(hand_face).max()}')
