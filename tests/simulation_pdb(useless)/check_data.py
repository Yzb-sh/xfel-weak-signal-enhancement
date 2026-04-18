import h5py
import numpy as np

with h5py.File('output/dataset/diffraction_data.h5', 'r') as f:
    print('Dataset info:')
    print(f'  Total samples: {f.attrs["total_samples"]}')
    print(f'  Image size: {f.attrs["image_size"]}')
    
    noisy = f['noisy']['sample_00000'][:]
    clean = f['clean']['sample_00000'][:]
    
    print(f'\nSample 0 stats:')
    print(f'  clean: max={clean.max():.1f}, mean={clean.mean():.1f}, min={clean.min():.1f}')
    print(f'  noisy: max={noisy.max():.1f}, mean={noisy.mean():.1f}, min={noisy.min():.1f}')
    
    print(f'\nSample 0 contrast: {noisy.std() / noisy.mean():.3f}')
    
    center = noisy.shape[0] // 2
    y, x = np.ogrid[:noisy.shape[0], :noisy.shape[1]]
    r = np.sqrt((x - center)**2 + (y - center)**2)
    
    center_intensity = noisy[r <= 5].mean()
    edge_intensity = noisy[(r >= 45) & (r <= 55)].mean()
    
    print(f'  Center intensity: {center_intensity:.1f}')
    print(f'  Edge intensity (r=50): {edge_intensity:.1f}')
    print(f'  Decay ratio: {center_intensity / edge_intensity if edge_intensity > 0 else float("inf"):.1f}')
