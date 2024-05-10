import numpy as np

a = np.load('/scratch/kys2020/climateML/TaiESM1/1.40625deg_np_20shards/normalize_mean.npz')
b = np.load('/scratch/kys2020/climateML/MPI-ESM/1.40625deg_np_10shards/normalize_mean.npz')
print(a.files)
print()
print(b.files)