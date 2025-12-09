import numpy as np
from collections import defaultdict

# Load data
cpu = np.loadtxt("soap_cpu.output")
gpu = np.loadtxt("soap_gpu.output")

# Extract columns
cpu_site  = cpu[:, 0].astype(int)
gpu_site  = gpu[:, 0].astype(int)

cpu_vals = cpu[:, 1:]   # 4 columns
gpu_vals = gpu[:, 1:]   # 4 columns

# Check matching
if not np.array_equal(cpu_site, gpu_site):
    raise ValueError("Site indices do not match between CPU and GPU files.")

# Absolute differences
diff = np.abs(gpu_vals - cpu_vals)

# Group diffs by site
groups = defaultdict(list)
for site, d in zip(cpu_site, diff):
    groups[site].append(d)

# Compute per-site averages safely
per_site_mean = {}
for site, rows in groups.items():
    if rows:  # only compute if there are rows
        per_site_mean[site] = np.mean(np.vstack(rows), axis=0)

# Convert to array sorted by site index
site_indices = np.array(sorted(per_site_mean.keys()))
site_means   = np.vstack([per_site_mean[s] for s in site_indices])

# Save per-site means to file
np.savetxt("per_site_mean_diff.txt", 
           np.column_stack([site_indices, site_means]),
           header="Site  Col1  Col2  Col3  Col4", fmt="%d %.6e %.6e %.6e %.6e")

# Final overall average
overall_mean = site_means.mean(axis=0)

# Output only overall mean
print("Overall mean across all sites:")
print(overall_mean)

# Optional: check for NaNs
print("NaNs in CPU values:", np.isnan(cpu_vals).any())
print("NaNs in GPU values:", np.isnan(gpu_vals).any())
print("NaNs in diff:", np.isnan(diff).any())
