import numpy as np
from collections import defaultdict

def check_matching(cpu_site, gpu_site, label):
    if not np.array_equal(cpu_site, gpu_site):
        raise ValueError(f"Site indices do not match for {label}.")

def per_site_mean_diff(cpu_site, diff):
    groups = defaultdict(list)
    for site, d in zip(cpu_site, diff):
        groups[site].append(d)

    site_indices = np.array(sorted(groups.keys()))
    site_means = np.vstack([
        np.mean(np.vstack(groups[s]), axis=0) for s in site_indices
    ])
    return site_indices, site_means


# ======================
# Energies
# ======================
cpu = np.loadtxt("cpu_energies.output")
gpu = np.loadtxt("gpu_energies.output")

cpu_site = cpu[:, 0].astype(int)
gpu_site = gpu[:, 0].astype(int)

cpu_E = cpu[:, 1:2]
gpu_E = gpu[:, 1:2]

check_matching(cpu_site, gpu_site, "energy")

diff_E = np.abs(gpu_E - cpu_E)
sites_E, site_mean_E = per_site_mean_diff(cpu_site, diff_E)

np.savetxt(
    "per_site_mean_diff_energy.txt",
    np.column_stack([sites_E, site_mean_E]),
    header="Site  |E_gpu - E_cpu|",
    fmt="%d %.6e"
)

print("Energy overall mean diff:", site_mean_E.mean(axis=0)[0])


# ======================
# Forces (Fx, Fy, Fz)
# ======================
cpu = np.loadtxt("cpu_forces.output")
gpu = np.loadtxt("gpu_forces.output")

cpu_site = cpu[:, 0].astype(int)
gpu_site = gpu[:, 0].astype(int)

cpu_F = cpu[:, 1:4]
gpu_F = gpu[:, 1:4]

check_matching(cpu_site, gpu_site, "forces")

diff_F = np.abs(gpu_F - cpu_F)
sites_F, site_mean_F = per_site_mean_diff(cpu_site, diff_F)

np.savetxt(
    "per_site_mean_diff_force.txt",
    np.column_stack([sites_F, site_mean_F]),
    header="Site  Fx  Fy  Fz",
    fmt="%d %.6e %.6e %.6e"
)

print("Force overall mean diff [Fx Fy Fz]:")
print(site_mean_F.mean(axis=0))


# ======================
# Virial (3x3 tensor)
# ======================
cpu = np.loadtxt("cpu_virial.output")
gpu = np.loadtxt("gpu_virial.output")

cpu_site = cpu[:, 0].astype(int)
gpu_site = gpu[:, 0].astype(int)

cpu_V = cpu[:, 1:4]
gpu_V = gpu[:, 1:4]

check_matching(cpu_site, gpu_site, "virial")

diff_V = np.abs(gpu_V - cpu_V)

sites_V, site_mean_V = per_site_mean_diff(cpu_site, diff_V)

# site_mean_V has shape (nsite, 3) but represents row-wise averages;
# reshape to full 3x3 per site if you want tensors
np.savetxt(
    "per_site_mean_diff_virial.txt",
    np.column_stack([sites_V, site_mean_V]),
    header="v1  v2  v3   (row-wise mean abs diff)",
    fmt="%.6e %.6e %.6e"
)

print("Virial overall mean diff (row-wise):")
print(site_mean_V.mean(axis=0))


# ======================
# NaN checks
# ======================
print("\nNaN checks:")
print("Energy CPU:", np.isnan(cpu_E).any(), "GPU:", np.isnan(gpu_E).any())
print("Force  CPU:", np.isnan(cpu_F).any(), "GPU:", np.isnan(gpu_F).any())
print("Virial CPU:", np.isnan(cpu_V).any(), "GPU:", np.isnan(gpu_V).any())
