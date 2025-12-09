import numpy as np

# Load the two files (assumes whitespace-separated, 4 columns each)
cpu = np.loadtxt("soap_cpu.output")
gpu = np.loadtxt("soap_gpu.output")

# Element-wise difference
diff = gpu - cpu

# Report simple quantization of differences
print("Per-column mean difference:", diff.mean(axis=0))
print("Per-column max difference :", diff.max(axis=0))
print("Per-column min difference :", diff.min(axis=0))
print("Per-column RMSE          :", np.sqrt((diff**2).mean(axis=0)))

# Optional: global error
print("Global RMSE:", np.sqrt((diff**2).mean()))
