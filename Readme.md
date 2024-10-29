This repository is for porting the new way of porting the radial expansion coefficients. 


# CPU 
## Compilation 
First load the modules:
```
module load LUMI
module load partition/C
``` 

# GPU

## Compilation 
First load the modules:
```
module load LUMI
module load partition/G
module load rocm
``` 

## Execution

```
srun -p dev-g --gpus-per-node=1  --ntasks-per-node=1 --nodes=1 -c 7     --time=01:00:00 --account=project_462000007 --mem-per-cpu=1750M bin/test.exe
```
