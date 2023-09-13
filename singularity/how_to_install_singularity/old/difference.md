export CUDA_VISIBLE_DEVICES=0



## ywata
### nvidia-smi
460.32.03

### echo $CUDA_VISIBLE_DEVICES 

### echo $PATH
/opt/conda/envs/rapids/bin:/opt/conda/bin:/root/.local/bin:/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin                             

### echo $LD_LIBRARY_PATH
/usr/local/cuda/lib64:/usr/local/lib:/usr/local/lib64:/.singularity.d/libs


## 8939
### nvidia-smi
460.56

### echo $CUDA_VISIBLE_DEVICES 
0,1,2,3

### echo $PATH
/opt/conda/envs/rapids/bin:/opt/conda/bin:/root/.local/bin:/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin  

### echo $LD_LIBRARY_PATH
/usr/local/cuda/lib64:/usr/local/lib:/usr/local/lib64:/.singularity.d/libs

### nvcc -V
10.1.243



## ws2
### nvidia-smi
460.32.03

### echo $CUDA_VISIBLE_DEVICES 


### echo $PATH
/opt/conda/envs/rapids/bin:/opt/conda/bin:/root/.local/bin:/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin 
### echo $LD_LIBRARY_PATH

/usr/local/cuda/lib64:/usr/local/lib:/usr/local/lib64:/.singularity.d/libs

### nvcc -V
10.1.243




print(torch.cuda.current_device())
