# Homework 9 - Distributed training


In this homework, we are focused on aspects of multi-node and multi-gpu (mnmg) model training.
The high level idea is to practice running multi-node training by adapting the code we develop in homework 5 (imagenet training from random weights) to run on two GPU nodes instead of one.

# Summary of Implementation

## 1. EC2 Instances
* Two g4dn.2xlarge instances - each has 8 vCPUs.
* Deep Learning AMI (Ubuntu 18.04) with 500GB of storage space to work with imagenet2012 downloaded to /data/
* Activate pytorch environment - `conda activate pytorch_latest_p37`
* Download apex - `git clone https://github.com/NVIDIA/apex` and install using `pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./` 
* Install Tensorboard `conda install -c conda-forge tensorboard`

## 2. Pytorch implementation of Imagenet classifier
* Use Apex for AMP ` model, optimizer = amp.initialize(model, optimizer, opt_level='O2')` and `model = apex.parallel.DistributedDataParallel(model)`
* opt_level=O2 casts the model to FP16, keeps batchnorms in FP32, maintains master weights in FP32, and implements dynamic loss scaling by default
* Scaling both forward pass and gradients using 
 `with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()`
* Multiprocessing is handled by `import torch.multiprocessing` and `spawn` is used to share CUDA tensors between processes.

## 3.Single node pytorch DDP program
*  `python imagenet-distributed.py  -n 1 -g 1 -nr 0 --epochs 2`
*  
* You'll need to demonstrate your command of [PyTorch DDP](https://pytorch.org/tutorials/beginner/dist_overview.html)
* Apply [PyTorch native AMP](https://pytorch.org/docs/stable/amp.html)
* Document your run using [Tensorboard](https://www.tensorflow.org/tensorboard) or [Weights and Biases](https://wandb.ai/home) 
* Hopefully, demonstrate that your training is ~2x faster than on a single GPU machine.

Tips:
* You could trt using g4dn.xlarge, but in our experience, they just don't have enough CPUs to keep the GPU fed, so the results will be slow.
* Ideally, you should be able to use EFS.  However, one must ensure that performance is good-- and we've seen issues.
* There is no need to train to the end (e.g. 90 epochs); it would be sufficient to run the training for 1-2 epochs, time it, and compare the results against a run on a sinle GPU instance.
* Please monitor the GPU utilization using nvidia-smi; as long as both GPUs are > 95% utilized, you are doing fine.


