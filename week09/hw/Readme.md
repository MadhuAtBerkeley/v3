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

```
Selected optimization level O2:  FP16 training with FP32 batchnorm and FP32 master weights.  
Defaults for this optimization level are:  
enabled                : True. 
opt_level              : O2. 
cast_model_type        : torch.float16
patch_torch_functions  : False
keep_batchnorm_fp32    : True
master_weights         : True
loss_scale             : dynamic
Processing user overrides (additional kwargs that are not None)...
After processing overrides, optimization options are:
enabled                : True
opt_level              : O2
cast_model_type        : torch.float16
patch_torch_functions  : False
keep_batchnorm_fp32    : True
master_weights         : True
loss_scale             : dynamic
```

* Scaling both forward pass and gradients using 
```
with amp.scale_loss(loss, optimizer) as scaled_loss: 
                scaled_loss.backward()
```
* Multiprocessing is handled by `import torch.multiprocessing` and `spawn` is used to share CUDA tensors between processes.
* Super convergence using - `scheduler = OneCycleLR(optimizer, max_lr=1.0, steps_per_epoch=len(train_loader), epochs=args.epochs)`
* Random sampler for every epoch - `train_sampler.set_epoch(epoch) `
* Batch_size of 256 for training and SGD as optimizer - `optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)`

## 3.Single node pytorch program
*  `python imagenet-distributed.py  -n 1 -g 1 -nr 0 --epochs 2`
*  Performance and timing after two epochs
```
Epoch: [1][4999/5005]	Loss 2.6922e+00 (3.1857e+00)	Acc@1  40.62 ( 33.63)	Acc@5  66.41 ( 57.94). 
Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 1048576.0. 
 * Train Acc@1 33.637 Train Acc@5 57.948. 
Epoch: [1][699/782]	Loss 2.2460e+00 (2.2532e+00)	Acc@1  51.56 ( 49.06)	Acc@5  75.00 ( 73.86). 
 * Val Acc@1 49.090 Val Acc@5 73.862. 
Training complete in: 1:30:40.328945. 
```

## 3.Dual node pytorch DDP program
*  Master - `python imagenet-distributed.py  -n 2 -g 1 -nr 0 --epochs 2` and worker - `python imagenet-distributed.py  -n 2 -g 1 -nr 1 --epochs 2`
*  Performance and timing after two epochs
```
Epoch: [1][2499/2503]	Loss 2.8808e+00 (3.2069e+00)	Acc@1  43.36 ( 33.21)	Acc@5  64.84 ( 57.46)
 * Train Acc@1 33.217 Train Acc@5 57.472
Epoch: [1][299/391]	Loss 2.3966e+00 (2.3767e+00)	Acc@1  43.75 ( 46.46)	Acc@5  71.88 ( 71.93)
 * Val Acc@1 46.780 Val Acc@5 72.148
Training complete in: 0:48:45.460161
```

* You'll need to demonstrate your command of [PyTorch DDP](https://pytorch.org/tutorials/beginner/dist_overview.html)
* Apply [PyTorch native AMP](https://pytorch.org/docs/stable/amp.html)
* Document your run using [Tensorboard](https://www.tensorflow.org/tensorboard) or [Weights and Biases](https://wandb.ai/home) 
* Hopefully, demonstrate that your training is ~2x faster than on a single GPU machine.

Tips:
* You could trt using g4dn.xlarge, but in our experience, they just don't have enough CPUs to keep the GPU fed, so the results will be slow.
* Ideally, you should be able to use EFS.  However, one must ensure that performance is good-- and we've seen issues.
* There is no need to train to the end (e.g. 90 epochs); it would be sufficient to run the training for 1-2 epochs, time it, and compare the results against a run on a sinle GPU instance.
* Please monitor the GPU utilization using nvidia-smi; as long as both GPUs are > 95% utilized, you are doing fine.


