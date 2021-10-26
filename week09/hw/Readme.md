# Homework 9 - Distributed training


In this homework, we are focused on aspects of multi-node and multi-gpu (mnmg) model training.
The high level idea is to practice running multi-node training by adapting the code we develop in homework 5 (imagenet training from random weights) to run on two GPU nodes instead of one.

# Summary of Implementation

The implementation is at https://github.com/MadhuAtBerkeley/v3/edit/main/week09/hw

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
The complete log at https://github.com/MadhuAtBerkeley/v3/blob/main/week09/hw/train_log_single_node.txt

## 4.Dual node pytorch DDP program
*  Master - `python imagenet-distributed.py  -n 2 -g 1 -nr 0 --epochs 2` and worker - `python imagenet-distributed.py  -n 2 -g 1 -nr 1 --epochs 2`
*  Performance and timing after two epochs
```
Epoch: [1][2499/2503]	Loss 2.8808e+00 (3.2069e+00)	Acc@1  43.36 ( 33.21)	Acc@5  64.84 ( 57.46)
 * Train Acc@1 33.217 Train Acc@5 57.472
Epoch: [1][299/391]	Loss 2.3966e+00 (2.3767e+00)	Acc@1  43.75 ( 46.46)	Acc@5  71.88 ( 71.93)
 * Val Acc@1 46.780 Val Acc@5 72.148
Training complete in: 0:48:45.460161
```
Complete log at https://github.com/MadhuAtBerkeley/v3/blob/main/week09/hw/train_log_dist_master.txt

## 5.Performance comparison
* Single node took 1:30 hours and Dual node took 0.48 minutes
* It can be seen that each node processed 5005 training batches in single node case while number of batches were split by half - to 2503 in dual node case.
* If trained for 6 epochs, dual node achieved 60% top-1 accuracy in 3 hours 18 minutes

```
Epoch: [5][2499/2503]	Loss 1.9711e+00 (2.0689e+00)	Acc@1  56.64 ( 53.57)	Acc@5  79.30 ( 76.53)
 * Train Acc@1 53.568 Train Acc@5 76.528
Epoch: [5][ 99/391]	Loss 1.3704e+00 (1.6763e+00)	Acc@1  67.19 ( 60.36)	Acc@5  87.50 ( 82.58)
Epoch: [5][199/391]	Loss 1.3637e+00 (1.6890e+00)	Acc@1  75.00 ( 59.87)	Acc@5  85.94 ( 82.58)
Epoch: [5][299/391]	Loss 1.6216e+00 (1.6860e+00)	Acc@1  60.94 ( 60.26)	Acc@5  84.38 ( 82.78)
 * Val Acc@1 60.012 Val Acc@5 82.660
Training complete in: 3:18:19.927112
```

https://github.com/MadhuAtBerkeley/v3/blob/main/week09/hw/train_log_worker.txt

## 6 Tensorboard Visualization
* Following code changes were added
```
writer = SummaryWriter("imagenet_log")
writer.add_scalar('Loss/train', loss.item(), train_log_count)
writer.add_scalar('Accuracy/train', acc1[0], train_log_count)
writer.add_scalar('Loss/Val', loss.item(), val_log_count)      
writer.add_scalar('Accuracy/Val', acc1[0], val_log_count)
writer.flush()
writer.close()

```
* `tensorboard dev upload --logdir 'imagenet_log'` was used to generate tensorboard log and share the results with others.
* The results are at https://tensorboard.dev/experiment/68QRpDH4Reue0RMKb062Tw/#scalars


