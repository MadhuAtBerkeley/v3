# Homework 5 - Deep Learning Frameworks - Training an image classifier on the ImageNet dataset from random weights.

This is a graded homework.

Due just before week 6 session

### The goal
The goal of the homework is to train an image classification network on the ImageNet dataset to the Top 1 accuracy of 60% or higher.

We suggest that you use PyTorch or PyTorch Lightning.  



# Summary of Implementation

The implementation is at https://github.com/MadhuAtBerkeley/v3/edit/main/week05/hw

## 1. EC2 Instances
* One g4dn.2xlarge instances with 8 vCPUs.
* Deep Learning AMI (Ubuntu 18.04) with 300GB of storage space to work with imagenet2012 downloaded to /data/
* Activate pytorch environment - `conda activate pytorch_latest_p37`
* Download apex - `git clone https://github.com/NVIDIA/apex` and install using `pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./` 


## 2 Key decisions to consider
* Which architecture to choose? Ans: Resnet18()
* Which optimizer to use? SGD was used `optimizer=torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=3e-5) `
* Super convergence with `scheduler = OneCycleLR(optimizer, max_lr=1.0, steps_per_epoch=len(train_loader), epochs=args.epochs)`.
* Should we change the learning rate while training? Ans:  Super converegence with OneCycleLR() maxLR = 1.0, min_LR = 0.1 and CosineAnnealing


### 3. To turn in
Training logs with achieved the Top 1 accuracy is below.. 60% Top 1 Accuracy in 6 epochs with super convergence
```
Epoch: [5][5000/5005]	Time  1.034 ( 2.526)	Data  0.586 ( 1.589)	Loss 1.839e+00 (2.142e+00)	Acc@1  57.81 ( 52.10)	Acc@5  76.95 ( 75.40)
Test: [700/782]	Time  1.275 ( 0.634)	Loss 1.8844e+00 (1.6804e+00)	Acc@1  57.81 ( 59.99)	Acc@5  81.25 ( 82.78)
 * Acc@1 60.192 Acc@5 82.816
lr: [4.005583818221238e-06]
```
