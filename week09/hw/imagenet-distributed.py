import os
import shutil
import random
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchvision.models as models
import torch.distributed as dist
import torch.backends.cudnn as cudnn
#from apex.parallel import DistributedDataParallel as DDP
#from apex import amp
from torch.optim.lr_scheduler import OneCycleLR


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
    
    
IMG_SIZE = 224
LR = 0.1
WEIGHT_DECAY = 3e-5

# https://w251hw05.s3.us-west-1.amazonaws.com/ILSVRC2012_img_val.tar
# https://w251hw05.s3.us-west-1.amazonaws.com/ILSVRC2012_img_train.tar
# python mnist-distributed.py  -n 1 -g 1 -nr 0

imagenet_mean_RGB = [0.47889522, 0.47227842, 0.43047404]
imagenet_std_RGB = [0.229, 0.224, 0.225]

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),# padding=4),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean_RGB, imagenet_std_RGB),
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean_RGB, imagenet_std_RGB),
])

# Data loading code
TRAINDIR='./data/train'
VALDIR = './data/val'
train_dataset = torchvision.datasets.ImageFolder(TRAINDIR, transform=transform_train)     
val_dataset = torchvision.datasets.ImageFolder(VALDIR, transform=transform_val)             
                                  
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
                    
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')                    
                                        
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
                                                
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '34.214.54.102'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(train, nprocs=args.gpus, args=(args,))


import torch.nn.functional as F

def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)
        
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
        
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
        
def save_checkpoint(state, is_best, filename='./checkpoint.pth.tar'):
    # save the model state!
    # state ??? 
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './model.pth.tar')        
        
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'        
        
def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    random.seed(0)
    torch.manual_seed(0)
    
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()
    #model = models.resnet18(pretrained=False)
    
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    train_batch_size = 256
    val_batch_size = 64
    best_acc1 = 0
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    #criterion = LabelSmoothingCrossEntropy(reduction='sum').cuda(gpu)
    
    if args.resume:
        if os.path.isfile(args.resume):
            loc = 'cuda:{}'.format(gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
           
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            
    cudnn.benchmark = True
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=train_batch_size,
                                               shuffle=False,
                                               num_workers=6,
                                               pin_memory=True,
                                               sampler=train_sampler)
                                               
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=val_batch_size,
                                               shuffle=False,
                                               num_workers=6,
                                               pin_memory=True,
                                               sampler=val_sampler)           
                                               
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
    
    # Use Super Convergence                                                                           
    scheduler = OneCycleLR(optimizer, max_lr=1.0, steps_per_epoch=len(train_loader), epochs=args.epochs)
    
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    
    
    #model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    #model = DDP(model)
    
    
    start = datetime.now()
    total_train_step = len(train_loader)
    total_val_step = len(val_loader)
    
    #for epoch in range(args.epochs):

    #########
    #########
    for epoch in range(args.start_epoch, args.epochs):
        #losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        train_sampler.set_epoch(epoch)        
        losses = AverageMeter('Loss', ':.4e')
        progress = ProgressMeter(len(train_loader),[losses, top1, top5], prefix="Epoch: [{}]".format(epoch))
        
    	  #train_sampler.set_epoch(epoch)
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # measure accuracy and record loss
            if gpu == 0:
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

            # Backward and optimize
            optimizer.zero_grad()
            
            #with amp.scale_loss(loss, optimizer) as scaled_loss:
            #    scaled_loss.backward()
                
            loss.backward()
            optimizer.step()
            
            scheduler.step()
            
            if (i + 1) % 100 == 0 and gpu == 0:
                #print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_train_step,loss.item()))
                progress.display(i)
        if(gpu == 0):
            print(' * Train Acc@1 {top1.avg:.3f} Train Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5)) 
            
        model.eval()     
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        losses = AverageMeter('Loss', ':.4e')
        progress = ProgressMeter(len(val_loader),[losses, top1, top5], prefix="Epoch: [{}]".format(epoch))                                                        
        for i, (images, labels) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            
            
            # Forward pass
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            
            
            # measure accuracy and record loss
            if gpu == 0:
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
            
            if (i + 1) % 100 == 0 and gpu == 0:
                #print('Epoch [{}/{}], Step [{}/{}], Val Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_val_step,loss.item()))
                progress.display(i)
                
                
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)    
            
           
            save_checkpoint({
                      'epoch': epoch + 1,
                      'arch': args.arch, 	
                      'state_dict': model.state_dict(),
                      'best_acc1': best_acc1,
                      'optimizer' : optimizer.state_dict(),
            }, is_best)
            
        model.train()
        if gpu == 0:
            print(' * Val Acc@1 {top1.avg:.3f} Val Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
                                                                               
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))


if __name__ == '__main__':
    main()
