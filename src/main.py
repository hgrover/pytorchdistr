# Code re-used and adapted from: https://pytorch.org/tutorials/beginner/aws_distributed_training_tutorial.html

import time

import argparse
import os
import pprint
import torch
from tqdm import tqdm

import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from numpy.random import seed

from metrics import AverageMeter, accuracy


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--distributed',
                        dest='distributed',
                        default=False,
                        help='If True, use DDP, else use DP.')

    args = parser.parse_args()
    distributed = args.distributed

    setup(distributed)
    train(distributed)
    cleanup(distributed)


def setup(distributed):
    if distributed:
        print("Initialize Process Group...")
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')

    # For reproducibility
    seed(42)
    torch.manual_seed(42)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train(distributed):
    if distributed:
        local_rank = int(os.environ.get('LOCAL_RANK'))
        device = torch.device(f'cuda:{local_rank}')  # Unique only on individual node.
    else:
        device = torch.device('cuda:0')

    torch.cuda.set_device(device)

    batch_size = 32

    # Number of additional worker processes for dataloading
    workers = 0

    num_epochs = 4
    starting_lr = 0.1

    # Define the transform for the data. Notice, we must resize to 224x224 with this dataset and model.
    transform = transforms.Compose(
        [transforms.Resize(224),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Initialize Datasets. STL10 will automatically download if not present
    trainset = datasets.STL10(root='./stl10_data', split='train', download=True, transform=transform)
    valset = datasets.STL10(root='./stl10_data', split='test', download=True, transform=transform)

    if distributed:
        # Create DistributedSampler to handle distributing the dataset across nodes during training
        # This can only be called after torch.distributed.init_process_group is called
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    else:
        train_sampler = None

    # Create the Dataloaders to feed data to the training and validation steps
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=(train_sampler is None),
                                               num_workers=workers, pin_memory=False, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                             pin_memory=False)

    if distributed:
        print(
            f"IN DEMO BASIC\nWorld size: {os.environ.get('WORLD_SIZE')}, Rank: {os.environ.get('RANK')}, LocalRank: {os.environ.get('LOCAL_RANK')}, Master: {os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')}")

    print("Initialize Model...")
    # Construct Model and wrap into appropriate wrapper
    model = models.resnet18(pretrained=False).cuda()
    if distributed:
        # Wrap model into DistributedDataParallel
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    else:
        # Wrap model into DataParallel, assuming GPU is available
        if torch.cuda.device_count() > 1:
            model = torch.nn.parallel.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), starting_lr, momentum=0.9, weight_decay=1e-4)

    best_prec1 = 0

    for epoch in tqdm(range(num_epochs)):
        pprint.pprint({'message': f'Starting epoch.',
                       'Rank': os.environ.get('RANK'),
                       'epoch': epoch})

        # Adjust learning rate according to schedule
        adjust_learning_rate(starting_lr, optimizer, epoch)

        # train for one epoch
        print("\nBegin Training Epoch {}".format(epoch + 1))
        train_epoch(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        print("Begin Validation @ Epoch {}".format(epoch + 1))
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint if desired
        # is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        print("Epoch Summary: ")
        print("\tEpoch Accuracy: {}".format(prec1))
        print("\tBest Accuracy: {}".format(best_prec1))


def train_epoch(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (ip, target) in enumerate(train_loader):
        pprint.pprint({'message': f'Starting batch.',
                       'Rank': os.environ.get('RANK'),
                       'epoch': epoch,
                       'batch': i})

        # measure data loading time
        data_time.update(time.time() - end)

        # Create non_blocking tensors for distributed training
        # noinspection DuplicatedCode
        ip = ip.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(ip)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), ip.size(0))
        top1.update(prec1[0], ip.size(0))
        top5.update(prec5[0], ip.size(0))

        # compute gradients in a backward pass
        optimizer.zero_grad()
        loss.backward()

        # Call step of optimizer to update model params
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time, data_time=data_time,
                                                                  loss=losses, top1=top1, top5=top5))


def adjust_learning_rate(initial_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (ip, target) in enumerate(val_loader):

            # noinspection DuplicatedCode
            ip = ip.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(ip)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), ip.size(0))
            top1.update(prec1[0], ip.size(0))
            top5.update(prec5[0], ip.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(i, len(val_loader),
                                                                      batch_time=batch_time, loss=losses,
                                                                      top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg


def cleanup(distributed):
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
