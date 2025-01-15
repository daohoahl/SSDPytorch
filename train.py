from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import time
import torch
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import numpy as np
import argparse
from clearml import Task

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'], type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT, help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int, help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='Initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for SGD')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='weights/', help='Directory for saving checkpoint models')
args = parser.parse_args()

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor' if args.cuda else 'torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

# Initialize ClearML Task
task = Task.init(project_name="SSD Training", task_name="SSD300 Training with ClearML")
task.connect(args)  # Automatically log all arguments

def train():
    cfg = coco if args.dataset == 'COCO' else voc
    dataset = COCODetection(root=args.dataset_root, transform=SSDAugmentation(cfg['min_dim'], MEANS)) if args.dataset == 'COCO' else VOCDetection(root=args.dataset_root, transform=SSDAugmentation(cfg['min_dim'], MEANS))

    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = torch.nn.DataParallel(ssd_net).cuda() if args.cuda else ssd_net

    if args.resume:
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        ssd_net.vgg.load_state_dict(vgg_weights)

    if not args.resume:
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, args.cuda)

    net.train()
    epoch_size = len(dataset) // args.batch_size
    data_loader = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=detection_collate, pin_memory=True)

    batch_iterator = iter(data_loader)
    loc_loss, conf_loss, epoch = 0, 0, 0

    logger = task.get_logger()  # Get ClearML logger
    num_epochs = 2  # Số epochs bạn muốn huấn luyện
    epoch_size = len(dataset) // args.batch_size
    max_iter = num_epochs * epoch_size

    for iteration in range(args.start_iter, cfg['max_iter']):
        if iteration in cfg['lr_steps']:
            step_index = cfg['lr_steps'].index(iteration)
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # Load training data
        images, targets = next(batch_iterator)
        images = Variable(images.cuda() if args.cuda else images)
        targets = [Variable(ann.cuda() if args.cuda else ann) for ann in targets]

        # Forward pass
        optimizer.zero_grad()
        out = net(images)
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c

        # Backpropagation
        loss.backward()
        optimizer.step()

        loc_loss += loss_l.item()
        conf_loss += loss_c.item()

        # Log scalars to ClearML
        logger.report_scalar("Loss", "Localization Loss", iteration, loss_l.item())
        logger.report_scalar("Loss", "Confidence Loss", iteration, loss_c.item())
        logger.report_scalar("Loss", "Total Loss", iteration, loss.item())
        logger.report_scalar("Learning Rate", "LR", iteration, current_lr)

        # Print training progress
        if iteration % 10 == 0:
            print(f"Iter {iteration} || Loss: {loss.item():.4f} || Loc Loss: {loss_l.item():.4f} || Conf Loss: {loss_c.item():.4f}")

        # Save model periodically
        if iteration % 5000 == 0 and iteration != 0:
            print(f"Saving state, iter: {iteration}")
            torch.save(ssd_net.state_dict(), f"{args.save_folder}/ssd300_{args.dataset}_{iteration}.pth")

    # Save final model
    torch.save(ssd_net.state_dict(), f"{args.save_folder}/{args.dataset}.pth")

def adjust_learning_rate(optimizer, gamma, step):
    lr = args.lr * (gamma ** step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def xavier(param):
    init.xavier_uniform(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

if __name__ == '__main__':
    train()
