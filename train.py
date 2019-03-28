import matplotlib
matplotlib.use('pdf')
import sys
import os
import logging
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import datetime
from floortrans.loaders.augmentations import (RandomCropToSizeTorch,
                                              ResizePaddedTorch,
                                              Compose,
                                              DictToTensor,
                                              ColorJitterTorch,
                                              RandomRotations)
from torchvision.transforms import RandomChoice
from torch.utils import data
from torch.nn.functional import softmax
from tqdm import tqdm

from floortrans.loaders import FloorplanSVG
from floortrans.models import get_model
from floortrans.losses import UncertaintyLoss
from floortrans.metrics import get_px_acc, runningScore
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt


def train(args, log_dir, writer, logger):

    with open(log_dir+'/args.json', 'w') as out:
        json.dump(vars(args), out, indent=4)

    # Augmentation setup
    if args.scale:
        aug = Compose([RandomChoice([RandomCropToSizeTorch(data_format='dict', size=(args.image_size, args.image_size)),
                                     ResizePaddedTorch((0, 0), data_format='dict', size=(args.image_size, args.image_size))]),
                       RandomRotations(format='cubi'),
                       DictToTensor(),
                       ColorJitterTorch()])
    else:
        aug = Compose([RandomCropToSizeTorch(data_format='dict', size=(args.image_size, args.image_size)),
                       RandomRotations(format='cubi'),
                       DictToTensor(),
                       ColorJitterTorch()])

    # Setup Dataloader
    writer.add_text('parameters', str(vars(args)))
    logging.info('Loading data...')
    train_set = FloorplanSVG(args.data_path, 'train.txt', format='lmdb',
                             augmentations=aug)
    val_set = FloorplanSVG(args.data_path, 'val.txt', format='lmdb',
                           augmentations=DictToTensor())

    if args.debug:
        num_workers = 0
        print("In debug mode.")
        logger.info('In debug mode.')
    else:
        num_workers = 8

    trainloader = data.DataLoader(train_set, batch_size=args.batch_size,
                                  num_workers=num_workers, shuffle=True, pin_memory=True)
    valloader = data.DataLoader(val_set, batch_size=1,
                                num_workers=num_workers, pin_memory=True)

    # Setup Model
    logging.info('Loading model...')
    input_slice = [21, 12, 11]
    if args.arch == 'hg_furukawa_original':
        model = get_model(args.arch, 51)
        criterion = UncertaintyLoss(input_slice=input_slice)
        if args.furukawa_weights:
            logger.info("Loading furukawa model weights from checkpoint '{}'".format(args.furukawa_weights))
            checkpoint = torch.load(args.furukawa_weights)
            model.load_state_dict(checkpoint['model_state'])
            criterion.load_state_dict(checkpoint['criterion_state'])

        model.conv4_ = torch.nn.Conv2d(256, args.n_classes, bias=True, kernel_size=1)
        model.upsample = torch.nn.ConvTranspose2d(args.n_classes, args.n_classes, kernel_size=4, stride=4)
        for m in [model.conv4_, model.upsample]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
    else:
        model = get_model(args.arch, args.n_classes)
        criterion = UncertaintyLoss(input_slice=input_slice)

    model.cuda()

    # Drawing graph for TensorBoard
    dummy = torch.zeros((2, 3, args.image_size, args.image_size)).cuda()
    model(dummy)
    writer.add_graph(model, dummy)

    params = [{'params': model.parameters(), 'lr': args.l_rate},
              {'params': criterion.parameters(), 'lr': args.l_rate}]
    if args.optimizer == 'adam-patience':
        optimizer = torch.optim.Adam(params, eps=1e-8, betas=(0.9, 0.999))
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args.patience, factor=0.5)
    elif args.optimizer == 'adam-patience-previous-best':
        optimizer = torch.optim.Adam(params, eps=1e-8, betas=(0.9, 0.999))
    elif args.optimizer == 'sgd':
        def lr_drop(epoch):
            return (1 - epoch/args.n_epoch)**0.9
        optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=10**-4, nesterov=True)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_drop)
    elif args.optimizer == 'adam-scheduler':
        def lr_drop(epoch):
            return 0.5 ** np.floor(epoch / args.l_rate_drop)
        optimizer = torch.optim.Adam(params, eps=1e-8, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_drop)

    first_best = True
    best_loss = np.inf
    best_loss_var = np.inf
    best_train_loss = np.inf
    best_acc = 0 
    start_epoch = 0
    running_metrics_room_val = runningScore(input_slice[1])
    running_metrics_icon_val = runningScore(input_slice[2])
    best_val_loss_variance = np.inf
    no_improvement = 0
    if args.weights is not None:
        if os.path.exists(args.weights):
            logger.info("Loading model and optimizer from checkpoint '{}'".format(args.weights))
            checkpoint = torch.load(args.weights)
            model.load_state_dict(checkpoint['model_state'])
            criterion.load_state_dict(checkpoint['criterion_state'])
            if not args.new_hyperparams:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                logger.info("Using old optimizer state.")
            logger.info("Loaded checkpoint '{}' (epoch {})".format(args.weights, checkpoint['epoch']))
        else:
            logger.info("No checkpoint found at '{}'".format(args.weights)) 

    for epoch in range(start_epoch, args.n_epoch):
        model.train()
        lossess = []
        losses = pd.DataFrame()
        variances = pd.DataFrame()
        ss = pd.DataFrame()
        # Training
        for i, samples in tqdm(enumerate(trainloader), total=len(trainloader),
                               ncols=80, leave=False):
            images = samples['image'].cuda(non_blocking=True)
            labels = samples['label'].cuda(non_blocking=True)

            outputs = model(images)

            loss = criterion(outputs, labels)
            lossess.append(loss.item())
            losses = losses.append(criterion.get_loss(), ignore_index=True)
            variances = variances.append(criterion.get_var(), ignore_index=True)
            ss = ss.append(criterion.get_s(), ignore_index=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = np.mean(lossess)
        avg_loss = np.inf
        loss = losses.mean()
        variance = variances.mean()
        s = ss.mean()

        logging.info("Epoch [%d/%d] Loss: %.4f" % (epoch+1, args.n_epoch, avg_loss))

        writer.add_scalars('training/loss', loss, global_step=1+epoch)
        writer.add_scalars('training/variance', variance, global_step=1+epoch)
        writer.add_scalars('training/s', s, global_step=1+epoch)
        current_lr = {'base': optimizer.param_groups[0]['lr'],
                      'var': optimizer.param_groups[1]['lr']}
        writer.add_scalars('training/lr', current_lr, global_step=1+epoch)

        # Validation
        model.eval()
        val_losses = pd.DataFrame()
        val_variances = pd.DataFrame()
        val_ss = pd.DataFrame()
        px_rooms = 0
        px_icons = 0
        total_px = 0
        for i_val, samples_val in tqdm(enumerate(valloader), total=len(valloader), ncols=80, leave=False):
            with torch.no_grad():
                images_val = samples_val['image'].cuda(non_blocking=True)
                labels_val = samples_val['label'].cuda(non_blocking=True)

                outputs = model(images_val)
                labels_val = F.interpolate(labels_val, size=outputs.shape[2:], mode='bilinear', align_corners=False)
                loss = criterion(outputs, labels_val)

                room_pred = outputs[0, input_slice[0]:input_slice[0]+input_slice[1]].argmax(0).data.cpu().numpy()
                room_gt = labels_val[0, input_slice[0]].data.cpu().numpy()
                running_metrics_room_val.update(room_gt, room_pred)

                icon_pred = outputs[0, input_slice[0]+input_slice[1]:].argmax(0).data.cpu().numpy()
                icon_gt = labels_val[0, input_slice[0]+1].data.cpu().numpy()
                running_metrics_icon_val.update(icon_gt, icon_pred)
                total_px += outputs[0, 0].numel()
                pr, pi = get_px_acc(outputs[0], labels_val[0], input_slice, 0)
                px_rooms += float(pr)
                px_icons += float(pi)

                val_losses = val_losses.append(criterion.get_loss(), ignore_index=True)
                val_variances = val_variances.append(criterion.get_var(), ignore_index=True)
                val_ss = val_ss.append(criterion.get_s(), ignore_index=True)

        val_loss = val_losses.mean()
        # print("CNN done", val_mid-val_start)
        val_variance = val_variances.mean()
        logging.info("val_loss: "+str(val_loss))
        writer.add_scalars('validation loss', val_loss, global_step=1+epoch)
        # print(val_variance)
        writer.add_scalars('validation variance', val_variance, global_step=1+epoch)
        if args.optimizer == 'adam-patience':
            scheduler.step(val_loss['total loss with variance'])
        elif args.optimizer == 'adam-patience-previous-best':
            if best_val_loss_variance > val_loss['total loss with variance']:
                best_val_loss_variance = val_loss['total loss with variance']
                no_improvement = 0
            else:
                no_improvement += 1
            if no_improvement >= args.patience:
                logger.info("No no_improvement for " + str(no_improvement) + " loading last best model and reducing learning rate.")
                checkpoint = torch.load(log_dir+"/model_best_val_loss_var.pkl")
                model.load_state_dict(checkpoint['model_state'])
                for i, p in enumerate(optimizer.param_groups):
                    optimizer.param_groups[i]['lr'] = p['lr'] * 0.1
                no_improvement = 0

        elif args.optimizer == 'sgd' or 'adam-scheduler':
            scheduler.step(epoch+1)

        val_variance = val_variances.mean()
        val_s = val_ss.mean()
        logger.info("val_loss: "+str(val_loss))
        room_score, room_class_iou = running_metrics_room_val.get_scores()
        writer.add_scalars('validation/room/general', room_score, global_step=1+epoch)
        writer.add_scalars('validation/room/IoU', room_class_iou['Class IoU'], global_step=1+epoch)
        writer.add_scalars('validation/room/Acc', room_class_iou['Class Acc'], global_step=1+epoch)
        running_metrics_room_val.reset()

        icon_score, icon_class_iou = running_metrics_icon_val.get_scores()
        writer.add_scalars('validation/icon/general', icon_score, global_step=1+epoch)
        writer.add_scalars('validation/icon/IoU', icon_class_iou['Class IoU'], global_step=1+epoch)
        writer.add_scalars('validation/icon/Acc', icon_class_iou['Class Acc'], global_step=1+epoch)
        running_metrics_icon_val.reset()

        writer.add_scalars('validation/loss', val_loss, global_step=1+epoch)
        writer.add_scalars('validation/variance', val_variance, global_step=1+epoch)
        writer.add_scalars('validation/s', val_s, global_step=1+epoch)

        if val_loss['total loss with variance'] < best_loss_var:
            best_loss_var = val_loss['total loss with variance']
            logger.info("Best validation loss with variance found saving model...")
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'criterion_state': criterion.state_dict(),
                     'optimizer_state': optimizer.state_dict(),
                     'best_loss': best_loss}
            torch.save(state, log_dir+"/model_best_val_loss_var.pkl")
            # Making example prediction images to TensorBoard
            if args.plot_samples:
                for i, samples_val in enumerate(valloader):
                    with torch.no_grad():
                        if i == 4:
                            break

                        images_val = samples_val['image'].cuda(non_blocking=True)
                        labels_val = samples_val['label'].cuda(non_blocking=True)

                        if first_best:
                            # save image and label
                            writer.add_image("Image "+str(i), images_val[0])
                            # add_labels_tensorboard(writer, labels_val)
                            for j, l in enumerate(labels_val.squeeze().cpu().data.numpy()):
                                fig = plt.figure(figsize=(18, 12))
                                plot = fig.add_subplot(111)
                                if j < 21:
                                    cax = plot.imshow(l, vmin=0, vmax=1)
                                else:
                                    cax = plot.imshow(l, vmin=0, vmax=19, cmap=plt.cm.tab20)
                                fig.colorbar(cax)
                                writer.add_figure("Image "+str(i)+" label/Channel "+str(j), fig)

                        outputs = model(images_val)

                        # add_predictions_tensorboard(writer, outputs, epoch)
                        pred_arr = torch.split(outputs, input_slice, 1)
                        heatmap_pred, rooms_pred, icons_pred = pred_arr

                        rooms_pred = softmax(rooms_pred, 1).cpu().data.numpy()
                        icons_pred = softmax(icons_pred, 1).cpu().data.numpy()

                        label = "Image "+str(i)+" prediction/Channel "

                        for j, l in enumerate(np.squeeze(heatmap_pred)):
                            fig = plt.figure(figsize=(18, 12))
                            plot = fig.add_subplot(111)
                            cax = plot.imshow(l, vmin=0, vmax=1)
                            fig.colorbar(cax)
                            writer.add_figure(label+str(j), fig, global_step=1+epoch)

                        fig = plt.figure(figsize=(18, 12))
                        plot = fig.add_subplot(111)
                        cax = plot.imshow(np.argmax(np.squeeze(rooms_pred), axis=0), vmin=0, vmax=19, cmap=plt.cm.tab20)
                        fig.colorbar(cax)
                        writer.add_figure(label+str(j+1), fig, global_step=1+epoch)

                        fig = plt.figure(figsize=(18, 12))
                        plot = fig.add_subplot(111)
                        cax = plot.imshow(np.argmax(np.squeeze(icons_pred), axis=0), vmin=0, vmax=19, cmap=plt.cm.tab20)
                        fig.colorbar(cax)
                        writer.add_figure(label+str(j+2), fig, global_step=1+epoch)

            first_best = False

        if val_loss['total loss'] < best_loss:
            best_loss = val_loss['total loss']
            logger.info("Best validation loss found saving model...")
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'criterion_state': criterion.state_dict(),
                     'optimizer_state': optimizer.state_dict(),
                     'best_loss': best_loss}
            torch.save(state, log_dir+"/model_best_val_loss.pkl")

        px_acc = room_score["Mean Acc"] + icon_score["Mean Acc"]
        if px_acc > best_acc:
            best_acc = px_acc
            logger.info("Best validation pixel accuracy found saving model...")
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'criterion_state': criterion.state_dict(),
                     'optimizer_state': optimizer.state_dict()}
            torch.save(state, log_dir+"/model_best_val_acc.pkl")

        if avg_loss < best_train_loss:
            best_train_loss = avg_loss
            logger.info("Best training loss with variance...")
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'criterion_state': criterion.state_dict(),
                     'optimizer_state': optimizer.state_dict()}
            torch.save(state, log_dir+"/model_best_train_loss_var.pkl")

    logger.info("Last epoch done saving final model...")
    state = {'epoch': epoch+1,
             'model_state': model.state_dict(),
             'criterion_state': criterion.state_dict(),
             'optimizer_state': optimizer.state_dict()}
    torch.save(state, log_dir+"/model_last_epoch.pkl")


if __name__ == '__main__':
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    parser = argparse.ArgumentParser(description='Hyperparameters')
    parser.add_argument('--arch', nargs='?', type=str, default='hg_furukawa_original',
                        help='Architecture to use.')
    parser.add_argument('--optimizer', nargs='?', type=str, default='adam-patience-previous-best',
                        help='Optimizer to use [\'adam, sgd\']')
    parser.add_argument('--data-path', nargs='?', type=str, default='data/cubicasa5k/',
                        help='Path to data directory')
    parser.add_argument('--n-classes', nargs='?', type=int, default=44,
                        help='# of the epochs')
    parser.add_argument('--n-epoch', nargs='?', type=int, default=1000,
                        help='# of the epochs')
    parser.add_argument('--batch-size', nargs='?', type=int, default=26,
                        help='Batch Size')
    parser.add_argument('--image-size', nargs='?', type=int, default=256,
                        help='Image size in training')
    parser.add_argument('--l-rate', nargs='?', type=float, default=1e-3,
                        help='Learning Rate')
    parser.add_argument('--l-rate-var', nargs='?', type=float, default=1e-3,
                        help='Learning Rate for Variance')
    parser.add_argument('--l-rate-drop', nargs='?', type=float, default=200,
                        help='Learning rate drop after how many epochs?')
    parser.add_argument('--patience', nargs='?', type=int, default=10,
                        help='Learning rate drop patience')
    parser.add_argument('--feature-scale', nargs='?', type=int, default=1,
                        help='Divider for # of features to use')
    parser.add_argument('--weights', nargs='?', type=str, default=None,
                        help='Path to previously trained model weights file .pkl')
    parser.add_argument('--furukawa-weights', nargs='?', type=str, default=None,
                        help='Path to previously trained furukawa model weights file .pkl')
    parser.add_argument('--new-hyperparams', nargs='?', type=bool,
                        default=False, const=True,
                        help='Continue training with new hyperparameters')
    parser.add_argument('--log-path', nargs='?', type=str, default='runs_cubi/',
                        help='Path to log directory')
    parser.add_argument('--debug', nargs='?', type=bool,
                        default=False, const=True,
                        help='Continue training with new hyperparameters')
    parser.add_argument('--plot-samples', nargs='?', type=bool,
                        default=False, const=True,
                        help='Plot floorplan segmentations to Tensorboard.')
    parser.add_argument('--scale', nargs='?', type=bool,
                        default=False, const=True,
                        help='Rescale to 256x256 augmentation.')
    args = parser.parse_args()

    log_dir = args.log_path + '/' + time_stamp + '/'
    writer = SummaryWriter(log_dir)
    logger = logging.getLogger('train')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_dir+'/train.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    train(args, log_dir, writer, logger)
