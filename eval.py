import numpy as np
from tensorboardX import SummaryWriter
import logging
import argparse
import torch
from datetime import datetime
from torch.utils import data
from floortrans.models import get_model
from floortrans.loaders import FloorplanSVG
from floortrans.loaders.augmentations import DictToTensor, Compose
from floortrans.metrics import get_evaluation_tensors, runningScore
from tqdm import tqdm

room_cls = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bedroom", "Bath", "Hallway", "Railing", "Storage", "Garage", "Other rooms"]
icon_cls = ["Empty", "Window", "Door", "Closet", "Electr. Appl.", "Toilet", "Sink", "Sauna bench", "Fire Place", "Bathtub", "Chimney"]


def print_res(name, res, cls_names, logger):
    basic_res = res[0]
    class_res = res[1]

    basic_names = ''
    basic_values = name
    basic_res_list = ["Overall Acc", "Mean Acc", "Mean IoU", "FreqW Acc"]
    for key in basic_res_list:
        basic_names += ' & ' + key
        val = round(basic_res[key] * 100, 1)
        basic_values += ' & ' + str(val)

    logger.info(basic_names)
    logger.info(basic_values)

    basic_res_list = ["IoU", "Acc"]
    logger.info("IoU & Acc")
    for i, name in enumerate(cls_names):
        iou = class_res['Class IoU'][str(i)]
        acc = class_res['Class Acc'][str(i)]
        iou = round(iou * 100, 1)
        acc = round(acc * 100, 1)
        logger.info(name + " & " + str(iou) + " & " + str(acc) + " \\\\ \\hline")


def evaluate(args, log_dir, writer, logger):

    normal_set = FloorplanSVG(args.data_path, 'test.txt', format='lmdb', lmdb_folder='cubi_lmdb/', augmentations=Compose([DictToTensor()]))
    data_loader = data.DataLoader(normal_set, batch_size=1, num_workers=0)

    checkpoint = torch.load(args.weights)
    # Setup Model
    model = get_model(args.arch, 51)
    n_classes = args.n_classes
    split = [21, 12, 11]
    model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    model.cuda()

    score_seg_room = runningScore(12)
    score_seg_icon = runningScore(11)
    score_pol_seg_room = runningScore(12)
    score_pol_seg_icon = runningScore(11)
    with torch.no_grad():
        for count, val in tqdm(enumerate(data_loader), total=len(data_loader),
                               ncols=80, leave=False):
            logger.info(count)
            things = get_evaluation_tensors(val, model, split, logger, rotate=True)

            label, segmentation, pol_segmentation = things

            score_seg_room.update(label[0], segmentation[0])
            score_seg_icon.update(label[1], segmentation[1])

            score_pol_seg_room.update(label[0], pol_segmentation[0])
            score_pol_seg_icon.update(label[1], pol_segmentation[1])

    print_res("Room segmentation", score_seg_room.get_scores(), room_cls, logger)
    print_res("Room polygon segmentation", score_pol_seg_room.get_scores(), room_cls, logger)
    print_res("Icon segmentation", score_seg_icon.get_scores(), icon_cls, logger)
    print_res("Icon polygon segmentation", score_pol_seg_icon.get_scores(), icon_cls, logger)


if __name__ == '__main__':
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    parser = argparse.ArgumentParser(description='Settings for evaluation')
    parser.add_argument('--arch', nargs='?', type=str, default='hg_furukawa_original',
                        help='Architecture to use [\'hg_furukawa_original, segnet etc\']')
    parser.add_argument('--data-path', nargs='?', type=str, default='data/cubicasa5k/',
                        help='Path to data directory')
    parser.add_argument('--n-classes', nargs='?', type=int, default=44,
                        help='# of the epochs')
    parser.add_argument('--weights', nargs='?', type=str, default=None,
                        help='Path to previously trained model weights file .pkl')
    parser.add_argument('--log-path', nargs='?', type=str, default='runs_cubi/',
                        help='Path to log directory')

    args = parser.parse_args()

    log_dir = args.log_path + '/' + time_stamp + '/'
    writer = SummaryWriter(log_dir)
    logger = logging.getLogger('eval')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_dir+'/eval.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    evaluate(args, log_dir, writer, logger)
