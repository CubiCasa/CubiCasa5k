import torch
import random
import numpy as np
from math import inf
from floortrans.loaders import svg_utils
import cv2


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, sample):
        for a in self.augmentations:
            sample = a(sample)

        return sample

# 0. I
# 1. I top to right
# 2. I vertical flip
# 3. I top to left
# 4. L horizontal flip
# 5. L
# 6. L vertical flip
# 7. L horizontal and vertical flip
# 8. T
# 9. T top to right
# 10. T top to down
# 11. T top to left
# 12. X or +
# 13. Opening left corner
# 14. Opening right corner
# 15. Opening up corner
# 16. Opening down corer
# 17. Icon upper left
# 18. Icon upper right
# 19. Icon lower left
# 20. Icon lower right



class RandomRotations(object):
    def __init__(self, format='furu'):
        if format == 'furu':
            self.augment = self.furu
        elif format == 'cubi':
            self.augment = self.cubi

    def __call__(self, sample):
        return self.augment(sample)

    def cubi(self, sample):
        fplan = sample['image']
        segmentation = sample['label']
        heatmap_points = sample['heatmaps']
        scale = sample['scale']
        num_of_rotations = int(torch.randint(0, 3, (1,)))
        hmapp_convert_map = {0: 1, 1: 2, 2: 3, 3: 0, 4: 5, 5: 6, 6: 7, 7: 4, 8: 9, 9: 10,
                             10: 11, 11: 8, 12: 12, 13: 15, 14: 16, 15: 14, 16: 13,
                             17: 18, 18: 20, 19: 17, 20: 19}

        for i in range(num_of_rotations):
            fplan = fplan.transpose(2, 1).flip(2)
            segmentation = segmentation.transpose(2, 1).flip(2)
            points_rotated = dict()
            for junction_type, points in heatmap_points.items():
                new_junction_type = hmapp_convert_map[junction_type]
                new_heatmap_points = []
                for point in points:
                    x = fplan.shape[1] - 1 - point[1]
                    y = point[0]
                    # if y > 256 or x > 256:
                        # __import__('ipdb').set_trace()
                    new_heatmap_points.append([x, y])

                points_rotated[new_junction_type] = new_heatmap_points

            heatmap_points = points_rotated

        sample = {'image': fplan,
                  'label': segmentation,
                  'scale': scale,
                  'heatmaps': heatmap_points}

        return sample

    def furu(self, sample):
        fplan = sample['image']
        segmentation = sample['label']
        heatmap_points = sample['heatmap_points']
        num_of_rotations = int(torch.randint(0, 3, (1,)))
        for i in range(num_of_rotations):
            fplan = fplan.transpose(2, 1).flip(2)
            segmentation = segmentation.transpose(2, 1).flip(2)

            hmapp_convert_map = {0: 1, 1: 2, 2: 3, 3: 0, 4: 5, 5: 6, 6: 7, 7: 4, 8: 9, 9: 10,
                                 10: 11, 11: 8, 12: 12, 13: 15, 14: 16, 15: 14, 16: 13,
                                 17: 18, 18: 20, 19: 17, 20: 19}

            points_rotated = dict()
            for junction_type, points in heatmap_points.items():
                new_junction_type = hmapp_convert_map[junction_type]
                new_heatmap_points = []
                for point in points:
                    new_heatmap_points.append([fplan.shape[1]-1-point[1], point[0]])

                points_rotated[new_junction_type] = new_heatmap_points

            heatmap_points = points_rotated

        sample = {'image': fplan,
                  'label': segmentation,
                  'heatmap_points': heatmap_points}

        return sample


def clip_heatmaps(heatmaps, minx, maxx, miny, maxy):
    def clip(p):
        return (p[0] < maxx and
                p[0] >= minx and
                p[1] < maxy and
                p[1] >= miny)
    res = {}
    for key, value in heatmaps.items():
        res[key] = list(filter(clip, value))
        for i, e in enumerate(res[key]):
            res[key][i] = (e[0]-minx, e[1]-miny)

    return res


class DictToTensor(object):
    def __init__(self, data_format='cubi'):
        if data_format == 'cubi':
            self.augment = self.cubi
        elif data_format == 'furukawa':
            self.augment = self.furukawa

    def __call__(self, sample):
        return self.augment(sample)

    def cubi(self, sample):
        image, label = sample['image'], sample['label']
        _, height, width = label.shape
        heatmaps = sample['heatmaps']
        scale = sample['scale']

        heatmap_tensor = np.zeros((21, height, width))
        for channel, coords in heatmaps.items():
            for x, y in coords:
                if x >= width:
                    x -= 1
                if y >= height:
                    y -= 1
                heatmap_tensor[int(channel), int(y), int(x)] = 1

        # Gaussian filter
        kernel = svg_utils.get_gaussian2D(int(30*scale))
        for i, h in enumerate(heatmap_tensor):
            heatmap_tensor[i] = cv2.filter2D(h, -1, kernel)

        heatmap_tensor = torch.FloatTensor(heatmap_tensor)

        label = torch.cat((heatmap_tensor, label), 0)

        return {'image': image, 'label': label}

    def furukawa(self, sample):
        image, label = sample['image'], sample['label']
        _, height, width = label.shape
        heatmap_points = sample['heatmap_points']

        heatmap_tensor = np.zeros((21, height, width))
        for channel, coords in heatmap_points.items():
            for x, y in coords:
                heatmap_tensor[int(channel), int(y), int(x)] = 1

        # Gaussian filter
        kernel = svg_utils.get_gaussian2D(13)
        for i, h in enumerate(heatmap_tensor):
            heatmap_tensor[i] = cv2.filter2D(h, -1, kernel, borderType=cv2.BORDER_CONSTANT, delta=0)

        heatmap_tensor = torch.FloatTensor(heatmap_tensor)

        label = torch.cat((heatmap_tensor, label), 0)

        return {'image': image, 'label': label}


class RotateNTurns(object):

    def rot_tensor(self, t, n):
        # One turn clock wise
        if n == 1:
            t = t.flip(2).transpose(3, 2)
        # One turn counter clock wise
        elif n == -1:
            t = t.transpose(3, 2).flip(2)
        # Two turns clock wise
        elif n == 2:
            t = t.flip(2).flip(3)

        return t

    def rot_points(self, t, n):
        # Swapping corner ts 
        t_sorted = t.clone().detach()
        # One turn clock wise
        if n == 1:
            # I junctions
            t_sorted[:, 1] = t[:, 0]
            t_sorted[:, 2] = t[:, 1]
            t_sorted[:, 3] = t[:, 2]
            t_sorted[:, 0] = t[:, 3]
            # L junctions
            t_sorted[:, 5] = t[:, 4]
            t_sorted[:, 6] = t[:, 5]
            t_sorted[:, 7] = t[:, 6]
            t_sorted[:, 4] = t[:, 7]
            # T junctions
            t_sorted[:, 9] = t[:, 8]
            t_sorted[:, 10] = t[:, 9]
            t_sorted[:, 11] = t[:, 10]
            t_sorted[:, 8] = t[:, 11]
            # Opening corners
            t_sorted[:, 15] = t[:, 13]
            t_sorted[:, 16] = t[:, 14]
            t_sorted[:, 14] = t[:, 15]
            t_sorted[:, 13] = t[:, 16]
            # Icon corners
            t_sorted[:, 18] = t[:, 17]
            t_sorted[:, 20] = t[:, 18]
            t_sorted[:, 17] = t[:, 19]
            t_sorted[:, 19] = t[:, 20]
        # One turn counter clock wise
        elif n == -1:
            # I junctions
            t_sorted[:, 3] = t[:, 0]
            t_sorted[:, 0] = t[:, 1]
            t_sorted[:, 1] = t[:, 2]
            t_sorted[:, 2] = t[:, 3]
            # L junctions
            t_sorted[:, 7] = t[:, 4]
            t_sorted[:, 4] = t[:, 5]
            t_sorted[:, 5] = t[:, 6]
            t_sorted[:, 6] = t[:, 7]
            # T junctions
            t_sorted[:, 11] = t[:, 8]
            t_sorted[:, 8] = t[:, 9]
            t_sorted[:, 9] = t[:, 10]
            t_sorted[:, 10] = t[:, 11]
            # Opening corners
            t_sorted[:, 16] = t[:, 13]
            t_sorted[:, 15] = t[:, 14]
            t_sorted[:, 13] = t[:, 15]
            t_sorted[:, 14] = t[:, 16]
            # Icon corners
            t_sorted[:, 19] = t[:, 17]
            t_sorted[:, 17] = t[:, 18]
            t_sorted[:, 20] = t[:, 19]
            t_sorted[:, 18] = t[:, 20]
        # Two turns clock wise
        elif n == 2:
            t_sorted = t.clone().detach()
            # I junctions
            t_sorted[:, 2] = t[:, 0]
            t_sorted[:, 3] = t[:, 1]
            t_sorted[:, 0] = t[:, 2]
            t_sorted[:, 4] = t[:, 3]
            # L junctions
            t_sorted[:, 6] = t[:, 4]
            t_sorted[:, 7] = t[:, 5]
            t_sorted[:, 4] = t[:, 6]
            t_sorted[:, 5] = t[:, 7]
            # T junctions
            t_sorted[:, 10] = t[:, 8]
            t_sorted[:, 11] = t[:, 9]
            t_sorted[:, 8] = t[:, 10]
            t_sorted[:, 9] = t[:, 11]
            # Opening corners
            t_sorted[:, 14] = t[:, 13]
            t_sorted[:, 13] = t[:, 14]
            t_sorted[:, 16] = t[:, 15]
            t_sorted[:, 15] = t[:, 16]
            # Icon corners
            t_sorted[:, 20] = t[:, 17]
            t_sorted[:, 19] = t[:, 18]
            t_sorted[:, 18] = t[:, 19]
            t_sorted[:, 17] = t[:, 20]
        elif n == 0:
            return t_sorted

        return t_sorted

    def __call__(self, sample, data_type, n):
        if data_type == 'tensor':
            return self.rot_tensor(sample, n)
        elif data_type == 'points':
            return self.rot_points(sample, n)


class RandomCropToSizeTorch(object):
    def __init__(self, input_slice=[21, 1, 1], size=(256, 256), fill=(0, 0), data_format='tensor',
                 dtype=torch.float32, max_size=None):
        self.size = size
        self.width = size[0]
        self.height = size[1]
        self.dtype = dtype
        self.fill = fill
        self.max_size = max_size
        self.input_slice = input_slice

        if data_format == 'dict':
            self.augment = self.augment_dict
        elif data_format == 'tensor':
            self.augment = self.augment_tesor
        elif data_format == 'dict furu':
            self.augment = self.augment_dict_furu

    def __call__(self, sample):
        return self.augment(sample)

    def augment_tesor(self, sample):
        image, label = sample['image'], sample['label']
        img_w = image.shape[2]
        img_h = image.shape[1]
        pad_w = int(self.width / 2)
        pad_h = int(self.height / 2)

        new_w = self.width + max(img_w, self.width)
        new_h = self.height + max(img_h, self.height)

        new_image = torch.zeros([image.shape[0], new_h, new_w], dtype=self.dtype)
        new_image[:, pad_h:img_h+pad_h, pad_w:img_w+pad_w] = image

        new_heatmaps = torch.zeros([self.input_slice[0], new_h, new_w], dtype=self.dtype)
        new_heatmaps[:, pad_h:img_h+pad_h, pad_w:img_w+pad_w] = label[:self.input_slice[0]]

        new_rooms = torch.full((self.input_slice[1], new_h, new_w), self.fill[0])
        new_rooms[:, pad_h:img_h+pad_h, pad_w:img_w+pad_w] = label[self.input_slice[0]]
        new_icons = torch.full((self.input_slice[2], new_h, new_w), self.fill[1])
        new_icons[:, pad_h:img_h+pad_h, pad_w:img_w+pad_w] = label[self.input_slice[0]+self.input_slice[1]]

        label = torch.cat((new_heatmaps, new_rooms, new_icons), 0)
        image = new_image

        removed_up = random.randint(0, new_h - self.width)
        removed_left = random.randint(0, new_w - self.height)

        removed_down = new_h - self.height - removed_up
        removed_right = new_w - self.width - removed_left

        if removed_down == 0 and removed_right == 0:
            image = image[:, removed_up:, removed_left:]
            label = label[:, removed_up:, removed_left:]
        elif removed_down == 0:
            image = image[:, removed_up:, removed_left:-removed_right]
            label = label[:, removed_up:, removed_left:-removed_right]
        elif removed_right == 0:
            image = image[:, removed_up:-removed_down, removed_left:]
            label = label[:, removed_up:-removed_down, removed_left:]
        else:
            image = image[:, removed_up:-removed_down, removed_left:-removed_right]
            label = label[:, removed_up:-removed_down, removed_left:-removed_right]

        return {'image': image, 'label': label}

    def augment_dict(self, sample):
        image, label = sample['image'], sample['label']
        heatmap_points = sample['heatmaps']
        img_w = image.shape[2]
        img_h = image.shape[1]
        pad_w = int(self.width / 2)
        pad_h = int(self.height / 2)

        new_w = self.width + img_w
        new_h = self.height + img_h

        new_image = torch.full([image.shape[0], new_h, new_w], 255)
        new_image[:, pad_h:img_h+pad_h, pad_w:img_w+pad_w] = image

        new_rooms = torch.full((1, new_h, new_w), self.fill[0])
        new_rooms[:, pad_h:img_h+pad_h, pad_w:img_w+pad_w] = label[0]
        new_icons = torch.full((1, new_h, new_w), self.fill[1])
        new_icons[:, pad_h:img_h+pad_h, pad_w:img_w+pad_w] = label[1]

        label = torch.cat((new_rooms, new_icons), 0)
        image = new_image

        removed_up = random.randint(0, new_h - self.width)
        removed_left = random.randint(0, new_w - self.height)

        removed_down = new_h - self.height - removed_up
        removed_right = new_w - self.width - removed_left

        new_heatmap_points = dict()
        for junction_type, points in heatmap_points.items():
            new_heatmap_points_per_type = []
            for point in points:
                new_heatmap_points_per_type.append([point[0]+pad_w, point[1]+pad_h])

                new_heatmap_points[junction_type] = new_heatmap_points_per_type

        heatmap_points = new_heatmap_points

        if removed_down == 0 and removed_right == 0:
            image = image[:, removed_up:, removed_left:]
            label = label[:, removed_up:, removed_left:]
            heatmap_points = clip_heatmaps(heatmap_points, removed_left, inf, removed_up, inf)
        elif removed_down == 0:
            image = image[:, removed_up:, removed_left:-removed_right]
            label = label[:, removed_up:, removed_left:-removed_right]
            heatmap_points = clip_heatmaps(heatmap_points, removed_left, removed_left+self.width, removed_up, inf)
        elif removed_right == 0:
            image = image[:, removed_up:-removed_down, removed_left:]
            label = label[:, removed_up:-removed_down, removed_left:]
            heatmap_points = clip_heatmaps(heatmap_points, removed_left, inf, removed_up, removed_up+self.width)
        else:
            image = image[:, removed_up:-removed_down, removed_left:-removed_right]
            label = label[:, removed_up:-removed_down, removed_left:-removed_right]
            heatmap_points = clip_heatmaps(heatmap_points, removed_left, removed_left+self.width, removed_up, removed_up+self.height)

        return {'image': image, 'label': label, 'heatmaps': heatmap_points, 'scale': sample['scale']}


    def augment_dict_furu(self, sample):
        image, label = sample['image'], sample['label']
        heatmap_points = sample['heatmap_points']
        img_w = image.shape[2]
        img_h = image.shape[1]
        pad_w = int(self.width / 2)
        pad_h = int(self.height / 2)

        new_w = self.width + img_w
        new_h = self.height + img_h

        new_image = torch.full([image.shape[0], new_h, new_w], 255)
        new_image[:, pad_h:img_h+pad_h, pad_w:img_w+pad_w] = image

        new_rooms = torch.full((1, new_h, new_w), self.fill[0])
        new_rooms[:, pad_h:img_h+pad_h, pad_w:img_w+pad_w] = label[0]
        new_icons = torch.full((1, new_h, new_w), self.fill[1])
        new_icons[:, pad_h:img_h+pad_h, pad_w:img_w+pad_w] = label[1]

        label = torch.cat((new_rooms, new_icons), 0)
        image = new_image

        removed_up = random.randint(0, new_h - self.width)
        removed_left = random.randint(0, new_w - self.height)

        removed_down = new_h - self.height - removed_up
        removed_right = new_w - self.width - removed_left

        new_heatmap_points = dict()
        for junction_type, points in heatmap_points.items():
            new_heatmap_points_per_type = []
            for point in points:
                new_heatmap_points_per_type.append([point[0]+pad_w, point[1]+pad_h])

                new_heatmap_points[junction_type] = new_heatmap_points_per_type

        heatmap_points = new_heatmap_points

        if removed_down == 0 and removed_right == 0:
            image = image[:, removed_up:, removed_left:]
            label = label[:, removed_up:, removed_left:]
            heatmap_points = clip_heatmaps(heatmap_points, removed_left, inf, removed_up, inf)
        elif removed_down == 0:
            image = image[:, removed_up:, removed_left:-removed_right]
            label = label[:, removed_up:, removed_left:-removed_right]
            heatmap_points = clip_heatmaps(heatmap_points, removed_left, removed_left+self.width, removed_up, inf)
        elif removed_right == 0:
            image = image[:, removed_up:-removed_down, removed_left:]
            label = label[:, removed_up:-removed_down, removed_left:]
            heatmap_points = clip_heatmaps(heatmap_points, removed_left, inf, removed_up, removed_up+self.width)
        else:
            image = image[:, removed_up:-removed_down, removed_left:-removed_right]
            label = label[:, removed_up:-removed_down, removed_left:-removed_right]
            heatmap_points = clip_heatmaps(heatmap_points, removed_left, removed_left+self.width, removed_up, removed_up+self.height)

        return {'image': image, 'label': label, 'heatmap_points': heatmap_points}


class ColorJitterTorch(object):

    def __init__(self, b_var=0.4, c_var=0.4, s_var=0.4, dtype=torch.float32, version='dict'):
        self.b_var = b_var
        self.c_var = c_var
        self.s_var = s_var
        self.dtype = dtype
        self.version = version

    def __call__(self, sample):
        res = sample
        image = sample['image']
        image = self.brightness(image, self.b_var)
        image = self.contrast(image, self.c_var)
        image = self.saturation(image, self.s_var)
        res['image'] = image

        return res

    def blend(self, img_1, img_2, var):
        m = torch.tensor([0], dtype=self.dtype).uniform_(-var, var)
        alpha = 1 + m
        res = img_1 * alpha + (1 - alpha) * img_2
        res = torch.clamp(res, min=0, max=255) 

        return res

    def grayscale(self, img):
        red = img[0] * 0.299
        green = img[1] * 0.587
        blue = img[2] * 0.114
        gray = red + green + blue
        gray = torch.clamp(gray, min=0, max=255)
        res = torch.stack((gray, gray, gray), dim=0)

        return res

    def saturation(self, img, var):
        res = self.grayscale(img)
        res = self.blend(img, res, var)

        return res

    def brightness(self, img, var):
        res = torch.zeros(img.shape)
        res = self.blend(img, res, var)

        return res

    def contrast(self, img, var):
        res = self.grayscale(img)
        mean_color = res.mean()
        res = torch.full(res.shape, mean_color)
        res = self.blend(img, res, var)

        return res


class ResizePaddedTorch(object):

    def __init__(self, fill, size=(256, 256),  both=True, dtype=torch.float32, data_format='tensor'):
        self.size = size
        self.width = size[0]
        self.height = size[1]
        self.both = both
        self.dtype = dtype
        self.fill = fill
        self.fill_cval = 255
        if data_format == 'tensor':
            self.augment = self.augment_tensor
        elif data_format == 'dict furu':
            self.augment = self.augment_dict_furu
        elif data_format == 'dict':
            self.augment = self.augment_dict
            self.fill_cval = 1

    def __call__(self, sample):
        # image 1: Bi-cubic interpolation as in original paper
        image, _, _, _ = self.resize_padded(sample['image'], self.size, fill_cval=self.fill_cval, image=True, mode='bilinear', aling_corners=False)
        sample['image'] = image

        return self.augment(sample)

    def augment_tensor(self, sample):
        image, label = sample['image'], sample['label']

        if self.both:
            # labels 0: Nearest-neighbor interpolation
            heatmaps, _, _, _ = self.resize_padded(label[:21], self.size, mode='bilinear', aling_corners=False)
            rooms_padded, _, _, _ = self.resize_padded(label[[21]], self.size, mode='nearest', fill_cval=self.fill[0])
            icons_padded, _, _, _ = self.resize_padded(label[[22]], self.size, mode='nearest', fill_cval=self.fill[1],)
            label = torch.cat((heatmaps, rooms_padded, icons_padded), dim=0)

        return {'image': image, 'label': label}

    def augment_dict_furu(self, sample):
        image, label = sample['image'], sample['label']
        heatmap_points = sample['heatmap_points']

        rooms_padded, _, _, _ = self.resize_padded(label[[0]], self.size, mode='nearest', fill_cval=self.fill[0])
        icons_padded, ratio, y_pad, x_pad = self.resize_padded(label[[1]], self.size, mode='nearest', fill_cval=self.fill[1])
        label = torch.cat((rooms_padded, icons_padded), dim=0)

        new_heatmap_points = dict()
        for junction_type, points in heatmap_points.items():
            new_heatmap_points_per_type = []
            for point in points:
                # Indexing starts from 0 but when we multiply with the ratio we need to start from 0.
                new_x = point[0] * ratio + x_pad
                new_y = point[1] * ratio + y_pad
                new_heatmap_points_per_type.append([new_x, new_y])
                new_heatmap_points[junction_type] = new_heatmap_points_per_type

        heatmap_points = new_heatmap_points

        return {'image': image, 'label': label, 'heatmap_points': heatmap_points}

    def augment_dict(self, sample):
        image, label = sample['image'], sample['label']
        heatmap_points = sample['heatmaps']
        scale = sample['scale']

        rooms_padded, _, _, _ = self.resize_padded(label[[0]], self.size, mode='nearest', fill_cval=self.fill[0])
        icons_padded, ratio, y_pad, x_pad = self.resize_padded(label[[1]], self.size, mode='nearest', fill_cval=self.fill[1])
        label = torch.cat((rooms_padded, icons_padded), dim=0)

        new_heatmap_points = dict()
        for junction_type, points in heatmap_points.items():
            new_heatmap_points_per_type = []
            for point in points:
                # Indexing starts from 0 but when we multiply with the ratio we need to start from 0.
                new_x = point[0] * ratio + x_pad
                new_y = point[1] * ratio + y_pad
                if new_y < 256 and new_x < 256 and new_y >= 0 and new_x >= 0:
                    # __import__('ipdb').set_trace()
                    new_heatmap_points_per_type.append([new_x, new_y])
                    new_heatmap_points[junction_type] = new_heatmap_points_per_type

        heatmap_points = new_heatmap_points

        return {'image': image, 'label': label, 'heatmaps': heatmap_points, 'scale': scale}

    def resize_padded(self, img, new_shape, image=False, fill_cval=0, mode='nearest',
                      aling_corners=None):
        new_shape = torch.tensor([img.shape[0], new_shape[0], new_shape[1]], dtype=self.dtype)
        old_shape = torch.tensor(img.shape, dtype=self.dtype)

        ratio = (new_shape / old_shape).min()
        img_s = torch.tensor(img.shape[1:], dtype=self.dtype)
        interm_shape = (ratio * img_s).ceil()

        interm_shape = [interm_shape[0], interm_shape[1]]

        img = img.unsqueeze(0)
        interm_img = torch.nn.functional.interpolate(img, size=interm_shape, mode=mode, align_corners=aling_corners)
        interm_img = interm_img.squeeze(0)

        a = (interm_img.shape[0], self.size[0], self.size[1])

        new_img = torch.full(a, fill_cval)

        x_pad = int((self.width - interm_img.shape[1]) / 2)
        y_pad = int((self.height - interm_img.shape[2]) / 2)

        new_img[:, x_pad:interm_img.shape[1]+x_pad, y_pad:interm_img.shape[2]+y_pad] = interm_img

        return new_img, ratio, x_pad, y_pad
