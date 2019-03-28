import torch
from torch.nn import Parameter, Module
from torch.nn.functional import mse_loss, cross_entropy, interpolate
import pandas as pd


class UncertaintyLoss(Module):
    def __init__(self, input_slice=[21, 13, 17],
                 target_slice=[21, 1, 1], sub=0,
                 cuda=True, mask=False):
        super(UncertaintyLoss, self).__init__()
        self.input_slice = input_slice
        self.target_slice = target_slice
        self.loss = None
        self.loss_rooms = None
        self.loss_icons = None
        self.loss_heatmap = None
        self.mask = mask
        self.sub = sub
        self.cuda = cuda
        self.log_vars = Parameter(torch.tensor([0, 0], requires_grad=True, dtype=torch.float32).cuda())
        self.log_vars_mse = Parameter(torch.zeros(input_slice[0], requires_grad=True, dtype=torch.float32).cuda())

    def forward(self, input, target):
        n, c, h, w = input.size()
        nt, ct, ht, wt = target.size()
        if h != ht or w != wt:  # upsample labels
            target = target.unsqueeze(1)
            target = interpolate(target, size=(ct, h, w), mode='nearest')
            target = target.squeeze(1)

        pred_arr = torch.split(input, self.input_slice, 1)
        heatmap_pred, rooms_pred, icons_pred = pred_arr

        target_arr = torch.split(target, self.target_slice, 1)
        heatmap_target, rooms_target, icons_target = target_arr

        # removing empty dimension if batch size is 1
        rooms_target = torch.squeeze(rooms_target, 1)
        icons_target = torch.squeeze(icons_target, 1)

        # Segmentation labels to correct type
        if self.cuda:
            rooms_target = rooms_target.type(torch.cuda.LongTensor) - self.sub
            icons_target = icons_target.type(torch.cuda.LongTensor) - self.sub
        else:
            rooms_target = rooms_target.type(torch.LongTensor) - self.sub
            icons_target = icons_target.type(torch.LongTensor) - self.sub

        self.loss_rooms_var = cross_entropy(input=rooms_pred*torch.exp(-self.log_vars[0]), target=rooms_target)
        self.loss_icons_var = cross_entropy(input=icons_pred*torch.exp(-self.log_vars[1]), target=icons_target)

        self.loss_rooms = cross_entropy(input=rooms_pred, target=rooms_target)
        self.loss_icons = cross_entropy(input=icons_pred, target=icons_target)

        if self.mask:
            heatmap_mask = rooms_pred
            self.loss_heatmap_var, self.vars_sum, self.loss_heatmap = self.homosced_heatmap_mse_loss_mask(heatmap_pred, heatmap_target, heatmap_mask, self.log_vars_mse)
        else:
            self.loss_heatmap_var = self.homosced_heatmap_mse_loss(heatmap_pred, heatmap_target, self.log_vars_mse)
            self.loss_heatmap = mse_loss(input=heatmap_pred, target=heatmap_target)

        self.loss = self.loss_rooms + self.loss_icons + self.loss_heatmap
        # self.loss = self.loss_heatmap
        self.loss_var = self.loss_rooms_var + self.loss_icons_var + self.loss_heatmap_var
        # self.loss_var = self.loss_heatmap_var

        return self.loss_var

    def homosced_heatmap_mse_loss(self, input, target, logvars):
        # we have n heatmaps, i.e. n heatmap tasks
        n, ntasks, h, w = input.size()

        # make a 2d tensor from both input and target  so that we have n tasks cols
        preds = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, ntasks)
        targets = target.transpose(1, 2).transpose(2, 3).contiguous().view(-1, ntasks)

        # take elementwise subtraction and raise to the power of two
        diff = (preds - targets) ** 2

        # measure task dependent mse loss
        mse_loss_per_tasks = torch.sum(diff, 0) / (n * h * w)

        # apply uncertainty magic
        # w_mse_loss = torch.exp(-logvars) * mse_loss_per_tasks + logvars
        w_mse_loss = torch.exp(-logvars) * mse_loss_per_tasks + torch.log(1+torch.exp(logvars))

        # take sum and return it
        w_mse_loss_total = w_mse_loss.sum()

        return w_mse_loss_total

    def get_loss(self):
        d = {'total loss': [self.loss.data],
             'room loss': [self.loss_rooms.data],
             'icon loss': [self.loss_icons.data],
             'heatmap loss': [self.loss_heatmap.data],
             'total loss with variance': [self.loss_var.data],
             'room loss with variance': [self.loss_rooms_var.data],
             'icon loss with variance': [self.loss_icons_var.data],
             'heatmap loss with variance': [self.loss_heatmap_var.data]}
        return pd.DataFrame(data=d)

    def get_var(self):
        variance = torch.exp(self.log_vars.data)
        mse_variance = torch.exp(self.log_vars_mse.data)
        d = {'room variance': [variance[0]],
             'icon variance': [variance[1]]}
        for i, m in enumerate(mse_variance):
            key = 'heatmap ' + str(i)
            d[key] = [m]

        return pd.DataFrame(data=d)
    
    def get_s(self):
        s = self.log_vars.data
        mse_s = self.log_vars_mse.data
        d = {'room s': [s[0]],
             'icon s': [s[1]]}
        for i, m in enumerate(mse_s):
            key = 'heatmap s' + str(i)
            d[key] = [m]

        return pd.DataFrame(data=d)

    def homosced_heatmap_mse_loss_mask(self, input, target, heatmap_mask, logvars):
        # we have n heatmaps, i.e. n heatmap tasks
        n, ntasks, h, w = input.size()
        walls = torch.argmax(torch.nn.functional.softmax(heatmap_mask, 1), 1)
        walls = walls.unsqueeze(1)
        mask = (walls != 0)
        num_elem = mask.sum().to(torch.float32)
        mask = mask.repeat(1, 21, 1, 1)

        input = input * mask.to(torch.float32)
        target = target * mask.to(torch.float32)

        # make a 2d tensor from both input and target  so that we have n tasks cols
        preds = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, ntasks)
        targets = target.transpose(1, 2).transpose(2, 3).contiguous().view(-1, ntasks)

        # take elementwise subtraction and raise to the power of two
        diff = (preds - targets) ** 2

        # measure task dependent mse loss
        mse_loss_per_tasks = torch.sum(diff, 0) / num_elem

        # apply uncertainty magic
        w_mse_loss = torch.exp(-logvars) * mse_loss_per_tasks + logvars

        # take sum and return it
        w_mse_loss_total = w_mse_loss.sum()

        return w_mse_loss_total, logvars.data.sum(), mse_loss_per_tasks.sum()
