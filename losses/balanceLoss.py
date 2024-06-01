from torch import nn
import torch.nn.functional as F
import numpy as np
import torch



class GaussianLoss(nn.Module):
    def __init__(self, center_R, neg_weight=1.0):
        super(GaussianLoss, self).__init__()
        self.center_R = center_R
        self.neg_weight = neg_weight


    def forward(self, cls_input, center_rate):

        if self.center_R == 2:
            target = self.create_labels_2(cls_input.size(), center_rate)
        else:
            target = self.create_labels(cls_input.size(), center_rate)

        pos_mask = (target == 1)#384*384*b
        neg_mask = (target == 0)
        pos_num = pos_mask.sum().float()
        neg_num = neg_mask.sum().float()
        weight = target.new_zeros(target.size())

        ## ############## set gauss weight  #####################
        center_num =  torch.from_numpy(self.create_gaussian_mask(self.center_R,5.5)).unsqueeze(0).repeat(target.size(0), 1, 1)
        # ############## set hanning weight  #####################
        # center_num = torch.tensor(self.create_hanning_mask(self.center_R)).unsqueeze(0).repeat(int(target.size(0)), 1,1)  # 创建汉宁窗口 并且
        pos_center = (center_num != 0)
        weight[pos_mask] = center_num[pos_center].to(torch.float).cuda()
        weight[pos_mask] = weight[pos_mask]/ int(target.size(0))

        # ############## set normal weight  #####################

        weight[neg_mask] = 1 / neg_num * 80
        weight /= weight.sum()  # 归一化除以所有数值之和
        cls_loss = F.binary_cross_entropy_with_logits(cls_input, target,weight, reduction='sum')

        return cls_loss


    def create_gaussian_mask(self,window_size, sigma):
        x = np.linspace(-(window_size - 1) / 2, (window_size - 1) / 2, window_size)
        y = np.linspace(-(window_size - 1) / 2, (window_size - 1) / 2, window_size)
        xx, yy = np.meshgrid(x, y)
        gauss_mask = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        gauss_mask /= gauss_mask.sum()
        # gauss_mask /= np.max(gauss_mask)
        return gauss_mask

    def create_hanning_mask(self, center_R):
        hann_window = np.outer(  # np.outer 如果a，b是高维数组，函数会自动将其flatten成1维 ，用来求外积
            np.hanning(center_R + 2),
            np.hanning(center_R + 2))
        hann_window /= hann_window.sum()
        return hann_window[1:-1, 1:-1]


    def create_labels_2(self, size, rate):
        ratex, ratey = rate
        labels = np.zeros(size)
        # labels =torch.zeros(size).cuda()
        b, c, h, w = size
        X = ratex * (h - 1)
        Y = ratey * (w - 1)
        intX = np.floor(X).reshape(-1, 1)
        intY = np.floor(Y).reshape(-1, 1)
        CenterXY = np.concatenate((intX, intY), axis=-1)
        for i in range(b):
            CenterX, CenterY = CenterXY[i]
            labels[i, 0,int(CenterX - (self.center_R-1) // 2):int(CenterX + (self.center_R+1) // 2), int(CenterY - (self.center_R-1) // 2):int(CenterY + (self.center_R+1) // 2)] = 1.0
        labels_torch = torch.from_numpy(labels).cuda().float()
        return labels_torch
        return labels

    def create_labels(self, size, rate):
        ratex, ratey = rate
        labels = np.zeros(size)
        b, c, h, w = size
        X = ratex * (h - 1)
        Y = ratey * (w - 1)
        intX = np.round(X).reshape(-1, 1)
        intY = np.round(Y).reshape(-1, 1)
        CenterXY = np.concatenate((intX, intY), axis=-1)
        for i in range(b):
            CenterX, CenterY = CenterXY[i]
            pad_right = pad_left = pad_top = pad_bottom = 0
            if CenterX + self.center_R // 2 > h - 1:
                pad_bottom = int(CenterX + self.center_R // 2 - (h - 1))
            if CenterX - self.center_R // 2 < 0:
                pad_top = int(-1 * (CenterX - self.center_R // 2))
            if CenterY + self.center_R // 2 > h - 1:
                pad_right = int(CenterY + self.center_R // 2 - (w - 1))
            if CenterY - self.center_R // 2 < 0:
                pad_left = int(-1 * (CenterY - self.center_R // 2))
            new_label = np.pad(labels[i, 0], ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant',
                               constant_values=(-1, -1))
            new_center = [CenterX + pad_top, CenterY + pad_left]
            x1, x2, y1, y2 = new_center[0] - self.center_R // 2, \
                             new_center[0] + self.center_R // 2 + 1, \
                             new_center[1] - self.center_R // 2, \
                             new_center[1] + self.center_R // 2 + 1  # 为什么＋在创建标签时左闭右开
            label = new_label.copy()
            label[int(x1):int(x2), int(y1):int(y2)] = 1
            label_mask = new_label != -1
            new_label_out = label[label_mask].reshape(h, w)
            labels[i, :] = new_label_out

        labels_torch = torch.from_numpy(labels).cuda().float()
        return labels_torch




class LossFunc(nn.Module):
    def __init__(self, center_R, neg_weight=15.0, device_cuda=None, opt=None):
        super(LossFunc, self).__init__()
        self.center_R = center_R
        self.neg_weight = neg_weight

        self.cls_loss = GaussianLoss(center_R=center_R, neg_weight=self.neg_weight)


    def forward(self, input, center_rate):
        cls_input, loc_input = input
        # calc cls loss
        cls_loss = self.cls_loss(cls_input, center_rate)
        if loc_input is not None:
            loc_loss = self.loc_loss(cls_input, loc_input, center_rate)

            return cls_loss, loc_loss * self.loc_wight
        else:
            return cls_loss, torch.tensor((0))




