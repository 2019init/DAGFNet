import torch
import torch.nn as nn
from basicseg.base_model import Base_model
import copy
from collections import OrderedDict
from basicseg.metric import Binary_metric
import torch.nn.functional as F
import math
from skimage import measure
import numpy as np
from torch.cuda.amp import GradScaler, autocast


class PD_FA():
    def __init__(self):
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target = 0

    def update(self, preds, labels, size):
        preds = preds.cpu().numpy().astype('int64')
        labels = labels.cpu().numpy().astype('int64')

        batch_size = preds.shape[0]

        for b in range(batch_size):
            pred = preds[b, 0, :, :]  # 假设 preds 和 labels 的形状为 (B, 1, H, W)
            label = labels[b, 0, :, :]

            image = measure.label(pred, connectivity=2)
            coord_image = measure.regionprops(image)
            label = measure.label(label, connectivity=2)
            coord_label = measure.regionprops(label)

            self.target += len(coord_label)
            image_area_total = [np.array(region.area) for region in coord_image]
            image_area_match = []

            for i in range(len(coord_label)):
                centroid_label = np.array(list(coord_label[i].centroid))
                for m in range(len(coord_image)):
                    centroid_image = np.array(list(coord_image[m].centroid))
                    distance = np.linalg.norm(centroid_image - centroid_label)
                    area_image = np.array(coord_image[m].area)
                    if distance < 3:
                        image_area_match.append(area_image)
                        del coord_image[m]
                        break

            dismatch = [x for x in image_area_total if x not in image_area_match]
            self.dismatch_pixel += np.sum(dismatch)
            self.all_pixel += size[2] * size[3]  # size should be (B, C, H, W)
            self.PD += len(image_area_match)

    def get(self):
        Final_FA = self.dismatch_pixel / self.all_pixel
        Final_PD = self.PD / self.target
        return Final_PD, float(Final_FA)

    def reset(self):
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target = 0



class Seg_model(Base_model):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.scaler = GradScaler()
        self.init_model()

    def init_model(self):
        self.setup_net()
        self.setup_optimizer()
        self.setup_loss()
        self.setup_metric()
        self.setup_lr_schduler()

    def setup_metric(self):
        self.best_norm_metric = {'epoch':0., 'iou':0., 'net': None}
        self.best_mean_metric = {'epoch':0, 'iou':0., 'net': None}
        self.metric = Binary_metric()
        self.epoch_metric = {}
        self.batch_metric = {}
        
    def get_mean_metric(self, dist=False, reduction='mean'):
        if dist:
            return self.reduce_dict(self.metric.get_mean_result(), reduction)
        else:
            return self.dict_wrapper(self.metric.get_mean_result())

    def get_norm_metric(self, dist=False, reduction='mean'):
        if dist:
            return self.reduce_dict(self.metric.get_norm_result(), reduction)
        else:
            return self.dict_wrapper(self.metric.get_norm_result())

    def get_epoch_loss(self, dist=False, reduction='sum'):
        epoch_loss = copy.deepcopy(self.epoch_loss)
        self.reset_epoch_loss()
        if dist:
            return self.reduce_dict(epoch_loss, reduction)
        else:
            return self.dict_wrapper(epoch_loss)

    def get_batch_loss(self, dist=False, reduction='sum'):
        batch_loss = copy.deepcopy(self.batch_loss)
        self.reset_batch_loss()
        if dist:
            return self.reduce_dict(batch_loss, reduction)
        else:
            return self.dict_wrapper(batch_loss)

    def optimize_one_iter(self, data):
        if self.bd_loss:
            img, mask, dist_map = data
            img, mask, dist_map = img.to(self.device), mask.to(self.device), dist_map.to(self.device)
        else:
            img, mask = data
            img, mask = img.to(self.device), mask.to(self.device)
        pred = self.net(img)
        cur_loss = 0.
        self.optim.zero_grad()
        if not isinstance(pred, (list, tuple)):
            pred = [pred]
        #elif isinstance(pred, tuple):
            #pred = list(pred)  # 将元组转换为列表
        for idx, pred_ in enumerate(pred):
            pred_ = F.interpolate(pred_, mask.shape[2:], mode='bilinear', align_corners=False)
            if idx == 0:
                pred[0] = pred_
            for loss_type, loss_criteria in self.loss_fn.items():
                if loss_type == 'BD_loss':
                    loss = loss_criteria(pred_, dist_map) * self.loss_weight[loss_type][idx]
                else:
                    # # loss1
                     loss = loss_criteria(pred_, mask) * self.loss_weight[loss_type][idx] + \
                            torch.abs(torch.log(self.net.sigma1.clone()*self.net.sigma2.clone()))

                self.epoch_loss[loss_type + '_' + str(idx)] += loss.detach().clone()
                self.batch_loss[loss_type + '_' + str(idx)] += loss.detach().clone()
                cur_loss += loss
        self.optim.zero_grad()
        #cur_loss.backward()
        #在使用backward()函数时传入了create_graph=True参数，这可能会导致参数和梯度之间形成循环引用，进而导致内存泄漏。为避免这种情况，
        # 建议在创建计算图时使用autograd.grad来代替backward()函数，并在需要使用backward()时确保在使用后将参数的.grad字段重置为None
        cur_loss.backward(retain_graph=True)
        # # 使用autograd.grad创建计算图
        # gradients = torch.autograd.grad(cur_loss, parameters, create_graph=True)
        # # 对于每个参数，可以进一步计算梯度
        # for param, gradient in zip(parameters, gradients):
        #     param.grad = gradient
        #
        # # 确保在使用后将参数的grad字段重置为None
        # for param in parameters:
        #     param.grad = None
        #cur_loss.backward(retain_graph=True, create_graph=True)
        self.optim.step()

        with torch.no_grad():
            self.metric.update(pred=pred[0], target=mask)
        #return cur_loss

    def test_one_iter(self, data):
        with torch.no_grad():
            if self.bd_loss:
                img, mask, dist_map = data
                img, mask, dist_map = img.to(self.device), mask.to(self.device), dist_map.to(self.device)
            else:
                img, mask = data
                img, mask = img.to(self.device), mask.to(self.device)
            pred = self.net(img)
            #pred, _, _, _, _ = self.net(img)
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            #elif isinstance(pred, tuple):
                #pred = list(pred)  # 将元组转换为列表
            for idx, pred_ in enumerate(pred):
                pred_ = F.interpolate(pred_, mask.shape[2:], mode='bilinear', align_corners=False)
                if idx == 0:
                    pred[0] = pred_
                for loss_type, loss_criteria in self.loss_fn.items():
                    if loss_type == 'BD_loss':
                        loss = loss_criteria(pred_, dist_map) * self.loss_weight[loss_type][idx]
                    else:
                        # loss1
                        loss = loss_criteria(pred_, mask) * self.loss_weight[loss_type][idx] + \
                                torch.abs(torch.log(self.net.sigma1.clone() * self.net.sigma2.clone()))

                    self.epoch_loss[loss_type + '_' + str(idx)] += loss.detach().clone()
                    self.batch_loss[loss_type + '_' + str(idx)] += loss.detach().clone()
            self.metric.update(pred=pred[0], target=mask)
        return pred[0], mask


