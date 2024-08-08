import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
# from geomloss import SamplesLoss
import cv2
from qpth.qp import QPFunction

class SoftCrossEntropyLoss(nn.Module):
   def __init__(self):
      super().__init__()
      pass

   def forward(self, y_hat, y):
      p = F.log_softmax(y_hat, 1)
      loss = -(y*p).sum() / (y).sum()
      return loss

def info_nce(query, positive_keys, negative_keys, temperature=0.1, reduction='mean'):
    # Cosine between positive pairs
    # positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)
    positive_logit = query @ positive_keys.transpose(-2, -1).mean(1, keepdim=True)

    # Cosine between all query-negative combinations
    negative_logits = query @ negative_keys.transpose(-2, -1)

    # First index in last dimension are the positive samples
    logits = torch.cat([positive_logit, negative_logits], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)

def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[:, :1], device=y_pred.device)
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return neg_loss + pos_loss

def Log_Sum_Exp_Pairwise_Loss(pred, labels): #input of shape N,C
    pos = pred.clone()
    neg = pred.clone()
    pos[labels==0] = 0 
    neg[labels==1] = 0

    exp_sub = torch.exp(neg.unsqueeze(1) - pos.unsqueeze(2)) #N,1,C - N,C,1
    exp_sum = torch.sum(exp_sub, dim=(1,2))/(exp_sub.shape[1] * exp_sub.shape[2])
    loss_op = torch.log(1 + exp_sum)
    return loss_op # N

def masked_feat_dist(f1, f2, mask=None):
    feat_diff = f1 - f2
    pw_feat_dist = torch.norm(feat_diff, p=2, dim=1)
    if mask is not None:
        pw_feat_dist = pw_feat_dist * mask.squeeze(1)
    return torch.mean(pw_feat_dist)

def image_level_contrast(emb, label):
    loss_imc = 0
    emb = F.normalize(emb, eps=1e-6, dim=-1)
    batch_size = emb.shape[0]
    for i in range(batch_size):
        sim_pos = 1e-6
        sim_neg = 1e-6
        neg_list = range(i+1, batch_size)
        valid_pos = 0
        valid_neg = 0

        for j in neg_list:
            if all(torch.eq(label[i], label[j])):
                sim_pos = sim_pos + torch.exp((emb[i]*emb[j]).sum()/0.1)
                valid_pos += 1
            if torch.bitwise_and(label[i].long(), label[j].long()).sum()==0:
                sim_neg = sim_neg + torch.exp((emb[i]*emb[j]).sum()/0.1)
                valid_neg += 1
        if torch.is_tensor(sim_pos) and torch.is_tensor(sim_neg) and valid_neg>valid_pos:
            sim_pos = sim_pos
            sim_neg = sim_pos + sim_neg
            loss_imc = loss_imc -torch.log(sim_pos/sim_neg)
        else:
            del sim_neg
            del sim_pos
    
    loss_imc /= batch_size
    return loss_imc

def image_level_contrast_out(emb, label):
    loss_imc = 0
    emb = F.normalize(emb, dim=-1)
    batch_size = emb.shape[0]
    for i in range(batch_size):
        sim_poss = []
        sim_neg = 1e-6
        neg_list = range(i+1, batch_size)
    
        valid_neg = 0

        for j in neg_list:
            if all(torch.eq(label[i], label[j])):
                sim_poss.append(torch.exp((emb[i]*emb[j]).sum()/0.1))
       
            if torch.bitwise_and(label[i].long(), label[j].long()).sum()==0:
                sim_neg = sim_neg + torch.exp((emb[i]*emb[j]).sum()/0.1)
                valid_neg += 1
        if torch.is_tensor(sim_neg) and valid_neg>(batch_size/2):
            for sim_pos in sim_poss:
                sim_neg = sim_pos + sim_neg
                loss_imc = loss_imc -torch.log(sim_pos/sim_neg)/len(sim_poss)
        else:
            del sim_neg
            del sim_poss
    
    loss_imc
    return loss_imc

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1.0, size_average=True, weight=None):
        #[1,1,3,1,2,1,1,1,1,3,1,2,1,1,1,1,1,1,2,1,3]
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha).cuda()
        
        self.size_average = size_average
        self.weight = weight
    def forward(self, input, target):

        '''
        input of shape (n,c)
        label of shape (n,c)
        pt =
        {
            p    , if y = 1
            1 − p, otherwise
        }
        FL(pt) = −(1 − pt)γ * log(pt)
        '''

        pt = target * input + (1 - target) * (1 - input) 
        focal = -self.alpha * (1. - pt)**self.gamma * torch.log(pt + 1e-9)
        focal = torch.sum(focal, dim=1) #sum over c dim
        focal = focal / torch.sum(self.alpha)
        return torch.mean(focal)

def PixPro(fm1s, fm2s, coord1s, coord2s):
    coord1s = coord1s.long()
    coord2s = coord2s.long()
    batch_cos = 0
    for b in range(fm1s.shape[0]):
        fm1, coord1 = fm1s[b], coord1s[b]
        fm2, coord2 = fm2s[b], coord2s[b]
        fm1 = fm1[:,coord1[0]:coord1[0]+coord1[2], coord1[1]:coord1[1]+coord1[3]]
        fm2 = fm2[:,coord2[0]:coord2[0]+coord2[2], coord2[1]:coord2[1]+coord2[3]]

        batch_cos = batch_cos + torch.mean(F.cosine_similarity(fm1, fm2.detach(), dim=0)) 

    return 1-(batch_cos/fm1s.shape[0])



class FocalCELoss(nn.Module):
    def __init__(self, alpha=2, beta=1, gamma=2, reduction='mean'):
        super(FocalCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction=self.reduction)

        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        focal_loss = F.nll_loss(logpt, targets, reduction=self.reduction)

        loss = self.alpha * ce_loss + self.beta * focal_loss
        return loss


class FocalLoss_DEEPLAB(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss_DEEPLAB, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        

        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class SegFocalLoss(nn.Module):
    def __init__(self, alpha=1, beta=1, gamma=2, reduction='mean'):
        super(FocalCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        
        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        focal_loss = F.nll_loss(logpt, targets, reduction=self.reduction)

        loss = focal_loss
        return loss


class EMD(object):
    def __init__(self):
        # self.aspp = aspp
        pass

    def emd_inference_qpth(self, distance_matrix, weight1, weight2, form='QP', l2_strength=0.0001):
        """
        to use the QP solver QPTH to derive EMD (LP problem),
        one can transform the LP problem to QP,
        or omit the QP term by multiplying it with a small value,i.e. l2_strngth.
        :param distance_matrix: nbatch * element_number * element_number
        :param weight1: nbatch  * weight_number
        :param weight2: nbatch  * weight_number
        :return:
        emd distance: nbatch*1
        flow : nbatch * weight_number *weight_number
        """
        weight1 = F.relu(weight1) + 1e-5
        weight2 = F.relu(weight2) + 1e-5

        weight1 = (weight1 * weight1.shape[-1]) / weight1.sum(1).unsqueeze(1)
        weight2 = (weight2 * weight2.shape[-1]) / weight2.sum(1).unsqueeze(1)

        nbatch = distance_matrix.shape[0]
        nelement_distmatrix = distance_matrix.shape[1] * distance_matrix.shape[2]
        nelement_weight1 = weight1.shape[1]
        nelement_weight2 = weight2.shape[1]

        Q_1 = distance_matrix.reshape(-1, 1, nelement_distmatrix).float()

        if form == 'QP':
            # version: QTQ
            Q = torch.bmm(Q_1.transpose(2, 1), Q_1).float().cuda() + 1e-4 * torch.eye(
                nelement_distmatrix).float().cuda().unsqueeze(0).repeat(nbatch, 1, 1)  # 0.00001 *
            p = torch.zeros(nbatch, nelement_distmatrix).float().cuda()
        elif form == 'L2':
            # version: regularizer
            Q = (l2_strength * torch.eye(nelement_distmatrix).float()).cuda().unsqueeze(0).repeat(nbatch, 1, 1)
            p = distance_matrix.reshape(nbatch, nelement_distmatrix).float()
        else:
            raise ValueError('Unkown form')

        h_1 = torch.zeros(nbatch, nelement_distmatrix).float().cuda()
        h_2 = torch.cat([weight1, weight2], 1).float()
        h = torch.cat((h_1, h_2), 1)

        G_1 = -torch.eye(nelement_distmatrix).float().cuda().unsqueeze(0).repeat(nbatch, 1, 1)
        G_2 = torch.zeros([nbatch, nelement_weight1 + nelement_weight2, nelement_distmatrix]).float().cuda()
        # sum_j(xij) = si
        for i in range(nelement_weight1):
            G_2[:, i, nelement_weight2 * i:nelement_weight2 * (i + 1)] = 1
        # sum_i(xij) = dj
        for j in range(nelement_weight2):
            G_2[:, nelement_weight1 + j, j::nelement_weight2] = 1
        #xij>=0, sum_j(xij) <= si,sum_i(xij) <= dj, sum_ij(x_ij) = min(sum(si), sum(dj))
        G = torch.cat((G_1, G_2), 1)
        A = torch.ones(nbatch, 1, nelement_distmatrix).float().cuda()
        b = torch.min(torch.sum(weight1, 1), torch.sum(weight2, 1)).unsqueeze(1).float()
 
        flow = QPFunction(verbose=-1)(Q, p, G, h, A, b)

        emd_score = torch.sum((1 - Q_1).squeeze() * flow, 1)
        return emd_score, flow.view(-1, nelement_weight1, nelement_weight2)

    def emd_inference_opencv(self, cost_matrix, weight1, weight2):
        # cost matrix is a tensor of shape (M,N)
        # weight vector is of shape (M)
        cost_matrix = cost_matrix.detach().cpu().numpy()

        weight1 = F.relu(weight1) + 1e-5
        weight2 = F.relu(weight2) + 1e-5
        weight1 = (weight1 * (weight1.shape[0] / weight1.sum().item())).view(-1, 1).detach().cpu().numpy()
        weight2 = (weight2 * (weight2.shape[0] / weight2.sum().item())).view(-1, 1).detach().cpu().numpy()

        cost, _, flow = cv2.EMD(weight1, weight2, cv2.DIST_USER, cost_matrix)
        return cost, flow

    def emd_inference_opencv_test(self, distance_matrix, weight1, weight2):
        distance_list = []
        flow_list = []

        for i in range (distance_matrix.shape[0]):
            cost, flow=self.emd_inference_opencv(distance_matrix[i], weight1[i], weight2[i])
            distance_list.append(cost)
            flow_list.append(torch.from_numpy(flow))

        # emd_distance = torch.Tensor(distance_list).cuda().float()
        flow = torch.stack(flow_list, dim=0).cuda().float()

        return flow

    def compute_modified_cost_matrix(self, C, u, v, reg):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        modified_cost_matrix = (-C + u.unsqueeze(1) + v.unsqueeze(0)) / reg
        return modified_cost_matrix

    def sinkhorn_logsumexp(self, cost_matrix, weight1, weight2, reg=1e-1, maxiter=30, momentum=0., grad=True):

        def sinkhorn_iter():
            mu = weight1.squeeze(0)
            nu = weight2.squeeze(0)

            # m, n = cost_matrix.size()
            # mu = torch.FloatTensor(m).fill_(1./m)
            # nu = torch.FloatTensor(n).fill_(1./n)

            # device = cost_matrix.device
            # mu, nu = mu.to(device), nu.to(device)

            u, v = 0. * mu, 0. * nu
            for i in range(maxiter):
                modified_cost_matrix = self.compute_modified_cost_matrix(cost_matrix, u, v, reg)
                u = reg * (torch.log(mu+1e-6) - torch.logsumexp(modified_cost_matrix, dim=1)) + u
                v = reg * (torch.log(nu+1e-6) - torch.logsumexp(modified_cost_matrix.t(), dim=1)) + v

            modified_cost_matrix = self.compute_modified_cost_matrix(cost_matrix, u, v, reg)
            pi = torch.exp(modified_cost_matrix)
            sinkhorn_distance = torch.sum(pi * cost_matrix.detach())/torch.numel(pi)#/(pi.sum())
            return sinkhorn_distance

        if grad:
            sinkhorn_distance = sinkhorn_iter()   
        else:
            with torch.no_grad():
                sinkhorn_distance = sinkhorn_iter()

        return sinkhorn_distance

    def pair_wise_cos(self, x, y, form='CV'):
        # x (N,C,B)
        # y (M,C,B)
        # cos_sim_pairwise = F.cosine_similarity(x, y.unsqueeze(1), dim=-2)
        # cos_sim_pairwise = cos_sim_pairwise.permute((2, 1, 0)) #B,N,M
        cos_sim_pairwise = (x.unsqueeze(0)*y.unsqueeze(1)).sum(2) #M,N,B
        cos_sim_pairwise = cos_sim_pairwise.permute((2, 1, 0)) #B,N,M
        if form == 'QP':
            return cos_sim_pairwise
        else:
            return 1-cos_sim_pairwise
    
    def pair_wise_l2(self, x, y, p=2):
        # x (N,C,B)
        # y (M,C,B)
        cost_matrix = torch.sqrt(torch.sum((torch.abs(x.unsqueeze(0) - y.unsqueeze(1))) ** p, 2)) #M,N,B
        return cost_matrix.permute(2, 1, 0)

    def get_weight_vector(self, A, B):
        'A,B are of shape (B,C,H,W)'
        A = A.reshape(A.shape[0], A.shape[1], -1) #B,C,M
        B = B.reshape(B.shape[0], B.shape[1], -1) #B,C,N

        weight = A.transpose(1,2).bmm(B.mean(-1).unsqueeze(2)) #B,M,1

        return weight.squeeze(2)

    def static_matching(self, crops1, crops2):
        dists = []
        for i in range(len(crops1)):
            n, c, h1, w1 = crops1[i].shape
            _, _, h2, w2 = crops2[i].shape

            rand_long1 = 7
            if w1 < h1:
                target_shape = (int(round(w1 * rand_long1 / h1)), rand_long1)
            else:
                target_shape = (rand_long1, int(round(h1 * rand_long1 / w1)))
            x = F.interpolate(crops1[i], size=target_shape, mode='bilinear', align_corners=True)
            x_flat = x.reshape(n, c, -1).permute(2, 1, 0)

            with torch.no_grad():
                rand_long2 = 7 #np.random.randint(7, 9)
                if w2 < h2:
                    target_shape = (int(round(w2 * rand_long2 / h2)), rand_long2)
                else:
                    target_shape = (rand_long2, int(round(h2 * rand_long2 / w2)))
                y = F.interpolate(crops2[i], size=target_shape, mode='bilinear', align_corners=True)
                y_flat = y.reshape(n, c, -1).permute(2, 1, 0)
  
            cosine_distance_matrix = self.pair_wise_cos(x_flat, y_flat) #B,N,M
            dists.append(cosine_distance_matrix.mean())
        dists.sort()
        return dists[0] + dists[1]

    def dynamic_matching(self, crops1, crops2):
        losses = 0
        # emds = {}
        if len(crops1) == 0:
            return 0
            
        for i in range(len(crops1)):
            batch_crops1 = crops1[i]
            batch_crops2 = crops2[i]
            emds = {}

            for j in range(len(batch_crops1)):
                crop1 = batch_crops1[j]
                n, c, h1, w1 = crop1.shape
                x = crop1
                x_flat = crop1.reshape(n, c, -1).permute(2, 1, 0)

                for k in range(len(batch_crops2)):
                    crop2 = batch_crops2[k]
                    _, _, h2, w2 = crop2.shape
                    y = crop2
                    y_flat = crop2.reshape(n, c, -1).permute(2, 1, 0)

                    with torch.no_grad():
                        cosine_distance_matrix = self.pair_wise_cos(x_flat, y_flat) #B,N,M

                        weight1 = self.get_weight_vector(x, y) #B,N
                        weight2 = self.get_weight_vector(y, x) #B,M
                        emd_score = self.sinkhorn_logsumexp(cosine_distance_matrix.squeeze(0), weight1, weight2, maxiter=10, grad=False)

                    emds[emd_score] = ((x_flat, y_flat), (weight1, weight2))
                #every i
            #all pairs in a batch

        #all batches
            top1_no_grad = sorted(emds.items(), key=lambda x:x[0])[0][1]
            (x_flat, y_flat), (weight1, weight2) = top1_no_grad
            # weight1 = self.get_weight_vector(x, y) #B,N
            # weight2 = self.get_weight_vector(y, x) #B,M
            dist_mat = self.pair_wise_cos(x_flat, y_flat).squeeze(0)
            top1_emd = self.sinkhorn_logsumexp(dist_mat, weight1, weight2, maxiter=10)
            losses = losses + top1_emd
            del emds
        return losses/len(crops1)

    def __call__(self, crops1, crops2, mode='static'):
        if mode == 'static':
            loss = self.static_matching(crops1, crops2)

        elif mode == 'dynamic':
            loss = self.dynamic_matching(crops1, crops2)        
        
        else:
            print('Unrecognised mode!')

        return loss
    

class DiceLoss(nn.Module):
    def __init__(self, num_classes=21, ignore_index=255, smooth=1., reduction='mean'):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs, targets):
        total_loss = 0
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
        inputs = inputs.transpose(1, 2).contiguous()
        targets = targets.view(-1, inputs.size(2))

        for cls in range(self.num_classes):
            dice_target = (targets == cls).float()
            dice_input = inputs[:, cls]
            intersection = (dice_input * dice_target).sum()
            total_loss += 1 - ((2. * intersection + self.smooth) /
                               (dice_input.sum() + dice_target.sum() + self.smooth))

        if self.reduction == 'mean':
            total_loss /= self.num_classes
        return total_loss


class IoULoss(nn.Module):
    def __init__(self, num_classes=21, ignore_index=255, smooth=1., reduction='mean'):
        super(IoULoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs, targets):
        total_loss = 0
        inputs = inputs.argmax(dim=1) # Get the predicted classes

        for cls in range(self.num_classes):
            if cls != self.ignore_index:
                iou_target = (targets == cls).float()
                iou_input = (inputs == cls).float()
                intersection = (iou_input * iou_target).sum()
                union = iou_input.sum() + iou_target.sum() - intersection
                total_loss += 1 - ((intersection + self.smooth) / (union + self.smooth))

        if self.reduction == 'mean':
            total_loss /= (self.num_classes - 1)
        return total_loss

class SoftIoULoss(nn.Module):
    def __init__(self, num_classes=21, ignore_index=255, smooth=1., reduction='mean'):
        super(SoftIoULoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Softmax the inputs
        inputs = F.softmax(inputs, dim=1)
        
        # Create a tensor of one hot encoded targets
        one_hot_targets = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()

        intersection = inputs * one_hot_targets
        union = inputs + one_hot_targets - intersection

        iou = (intersection.sum() + self.smooth) / (union.sum() + self.smooth)
        iou_loss = 1 - iou

        if self.reduction == 'mean':
            iou_loss = iou_loss.mean()

        return iou_loss

class SoftDiceLoss(nn.Module):
    def __init__(self, num_classes=21, ignore_index=255, smooth=1., reduction='mean'):
        super(SoftDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Softmax the inputs
        inputs = F.softmax(inputs, dim=1)

        # Create a tensor of one hot encoded targets
        one_hot_targets = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()

        intersection = inputs * one_hot_targets
        dice_coeff = (2. * intersection.sum() + self.smooth) / (inputs.sum() + one_hot_targets.sum() + self.smooth)

        dice_loss = 1 - dice_coeff

        if self.reduction == 'mean':
            dice_loss = dice_loss.mean()

        return dice_loss

class SpatialContinuityLoss(nn.Module):
    def __init__(self, threshold=0.3, reduction='mean'):
        super(SpatialContinuityLoss, self).__init__()
        self.threshold = threshold
        self.reduction = reduction

    def forward(self, inputs, class_label):
        # Calculate the difference between adjacent pixel-values
        diff_i = torch.abs(inputs[:, :, :, 1:] - inputs[:, :, :, :-1])
        diff_j = torch.abs(inputs[:, :, 1:, :] - inputs[:, :, :-1, :])

        # Only consider differences larger than the threshold
        diff_i = torch.where(diff_i > self.threshold, diff_i, torch.zeros_like(diff_i))
        diff_j = torch.where(diff_j > self.threshold, diff_j, torch.zeros_like(diff_j))

        # Pad to make it have the same shape as inputs
        diff_i = torch.nn.functional.pad(diff_i, (0, 1, 0, 0))  # Pad one column of zeros on the right
        diff_j = torch.nn.functional.pad(diff_j, (0, 0, 0, 1))  # Pad one row of zeros at the bottom

        # Mask out the classes not in class_label
        mask = torch.zeros_like(inputs)
        for i in range(inputs.shape[0]):  # For each sample in the batch
            mask[i, class_label[i].long(), :, :] = 1
        diff_i = diff_i * mask
        diff_j = diff_j * mask

        if self.reduction == 'sum':
            return torch.sum(diff_i) + torch.sum(diff_j)
        elif self.reduction == 'mean':
            return torch.mean(diff_i) + torch.mean(diff_j)
        elif self.reduction == 'none':
            return diff_i + diff_j
        else:
            raise ValueError(f'Reduction {self.reduction} is not supported.')






