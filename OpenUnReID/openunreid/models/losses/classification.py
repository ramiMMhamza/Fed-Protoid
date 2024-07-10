from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb
import scipy
__all__ = ["CrossEntropyLoss", "SoftEntropyLoss","KDLoss","MMD_loss","KLdivLoss"]
import scipy.linalg
import torch

def adjoint(A, E, f):
    A_H = A.T.conj().to(E.dtype)
    n = A.size(0)
    M = torch.zeros(2*n, 2*n, dtype=E.dtype, device=E.device)
    M[:n, :n] = A_H
    M[n:, n:] = A_H
    M[:n, n:] = E
    return f(M)[:n, n:].to(A.dtype)

def logm_scipy(A):
    return torch.from_numpy(scipy.linalg.logm(A.cpu(), disp=False)[0]).to(A.device)

class Logm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        assert A.ndim == 2 and A.size(0) == A.size(1)  # Square matrix
        assert A.dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128)
        ctx.save_for_backward(A)
        return logm_scipy(A)

    @staticmethod
    def backward(ctx, G):
        A, = ctx.saved_tensors
        return adjoint(A, G, logm_scipy)


class CrossEntropyLoss(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
    num_classes (int): number of classes.
    epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

        self.logsoftmax = nn.LogSoftmax(dim=1)
        assert self.num_classes > 0
        

    def forward(self, results, targets):
        """
        Args:
        inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        targets: ground truth labels with shape (num_classes)
        """

        inputs = results["prob"]

        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes

        loss = (-targets * log_probs).mean(0).sum()
        loss_vec = (-targets * log_probs)
        return loss,loss_vec
    

class SoftEntropyLoss(nn.Module):
    def __init__(self):
        super(SoftEntropyLoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, results, results_mean):
        assert results_mean is not None

        inputs = results["prob"]
        targets = results_mean["prob"]

        log_probs = self.logsoftmax(inputs)
        loss = (-self.softmax(targets).detach() * log_probs).mean(0).sum()
        loss_vec = (-self.softmax(targets).detach() * log_probs)
        return loss,loss_vec
    
class KDLoss(nn.Module):
    def __init__(self):
        super(KDLoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.mse = nn.MSELoss( )
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.logm = Logm.apply
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(2048,affine=False)
        # self.log = scipy.linalg.logm()
    # def forward(self, results, results_mean):
    #     assert results_mean is not None
    #     T = 2
    #     inputs = results
    #     targets = results_mean

    #     log_probs = self.logsoftmax(inputs)
    #     loss = (-self.softmax(targets).detach() * log_probs).mean(0).sum()
    #     loss = nn.KLDivLoss()(self.logsoftmax(inputs/T),
    #                          self.softmax(targets/T))*(T*T)
    #     loss_vec = (-self.softmax(targets).detach() * log_probs)
        
    #     return loss,loss_vec
    def forward(self, results, targets):
        """
        Args:
        inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        targets: ground truth labels with shape (num_classes)
        """

        # targets = (targets - targets.min(1,keepdim=True)[0])/(targets.max(1,keepdim=True)[0]-targets.min(1,keepdim=True)[0])
        # results = (results - results.min(1,keepdim=True)[0])/(results.max(1,keepdim=True)[0]-targets.min(1,keepdim=True)[0])
        targets = F.normalize(targets, p=2, dim=1)
        results = F.normalize(results, p=2, dim=1)
        r_student = torch.mm(targets,targets.T)
        r_teacher = torch.mm(results,results.T)
       
        r_student_ = r_student.unsqueeze(0).expand(int(r_student.size(0)), int(r_student.size(0)),int(r_student.size(1)))
        r_teacher_ = r_teacher.unsqueeze(0).expand(int(r_teacher.size(0)), int(r_teacher.size(0)),int(r_teacher.size(1)))
        kd_loss = torch.sqrt(((r_student_-r_teacher_)**2).sum()+0.00001) #/(r_student_-r_teacher_)**2.max()

        return kd_loss

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
    
    def mm2(self, target_features, batch_protos):
        mean_target_features = target_features.mean(dim=0)
        mean_batch_protos = batch_protos.mean(dim=0)

        L2_mean = (mean_target_features-mean_batch_protos)**2
        return L2_mean.sum()
    
    def mm_var2(self, target_features, batch_protos):
        mean_target_features = target_features.mean(dim=0)
        mean_batch_protos = batch_protos[0].mean(dim=0)

        mm2_target_features = (target_features**2).mean(dim=0)
        mm2_batch_protos = batch_protos[1].mean(dim=0) 

        L2_mean = (mean_target_features-mean_batch_protos)**2 +(mm2_target_features-mm2_batch_protos)**2
        return L2_mean.sum()
    
    def guassian_kernel(self, target_features, batch_protos, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(target_features.size()[0])+int(batch_protos.size()[0])
        total = torch.cat([target_features, batch_protos], dim=0)
        
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 

        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, target_features, batch_protos,kernel='gaussian'):
        if kernel == 'gaussian':
            batch_size = int(target_features.size()[0])
            kernels = self.guassian_kernel(target_features, batch_protos, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = kernels[:batch_size, :batch_size]
            YY = kernels[batch_size:, batch_size:]
            XY = kernels[:batch_size, batch_size:]
            YX = kernels[batch_size:, :batch_size]
            loss = torch.mean(XX + YY - XY -YX)
        elif kernel == 'mm2':
            loss = self.mm2(target_features, batch_protos)
        elif kernel == 'mm_var2':
            loss = self.mm_var2(target_features, batch_protos)
        else:
            assert False
        return loss
    

  
class KLdivLoss(nn.Module):
    def __init__(self):
        super(KLdivLoss, self).__init__()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.logsoftmax = nn.LogSoftmax(dim=1)
    def forward(self, results, targets):
        """
        Args:
        inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        targets: ground truth labels with shape (num_classes)
        """
        # print(results.size())
        # print(targets.size())
        # print(torch.mm(results,results.T).size())
        # print(self.cos(results,results))
        print(targets)
        print(self.logsoftmax(targets))
        output = self.kl_loss(self.logsoftmax(results), targets)

        return output

class KDLoss_2(nn.Module):
    def __init__(self, margin=0,Ttype='absolute'):
        super(KDLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.Ttype = Ttype

    def triplet_loss(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        dist_p = torch.mean(dist_ap).data[0]
        dist_n = torch.mean(dist_an).data[0]
        return loss, prec, dist_p, dist_n,dist_ap, dist_an, dist
    def forward(self,embed_feat_S,embed_feat_T,labels):
        loss_net, inter_, dist_ap, dist_an, dis_pos, dis_neg,dis = self.triplet_loss(embed_feat_S, labels)
        loss_net_T, inter_T, dist_ap_T, dist_an_T,dis_pos_T, dis_neg_T,dis_T = self.triplet_loss(embed_feat_T, labels)
        if self.Ttype == 'relative':
                loss_distillation = 0.0*torch.mean(F.pairwise_distance(embed_feat_S,embed_feat_T))
                loss_distillation += torch.mean(torch.norm(dis-dis_T,p=2))
        elif self.Ttype == 'absolute':
            loss_distillation = torch.mean(F.pairwise_distance(embed_feat_S,embed_feat_T))
        return loss_distillation


