import torch
import torch.nn as nn

# class ICUClassifier(torch.nn.Module):
#     def __init__(self, alpha=0, beta=1e-4, num_classes=10, feat_dim=512):
#         super(ICUClassifier, self).__init__()
#         self.alpha = alpha
#         self.beta = beta
#         self.num_classes = num_classes
#         self.feat_dim = feat_dim
#         self.ce = torch.nn.CrossEntropyLoss()
#         self.centers = torch.nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
#         self.vars_inv = torch.nn.Parameter(torch.ones(self.num_classes, self.feat_dim).cuda())
#
#     def forward(self, feature, labels):
#         batch_size = feature.size(0) ##
#         # self.centers = torch.nn.Parameter(torch.where(torch.isnan(self.centers), torch.full_like(self.centers, 0), self.centers))
#         # self.vars_inv = torch.nn.Parameter(torch.where(torch.isnan(self.vars_inv), torch.full_like(self.vars_inv, 1), self.vars_inv))
#         YA = torch.mul(self.centers, self.vars_inv)
#         YAA = torch.mul(YA, self.vars_inv)
#         XAAY = torch.matmul(feature, YAA.t())
#         XAAX = torch.sum((torch.matmul(feature ** 2, (self.vars_inv ** 2).t())), dim=1, keepdim=True)
#         YAAY = torch.sum((YA.t())**2, dim=0, keepdim=True)
#
#         # (1, number class)
#         neg_sqr_dist = -0.5 * (XAAX - 2.0 * XAAY + YAAY)
#
#         # log_vars = torch.sum(torch.log(self.vars_inv.t()), dim=0, keepdim=True)  # (1, number class)
#         # XY = torch.matmul(feature, self.centers.t()) # batch size , num class
#         # XX = torch.sum((feature**2), dim=1, keepdim=True) # batch size, 1
#         # YY = torch.sum((self.centers.t())**2, dim=0, keepdim=True) #  1 , num class
#         # neg_sqr_dist = -0.5 * (XX - 2.0 * XY + YY)
#         # neg_sqr_dist = XY
#         if labels is None:
#             psudo_labels = torch.argmax(neg_sqr_dist, dim=1)
#             means_batch = torch.index_select(self.centers, 0, psudo_labels)
#             log_vars = torch.sum(torch.log(self.vars_inv.t()), dim=0, keepdim=True)
#             likelihood_reg_loss = torch.mean((means_batch - feature) ** 2)
#             return neg_sqr_dist + 0.5 * log_vars, likelihood_reg_loss
#
#         means_batch = torch.index_select(self.centers, 0, labels)
#         vars_batch = torch.index_select(self.vars_inv, 0, labels)
#
#         # print(torch.max(self.vars_inv.t()),torch.min(self.vars_inv.t()))
#         log_vars = torch.sum(torch.log(self.vars_inv.t()+1e-10), dim=0, keepdim=True)
#         log_vars = log_vars.expand(batch_size, self.num_classes)
#
#         BETA = log_vars * (1 + self.beta) # log_vars + torch.log((1 + self.beta))
#         one_hot = torch.zeros(neg_sqr_dist.size(), device='cuda')
#         one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
#         intra_margin = (one_hot * BETA) + ((1.0 - one_hot) * log_vars)
#         distance = neg_sqr_dist + 0.5 * intra_margin
#         vars_batch = torch.index_select(self.vars_inv, 0, labels)
#
#         ALPHA = distance * (1 + self.alpha)
#         inter_margin = (one_hot * ALPHA) + ((1.0 - one_hot) * distance)
#         logits_with_margin = inter_margin
#         likelihood_reg_loss = 0.1 * torch.mean(
#             ((means_batch - feature) ** 2).clamp(min=1e-12, max=1e+12)) + 0.1 * torch.mean((torch.sqrt(
#             (((means_batch - feature) ** 2 - 1 / (vars_batch) ** 2) ** 2).clamp(min=1e-12, max=1e+12))))
#         return logits_with_margin, likelihood_reg_loss
class ICUClassifier(torch.nn.Module):
    def __init__(self, alpha=1e-3, num_classes=10, feat_dim=512):
        super(ICUClassifier, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.ce = torch.nn.CrossEntropyLoss()
        self.centers = torch.nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())


    def forward(self, feature, labels):
        batch_size = feature.size(0) ##
        XY = torch.matmul(feature, self.centers.t()) # batch size , num class
        XX = torch.sum((feature**2), dim=1, keepdim=True) # batch size, 1
        YY = torch.sum((self.centers.t())**2, dim=0, keepdim=True) #  1 , num class
        neg_sqr_dist = -0.5 * (XX - 2.0 * XY + YY)

        if labels is None:
            psudo_labels = torch.argmax(neg_sqr_dist, dim=1)
            means_batch = torch.index_select(self.centers, 0, psudo_labels)
            likelihood_reg_loss = torch.mean((means_batch - feature) ** 2)
            return neg_sqr_dist, likelihood_reg_loss

        means_batch = torch.index_select(self.centers, 0, labels)
        one_hot = torch.zeros(neg_sqr_dist.size(), device='cuda')
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        ALPHA = neg_sqr_dist * (1 + self.alpha)
        inter_margin = (one_hot * ALPHA) + ((1.0 - one_hot) * neg_sqr_dist)
        logits_with_margin = inter_margin
        likelihood_reg_loss = 0.1 * torch.mean(
            ((means_batch - feature) ** 2).clamp(min=1e-12, max=1e+12))
        return logits_with_margin, likelihood_reg_loss

def create_model(feat_dim, num_classes=1000, stage1_weights=False, dataset=None, log_dir=None, test=False, use_effect=True, num_head=None, tau=None, alpha=None, gamma=None, *args):
    #print('Loading Causal Norm Classifier with use_effect: {}, num_head: {}, tau: {}, alpha: {}, gamma: {}.'.format(str(use_effect), num_head, tau, alpha, gamma))
    clf = ICUClassifier(num_classes = num_classes, feat_dim = feat_dim)

    return clf