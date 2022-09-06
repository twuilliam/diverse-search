import ipdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import pairwise_distances


class MultiNet(nn.Module):
    '''Multitask network'''
    def __init__(self, n_input, tasks, weight_init=False):
        super(MultiNet, self).__init__()
        self.fc_tasks = nn.ModuleList([nn.Linear(n_input, t)
                                       for t in tasks])
        if weight_init:
            self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # # frobenius norm of 1
                n = m.out_features
                m.weight.data.normal_(0, (2. / n) ** 0.5)
                m.weight.data = m.weight.div(m.weight.norm()).data

    def forward(self, x):
        out = [fc(x) for fc in self.fc_tasks]
        return out


class LinearNet(nn.Module):
    '''Linear network'''
    def __init__(self, n_in, n_out):
        super(LinearNet, self).__init__()
        self.fc = nn.Linear(n_in, n_out)

    def forward(self, x):
        return self.fc(x)


class ClassifyNet(nn.Module):
    '''Classification network'''
    def __init__(self, backbone, multi):
        super(ClassifyNet, self).__init__()
        self.backbone = backbone
        self.multi = multi

    def forward(self, x):
        x = self.backbone(x)
        x = self.multi(x)
        return x


class CondNet(nn.Module):
    """Applies a condition mask to the output of a model
    Arguments:
        n_conditions (int): number of different similarity notions
        embedding_size: number of dimensions of the embedding output
        prein: Boolean indicating whether masks are initialized in equally sized disjoint
            sections (i.e. 1 for subspaces, 0.1 for the rest) or random otherwise
    """
    def __init__(self, n_conditions, embedding_size, n_attributes):
        super(CondNet, self).__init__()

        self.n_conditions = n_conditions
        self.step = int(embedding_size / n_conditions)
        self.n_attributes = n_attributes

        # initialize masks
        mask_array = np.zeros([n_conditions, embedding_size])
        for i in range(n_conditions):
            mask_array[i, i * self.step:(i + 1) * self.step] = 1

        # no gradients for the masks
        self.masks = nn.Parameter(
            torch.Tensor(mask_array), requires_grad=False)

        print('Creating %d masks of size %dd' %
              (self.n_conditions, self.step))

    def forward(self, embedded_x, c):
        masked_embedding = embedded_x * self.masks[c]
        return masked_embedding


class ProxyNet(nn.Module):
    """Retrieve proxies"""
    def __init__(self, n_classes, embedding_size, learned_proxies=False):
        super(ProxyNet, self).__init__()
        self.n_classes = n_classes
        self.embedding_size = embedding_size

        self.proxies = nn.Embedding(n_classes, embedding_size,
                                    scale_grad_by_freq=False)

        self.proxies.weight = nn.Parameter(
            torch.randn(self.n_classes, self.embedding_size),
            requires_grad=learned_proxies)

        self.normalize_proxies()

    def normalize_proxies(self):
        self.proxies.weight.data /= self.proxies.weight.data.norm(p=2, dim=1)[:, None]

    def forward(self, y_true):
        proxies_y_true = self.proxies(y_true)
        return proxies_y_true


class MultiProxyNet(nn.Module):
    def __init__(self, nets):
        super(MultiProxyNet, self).__init__()
        self.embs = nn.ModuleList(nets)

    def forward(self, cond, adjs):
        x = torch.cat([self.embs[c](Variable(torch.LongTensor([a])).cuda())
                       for c, a in zip(cond, adjs)])
        Z = [self.embs[c].proxies.weight for c in cond]
        return x, Z

    def forward_single(self, cond):
        # x = self.embs[cond](Variable(adjs))
        Z = self.embs[cond].proxies.weight
        # return x, Z
        return Z


################
# Proxy losses #
################
class ProxyClassifyNet(nn.Module):
    '''Triplet and classification network'''
    def __init__(self, backbone, multi, proxies, multi_gpu=False):
        super(ProxyClassifyNet, self).__init__()
        self.multi = multi
        self.proxies = proxies

        if multi_gpu:
            self.backbone = nn.DataParallel(backbone)
        else:
            self.backbone = backbone

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, y, get_proxies=True):
        # get features
        feat = self.backbone(x)

        # embedding Regularization
        reg_e = feat.norm(p=2, dim=1).mean()

        # classification
        out = self.multi(feat)

        if get_proxies:
            # normalization term
            den = (feat[:, None, :] - self.proxies.proxies.weight).pow(2).sum(2)

            # item proxy loss
            loss = self.criterion(-den, y)

            return out, loss, reg_e
        else:
            return out, reg_e

    def classify(self, x):
        x = self.backbone(x)
        out = self.multi(x)
        return out


class ProxyLoss(nn.Module):
    def __init__(self, model, cond_net,
                 proxy_adj_nets, proxy_noun_net,
                 csm,
                 multigpu=False, mode='concat'):
        super(ProxyLoss, self).__init__()
        if multigpu:
            self.model = nn.DataParallel(model)
        else:
            self.model = model
        self.cond_net = cond_net
        self.proxy_adj_nets = proxy_adj_nets
        self.proxy_noun_net = proxy_noun_net
        self.csm = csm
        self.criterion = nn.CrossEntropyLoss(reduce=False)
        self.mode = mode

    def create_supercenters(self):
        supercenters = []
        for i in range(self.csm.shape[1]):
            idx = self.csm[:, i].nonzero()
            supercenters.append(
                self.proxy_noun_net.proxies.weight[idx, :].mean(dim=0).squeeze())
        return torch.stack(supercenters)

    def compute_sub_loss(self, cond, idx):
        '''compute loss
        idx: indices to pick
        cond: subspace to pick
        '''
        Z = self.proxy_adj_nets.forward_single(cond)

        # mask proxies and the negatives
        ti = torch.from_numpy(np.asarray([cond], dtype=np.int64)).cuda()
        mx = self.cond_net(self.emb_imgs, ti)
        mZ = self.cond_net(Z, ti)

        distance_Z = (mx[:, None, :] - mZ).pow(2).sum(2)
        loss = self.criterion(-distance_Z, Variable(idx))

        # accuracy
        vv, min_idx = torch.min(distance_Z, 1)
        acc = (idx == min_idx.data).cpu().numpy()

        return loss, acc

    def forward(self, images, labels, masks):
        # parsing input
        # labels come in the form ICA
        items = labels[0]
        cats = labels[1]
        attrs = labels[2:]

        # masks come in form CA
        # normally all IC have a label, which is not the case of A
        m_cats = masks[0]
        m_attrs = masks[1:]

        # get embedding
        self.emb_imgs = self.model(images)

        # first, let's handle the different cases of cat and items
        if self.mode == 'concat':
            # cat loss
            i = len(m_attrs)
            loss_cp, acc_cp = self.compute_sub_loss(i, cats)
            acc_cp = np.mean(acc_cp)

            # item loss
            i = len(m_attrs) + 1
            loss_ip, acc_ip = self.compute_sub_loss(i, items)
            acc_ip = np.mean(acc_ip)

        elif self.mode == 'full':
            supercenters = self.create_supercenters()

            # cat loss
            # get noun proxies for samples
            # y_proxies_cat = supercenters[cats, :]

            # normalization term
            diff_cZ = self.emb_imgs[:, None, :] - supercenters
            denominator_cp = -diff_cZ.pow(2).sum(2)

            # cat proxy loss
            loss_cp = self.criterion(denominator_cp, Variable(cats))

            # cat accuracy
            vv, max_idx = torch.max(denominator_cp, 1)
            acc = (cats == max_idx.data).cpu().numpy()
            acc_cp = np.mean(acc)

            # item loss
            # normalization term
            diff_iZ = self.emb_imgs[:, None, :] - self.proxy_noun_net.proxies.weight
            denominator_ip = -diff_iZ.pow(2).sum(2)

            # item proxy loss
            loss_ip = self.criterion(denominator_ip, Variable(items))

            # item accuracy
            vv, max_idx = torch.max(denominator_ip, 1)
            acc = (items == max_idx.data).cpu().numpy()
            acc_ip = np.mean(acc)

        elif self.mode == 'partial':
            # cat loss
            i = len(m_attrs)
            loss_cp, acc_cp = self.compute_sub_loss(i, cats)
            acc_cp = np.mean(acc_cp)

            # item loss
            # normalization term
            diff_iZ = self.emb_imgs[:, None, :-self.cond_net.step] - self.proxy_noun_net.proxies.weight
            denominator_ip = -diff_iZ.pow(2).sum(2)

            # item proxy loss
            loss_ip = self.criterion(denominator_ip, Variable(items))
            # item accuracy
            vv, max_idx = torch.max(denominator_ip, 1)
            acc = (items == max_idx.data).cpu().numpy()
            acc_ip = np.mean(acc)

        elif self.mode == 'catasattr':
            # cat loss
            i = len(m_attrs)
            loss_cp, acc_cp = self.compute_sub_loss(i, cats)
            acc_cp = np.mean(acc_cp)

            # item loss
            # normalization term
            diff_iZ = self.emb_imgs[:, None, :] - self.proxy_noun_net.proxies.weight
            denominator_ip = -diff_iZ.pow(2).sum(2)

            # item proxy loss
            loss_ip = self.criterion(denominator_ip, Variable(items))
            # item accuracy
            vv, max_idx = torch.max(denominator_ip, 1)
            acc = (items == max_idx.data).cpu().numpy()
            acc_ip = np.mean(acc)

        # now let's focus on attributes
        # attr loss
        loss_mm = []
        acc_mm = []
        count = []
        # norm_factor = Variable(torch.sum(torch.stack(m_attrs, dim=1), dim=1))
        norm_factor = torch.sum(torch.stack(m_attrs, dim=1), dim=1)
        for i in range(len(m_attrs)):
            attr = attrs[i]
            # m_attr = Variable(m_attrs[i])
            m_attr = m_attrs[i]

            # if m_attr.data.sum() != 0:

            loss_m, acc_m = self.compute_sub_loss(i, attr)

            # masking
            loss_m_norm = loss_m / (norm_factor + 1e-8)
            loss_mm.append(m_attr * loss_m_norm)
            count.append(m_attr.data.sum())
            acc_mm.append(np.sum(m_attr.data.cpu().numpy() * acc_m))
            # acc_mm.append(np.sum(acc_m) / m_attr.data.sum())

        loss_mm = sum(loss_mm)
        # acc_mm = np.mean(acc_mm)
        acc_mm = np.sum(acc_mm) / np.sum(count)

        ##################
        # Regularization #
        ##################

        # embedding Regularization
        reg_e = self.emb_imgs.norm(p=2, dim=1).mean()

        return([loss_ip.mean(), loss_cp.mean(), loss_mm.mean()],
               [acc_ip, acc_cp, acc_mm], [reg_e])

    def compute_sub_scores(self, cond, idx):

        Z = self.proxy_adj_nets.forward_single(cond)

        # mask proxies and the negatives
        ti = torch.from_numpy(np.asarray([cond], dtype=np.int64)).cuda()
        mx = self.cond_net(self.emb_imgs, ti)
        mZ = self.cond_net(Z, ti)

        distance_Z = (mx[:, None, :] - mZ).pow(2).sum(2)

        # probs = torch.randn(10, 5)
        softmax = nn.Softmax(dim=1)
        probs = softmax(-distance_Z)

        _, max_idx = torch.max(probs, 1)
        one_hot = torch.FloatTensor(probs.shape)
        one_hot.zero_()
        one_hot[range(probs.shape[0]), max_idx.data.cpu().numpy()] = 1

        return probs.data.cpu().numpy(), one_hot.numpy()

    def binary_pred(self, images, labels, masks):
        # parsing input
        # labels come in the form ICA
        items = labels[0]
        cats = labels[1]
        attrs = labels[2:]

        # masks come in form CA
        # normally all IC have a label, which is not the case of A
        m_cats = masks[0]
        m_attrs = masks[1:]

        # get embedding
        self.emb_imgs = self.model(images)

        # first, let's handle the different cases of cat and items
        if self.mode == 'partial':
            # cat loss
            i = len(m_attrs)
            probs_cp, oh_cp = self.compute_sub_scores(i, cats)

        # now let's focus on attributes
        all_probs = []
        all_oh = []
        # norm_factor = Variable(torch.sum(torch.stack(m_attrs, dim=1), dim=1))
        norm_factor = torch.sum(torch.stack(m_attrs, dim=1), dim=1)
        for i in range(len(m_attrs)):
            attr = attrs[i]
            # m_attr = Variable(m_attrs[i])
            m_attr = m_attrs[i]

            probs_m, oh_m = self.compute_sub_scores(i, attr)

            all_probs.append(probs_m)
            all_oh.append(oh_m)

        # first all attribute scores, then categorical scores
        all_probs.append(probs_cp)
        all_oh.append(oh_cp)
        return np.hstack(all_probs), np.hstack(all_oh)


class CoopProxyLoss(nn.Module):
    def __init__(self, model, cond_net, proxy_noun_net, proxy_adj_nets, csm, multigpu=False):
        super(CoopProxyLoss, self).__init__()
        if multigpu:
            self.model = nn.DataParallel(model)
        else:
            self.model = model
        self.cond_net = cond_net
        self.proxy_noun_net = proxy_noun_net
        self.proxy_adj_nets = proxy_adj_nets
        self.csm = csm
        self.criterion = nn.CrossEntropyLoss(reduce=False)

    def create_supercenters(self):
        supercenters = []
        for i in range(self.csm.shape[1]):
            idx = self.csm[:, i].nonzero()
            supercenters.append(
                self.proxy_noun_net.proxies.weight[idx, :].mean(dim=0).squeeze())
        return torch.stack(supercenters)

    def forward(self, images, labels, masks):
        # parsing input
        items = labels[0]

        cats = labels[1]
        attrs = labels[2:]

        m_cats = masks[0]
        m_attrs = masks[1:]

        # compute common variables
        self.supercenters = self.create_supercenters()
        self.emb_imgs = self.model(images)

        #############
        # item loss #
        #############

        # normalization term
        diff_iZ = self.emb_imgs[:, None, :] - self.proxy_noun_net.proxies.weight
        denominator_ip = -diff_iZ.pow(2).sum(2)

        # item proxy loss
        loss_ip = self.criterion(denominator_ip, Variable(items))

        ############
        # cat loss #
        ############

        # get noun proxies for samples
        y_proxies_cat = self.supercenters[cats, :]

        # normalization term
        diff_cZ = self.emb_imgs[:, None, :] - y_proxies_cat
        denominator_cp = -diff_cZ.pow(2).sum(2)

        # cat proxy loss
        loss_cp = self.criterion(denominator_cp, Variable(cats))

        #################
        # NCA mask loss #
        #################
        # formalities
        loss_mm = []
        norm_factor = torch.sum(torch.stack(m_attrs, dim=1), dim=1)
        for i in range(len(m_attrs)):
            attr = attrs[i]
            m_attr = m_attrs[i]
            ti = torch.from_numpy(np.asarray([i], dtype=np.int64)).cuda()

            # get corresponding proxies and the negatives
            Z = self.proxy_adj_nets.forward_single(i)

            # mask proxies and the negatives
            mx = self.cond_net(self.emb_imgs, ti)
            mZ = self.cond_net(Z, ti)

            # denominator sum_i (exp(-d(x - p_i)))
            distance_Z = mx[:, None, :] - mZ
            m_norm_distance_Z = -distance_Z.pow(2).sum(2)

            # final loss
            loss_m = self.criterion(-m_norm_distance_Z, Variable(attr))

            # masking
            loss_m_norm = loss_m / (norm_factor + 1e-8)
            loss_mm.append(m_attr * loss_m_norm)

        ipdb.set_trace()
        loss_mm = sum(loss_mm)

        ##################
        # Regularization #
        ##################

        # embedding Regularization
        reg_e = self.emb_imgs.norm(p=2, dim=1).mean()

        return([loss_ip.mean(), loss_cp.mean(), loss_mm.mean()], [reg_e])

