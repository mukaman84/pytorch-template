import torch
import torch.nn as nn

from torch.nn.modules.module import Module
from torch.nn.functional import _Reduction
# from torch_scatter import scatter_add
import numpy as np
import logging

class disc_loss(Module):
    def __init__(self, delta_v=0.5, delta_d=3.0, param_var=1.0, param_dist=1.0, param_reg=0.001, EMBEDDING_FEATS_DIMS=4,
                 image_shape=[512,512]):
        super(disc_loss, self).__init__()
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.param_var = param_var
        self.param_dist = param_dist
        self.param_reg = param_reg
        self.feature_dim = EMBEDDING_FEATS_DIMS
        self.image_shape = image_shape



    def forward(self, inputs, instance_label):
        pix_embedding, y = inputs
        Loss = []
        for idx in range(instance_label.shape[0]):
            Loss.append(self.get_single_disc_loss(pix_embedding[idx], instance_label[idx]))
        try:
            Loss= torch.stack(Loss)
        except TypeError:
            print(111)
        Loss=Loss.mean()
        a=instance_label.cpu().numpy()
        return Loss


    def get_single_disc_loss(self,pix_embedding,instance_label):

        t = instance_label.cpu().numpy()

        instance_label = instance_label.view(instance_label.size()[0]*instance_label.size()[0])

        reshaped_pred = pix_embedding.view(self.feature_dim,-1)
        reshaped_pred = torch.transpose(reshaped_pred, 1, 0)

        unique_labels, counts = torch.unique(instance_label,return_counts=True)
        native_counts = counts




        unique_idx = instance_label.nonzero()
        # b = (instance_label==0).nonzero().squeeze()
        unique_idx = unique_idx.squeeze()



        # unique_idx = torch.cat((b,unique_idx),0)

        num_instances = len(unique_labels)

        # calculate instance pixel embedding mean vec
        # segmented_sum = reshaped_pred, unique_idx, num_instances

        # for

        # a = torch.zeros(self.feature_dim,self.feature_dim).cuda()
        # unique_idx = unique_idx.cuda().squeeze()
        # data_scatter = reshaped_pred.sum(dim=1)

        segmented_sum = self.unsorted_segment_sum(data=reshaped_pred, segment_ids=instance_label, num_segments=self.feature_dim)


        # segmented_sum = a.scatter_add(dim=0,index=instance_label.long().cuda(),src=reshaped_pred.float())

        # segmented_sum = segmented_sum.cpu().numpy()

        # segmented_sum = torch.scatter_add(input=reshaped_pred,index= unique_idx, dim=num_instances)

        temp_count = torch.ones(self.feature_dim).cuda()
        temp_count[unique_labels.long()]=counts.cuda().float()
        counts = temp_count

        segmented_sum[0,:] = torch.zeros(self.feature_dim).cuda().float()

        mu = torch.div(segmented_sum, counts.view(-1, 1))

        instance_label = instance_label.cuda().long()

        mu_expand = torch.index_select(mu, dim=0,index=instance_label)

        distance = torch.norm(torch.sub(mu_expand, reshaped_pred), dim=1)
        distance = torch.sub(distance, self.delta_v)
        distance = torch.clamp(distance, min=0.)
        distance = torch.mul(distance,distance)

        # sum_l_var_cluster = torch.tensor(0.).cuda()
        l_var = []
        for idx in range(1,self.feature_dim):
            if idx in unique_labels:
                instance_idx = (instance_label==idx).nonzero().squeeze()
                l_var.append(torch.mean(torch.index_select(distance,dim=0,index=instance_idx)))



        try:
            l_var = torch.stack(l_var)
        except RuntimeError as e:
            return torch.tensor(0.).cuda()
        l_var = torch.mean(l_var)



        # mu_interleaved_rep = tf.tile(mu, [num_instances, 1])

        mu_interleaved_rep = self.tile(mu, dim=0, n_tile=self.feature_dim)
        mu_interleaved_rep = mu_interleaved_rep.view(self.feature_dim * self.feature_dim, self.feature_dim)
        order_index = torch.LongTensor(np.concatenate([np.arange(self.feature_dim) for _ in range(self.feature_dim)]))
        mu_band_rep = torch.index_select(mu, 0, order_index.cuda())



        zero_vector = torch.zeros(1, dtype=torch.float).cuda()
        mu_band_rep = torch.mul(mu_band_rep,torch.ne(mu_interleaved_rep, zero_vector).cuda().float())
        mu_interleaved_rep = torch.mul(mu_interleaved_rep, torch.ne(mu_band_rep, zero_vector).cuda().float())

        b = mu_band_rep.cpu().detach().numpy()
        c = mu_interleaved_rep.cpu().detach().numpy()




        mu_diff = torch.sub(mu_band_rep, mu_interleaved_rep)
        a = mu_diff.cpu().detach().numpy()


        # aa = torch.abs(mu_diff)
        intermediate_tensor = torch.sum(torch.abs(mu_diff), dim=1)

        if intermediate_tensor.sum() != 0.0:
            bool_mask = torch.ne(intermediate_tensor, zero_vector)
            mu_diff_bool = self.boolean_mask(torch.norm(mu_diff,dim=1), bool_mask)




            # mu_norm = torch.norm(mu_diff_bool, axis=1)
            mu_norm = torch.sub(2. * self.delta_d, mu_diff_bool)
            mu_norm = torch.clamp(mu_norm, 0.)
            mu_norm = torch.mul(mu_norm,mu_norm)

            l_dist = mu_norm.mean()
        else:
            l_dist = torch.tensor(0.).cuda()

        l_reg = torch.mean(torch.norm(mu, dim=1))

        param_scale = 1.
        l_var = self.param_var * l_var
        l_dist = self.param_dist * l_dist
        l_reg = self.param_reg * l_reg

        loss = param_scale * (l_var + l_dist + l_reg)
        return loss


    def tile(self, a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        return torch.index_select(a, 0, order_index.cuda())

    def astensor(self, tensor_in, dtype='float'):
        """
        Convert to a PyTorch Tensor.

        Args:
            tensor_in (Number or Tensor): Tensor object

        Returns:
            torch.Tensor: A multi-dimensional matrix containing elements of a single data type.
        """
        dtypemap = {'float': torch.float, 'int': torch.int, 'bool': torch.uint8}
        try:
            dtype = dtypemap[dtype]
        except KeyError:
            print('Invalid dtype: dtype must be float, int, or bool.')
            raise

        tensor = torch.as_tensor(tensor_in, dtype=dtype)
        # Ensure non-empty tensor shape for consistency
        try:
            tensor.shape[0]
        except IndexError:
            tensor = tensor.expand(1)
        return tensor

    def boolean_mask(self, tensor, mask):
        mask = self.astensor(mask).type(torch.ByteTensor).cuda()
        return torch.masked_select(tensor, mask)

    def unsorted_segment_sum(self,data, segment_ids, num_segments):
        """
        Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

        :param data: A tensor whose segments are to be summed.
        :param segment_ids: The segment indices tensor.
        :param num_segments: The number of segments.
        :return: A tensor of same data type as the data argument.
        """
        # assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"
        #
        # segment_ids is a 1-D tensor repeat it to have the same shape as data
        if len(segment_ids.shape) == 1:
            s = torch.prod(torch.tensor(data.shape[1:])).long()
            segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])
        #
        assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

        shape = [num_segments] + list(data.shape[1:])
        tensor = torch.zeros(*shape).cuda().scatter_add(0, segment_ids.cuda().long(), data.cuda().float())
        tensor = tensor.type(data.dtype)

        return tensor

class FocalLoss(Module):
    def __init__(self, gamma=0, alpha=None, size_average=True, img_size=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.img_size = img_size

    def forward(self, inputs, targets):
        x,y=inputs
        loss = 0
        for idx in range(2):
            # if idx==0:
            p = torch.where(torch.eq(targets, idx),y[:, idx, ...], torch.ones_like(y[:, idx, ...]))
            # else:
            #     p = torch.where(torch.eq(targets, 1), y[:, idx, ...], torch.ones_like(y[:, idx, ...]))
            loss += -1 * torch.sum(self.alpha[idx] * torch.pow((1 - p), self.gamma) * torch.log(p))

        # if self.size_average:
        #     return loss / self.img_size
        # else:
        return loss


class SegmentationCELosses(Module):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False, reduce=None):
        super(SegmentationCELosses, self).__init__()

        # if size_average is not None or reduce is not None:
        self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        # else:
        #     self.reduction = reduction

        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def forward(self, logit, target):
        # logit = logit[0]
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

class SegmentationfocalLosses(Module):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        super(SegmentationfocalLosses, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def forward(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss



class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        # logit = logit[0]
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




