from functools import wraps
import random

import torch
from torch.nn import Parameter

class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, md_weights=[], dropout=0, variational=False, md=None, rescale=True):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self.md_weights = md_weights
        self.md = md
        if md is not None:
            self.end_prob, self.min_length = md
        self.rescale = rescale
        self.fixed = False
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return
    
    def linear_prune(self, keep_pct):
        self.keep_pct = keep_pct
        self.fixed = True

    def norm_prune(self, drop_pct):
        hh = getattr(self.module, "weight_hh_l0_raw").chunk(4, 0)
        ih = getattr(self.module, "weight_ih_l0_raw").chunk(4, 0)
        hh = [h.norm(p=1, dim=1) for h in hh]
        ih = [h.norm(p=1, dim=1) for h in ih]
        norms = [h + i for h, i in zip(hh, ih)]
        norms = torch.stack(norms).mean(0)
        _, indices = norms.sort()
        indices = indices[:int(drop_pct * indices.size(0))]
        for name_w in self.md_weights:
            raw_w = getattr(self.module, name_w + "_raw")
            raw_w.data[indices, ...] = 0

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in set(self.md_weights).union(set(self.weights)):
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda: mask = mask.cuda()
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
            else:
                w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(self.module, name_w, w)
        for i, name_w in enumerate(self.md_weights):
            raw_w = getattr(self.module, name_w + '_raw')
            if name_w in self.weights:
                w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
            else:
                w = raw_w
            if self.training:
                if i == 0:
                    sz = w.size(0) // 4
                    mask = torch.ones(sz)
                    if w.is_cuda:
                        mask = mask.cuda()
                    min_length = int(self.min_length * sz)
                    size = int((sz - self.min_length) / (1 - self.end_prob))
                    end_idx = min(random.randint(min_length, min_length + size - 1), sz)
                    
                    if self.rescale:
                        ones = torch.ones(min_length)
                        arange = torch.arange(1, self.end_prob, -(1 - self.end_prob) / (sz - min_length))
                        rescale = torch.cat([ones, arange])
                        if w.is_cuda:
                            rescale = rescale.cuda()
                        rescale = rescale.repeat(4)
                    mask[end_idx:] = 0
                    mask = mask.repeat(4)
                if self.rescale:
                    rs = rescale
                    if w.dim() == 2:
                        rs = rs.unsqueeze(1)
                        rs = rs.expand_as(w)
                z = mask
                if w.dim() == 2:
                    z = z.unsqueeze(1)
                    z = z.expand_as(w)
                w = w * z
                if self.rescale:
                    w = w / rs
            else:
                w = torch.nn.functional.dropout(raw_w, p=0.1, training=False) # no-op
                if self.fixed:
                    w.data[int(self.keep_pct * w.size(0)):, ...] = 0
            setattr(self.module, name_w, w)

    def forward(self, *args, refresh=True):
        self._setweights()
        return self.module.forward(*args)

if __name__ == '__main__':
    import torch
    from weight_drop import WeightDrop

    # Input is (seq, batch, input)
    x = torch.autograd.Variable(torch.randn(2, 1, 10)).cuda()
    h0 = None

    ###

    print('Testing WeightDrop')
    print('=-=-=-=-=-=-=-=-=-=')

    ###

    print('Testing WeightDrop with Linear')

    lin = WeightDrop(torch.nn.Linear(10, 10), ['weight'], dropout=0.9)
    lin.cuda()
    run1 = [x.sum() for x in lin(x).data]
    run2 = [x.sum() for x in lin(x).data]

    print('All items should be different')
    print('Run 1:', run1)
    print('Run 2:', run2)

    assert run1[0] != run2[0]
    assert run1[1] != run2[1]

    print('---')

    ###

    print('Testing WeightDrop with LSTM')

    wdrnn = WeightDrop(torch.nn.LSTM(10, 10), ['weight_hh_l0'], dropout=0.9)
    wdrnn.cuda()

    run1 = [x.sum() for x in wdrnn(x, h0)[0].data]
    run2 = [x.sum() for x in wdrnn(x, h0)[0].data]

    print('First timesteps should be equal, all others should differ')
    print('Run 1:', run1)
    print('Run 2:', run2)

    # First time step, not influenced by hidden to hidden weights, should be equal
    assert run1[0] == run2[0]
    # Second step should not
    assert run1[1] != run2[1]

    print('---')
