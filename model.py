import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop

import candle

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, 
            tie_weights=False, binarized=False, collect_stats=False):
        super(RNNModel, self).__init__()
        self.binarized = binarized
        self.collect_stats = collect_stats
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.ctx = ctx = candle.TernaryQuantizeContext()
        self.scale = nn.Parameter(torch.Tensor([0]))
        # self.ternary = ctx.activation(k=8)
        self.encoder = ctx.bypass(nn.Embedding(ntoken, ninp))
        self.md = candle.LinearMarkovDropout(0.7, min_length=0.5, size=ninp)
        if binarized:
            self.decode_bn = ctx.bypass(nn.BatchNorm1d(ninp))
        elif collect_stats:
            self.encode_bn = ctx.moment_stat(name="encoder")
        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        if rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), save_prev_x=True, 
                    zoneout=0, window=2 if l == 0 else 1, output_gate=True, binarized=binarized, ctx=ctx, 
                    collect_stats=collect_stats, scale=self.scale) for l in range(nlayers)]
            for rnn in self.rnns:
                if binarized:
                    rnn.linear.hook_weight(candle.WeightDrop, p=wdrop)
                    # rnn.linear.hook_weight(candle.SignFlip, p=wdrop)
                else:
                    rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
        # print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        # self.decoder = ctx.wrap(nn.Linear(nhid, ntoken), soft=True, scale=self.scale) if binarized else ctx.bypass(nn.Linear(nhid, ntoken))
        self.decoder = ctx.bypass(nn.Linear(nhid, ntoken))

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            if binarized:
                pass
                self.decoder.weight = self.encoder.weight
                # self.decoder.tie_weight(self.encoder.weight)
            else:
                self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

    def norm_prune(self, pct=0.4):
        for rnn in self.rnns:
            candle.prune_qrnn(rnn.linear, 1 - math.sqrt(1 - pct))

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        if not self.binarized:
            self.encoder.weight.data.uniform_(-initrange, initrange)
            self.decoder.bias.data.fill_(0)
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        # if self.binarized:
        #     emb = self.ternary(emb)
        #emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti) # disabled because of Markov dropout
        # emb = self.md(emb.permute(1, 2, 0)).permute(2, 0, 1)

        raw_output = emb
        new_hidden = []
        #raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            # if self.binarized:
            #     raw_output = self.ternary(raw_output)
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                #self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth) # disabled because of Markov dropout
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout) # disabled because of Markov dropout
        outputs.append(output)

        result = output.view(output.size(0)*output.size(1), output.size(2))
        # if self.binarized:
        #     result = self.decode_bn(result)
        result = self.decoder(result)
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result.view(output.size(0), output.size(1), result.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()),
                    Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()))
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
