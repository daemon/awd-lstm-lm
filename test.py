import torch
from torch.autograd import Variable

torch.manual_seed(0)

dim = 10000
lin1 = torch.nn.Linear(dim,10)
infeat = torch.FloatTensor(1,dim).fill_(1.0)

infeat = Variable(infeat)

out = lin1(infeat)
out_new = lin1(infeat)

print(out)
print(out_new)
print("sum of abs diff is {}".format((out - out_new).abs().sum()))

lin2 = torch.nn.Linear(dim,100)

out = lin2(infeat)
out_new = lin2(infeat)

print(out)
print(out_new)
print("sum of abs diff is {}".format((out - out_new).abs().sum()))
