import torch.nn as nn

from .head import CustomUpsampler


class PN(nn.Module):
    def __init__(self,opt):
        super(PN, self).__init__()

        self.upsampler=CustomUpsampler()



    def forward(self, z,x, opt=None):
        x=x[0:3]
        #z=z[1:4]
        x = x[2]
        # x= self.project_xy(x)#(b,1,96,96)
        cls=self.upsampler(x)

        # cls = F.interpolate(x, size=opt.Satellitehw, mode='nearest')  #
        return cls,None


