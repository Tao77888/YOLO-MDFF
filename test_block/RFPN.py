import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

class SNI(nn.Module):
    '''
    https://github.com/AlanLi1997/rethinking-fpn
    soft nearest neighbor interpolation for up-sampling
    secondary features aligned
    '''
    def __init__(self, up_f=2):
        super(SNI, self).__init__()
        self.us = nn.Upsample(None, up_f, 'nearest')
        self.alpha = 1/(up_f**2)

    def forward(self, x):
        return self.alpha*self.us(x)


class GSConvE(nn.Module):
    '''
    GSConv enhancement for representation learning: generate various receptive-fields and
    texture-features only in one Conv module
    https://github.com/AlanLi1997/slim-neck-by-gsconv
    '''
    def __init__(self, c1, c2, k=1, s=1, g=1, d=1, act=False):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, d, act)
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_, c_, 3, 1, 1, bias=False),
            nn.Conv2d(c_, c_, 3, 1, 1, groups=c_, bias=False),
            nn.ReLU6()  #gelu:0.773; rule6:0.775
        )

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        y = torch.cat((x1, x2), dim=1)
        # shuffle
        y = y.reshape(y.shape[0], 2, y.shape[1] // 2, y.shape[2], y.shape[3])
        y = y.permute(0, 2, 1, 3, 4)
        return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])


if __name__ == '__main__':

    input = torch.randn(128, 128, 8, 8)
    dsconv = GSConvE(128, 128)
    output = dsconv(input)
    print(output.shape)#woshixuetao