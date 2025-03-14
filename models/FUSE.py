
import torch.nn as nn

class DFIM(nn.Module):
    def __init__(self, planes_high, planes_low, planes_out):
        super(DFIM, self).__init__()
        self.pa = nn.Sequential(
            # nn.Conv2d(planes_low, planes_low//4, kernel_size=1),
            nn.Conv2d(planes_low, planes_low//4, kernel_size=1),
            nn.BatchNorm2d(planes_low//4),
            nn.ReLU(True),

            # nn.Conv2d(planes_low//4, planes_low, kernel_size=1),
            nn.Conv2d(planes_low//4, planes_low, kernel_size=1),
            nn.BatchNorm2d(planes_low),
            nn.Sigmoid(),
        )
        self.plus_conv = nn.Sequential(
            # nn.Conv2d(planes_high, planes_low, kernel_size=1),
            nn.Conv2d(planes_high, planes_low, kernel_size=1),
            nn.BatchNorm2d(planes_low),
            nn.ReLU(True)
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # nn.Conv2d(planes_low, planes_low//4, kernel_size=1),
            nn.Conv2d(planes_low, planes_low//4, kernel_size=1),
            nn.BatchNorm2d(planes_low//4),
            nn.ReLU(True),

            # nn.Conv2d(planes_low//4, planes_low, kernel_size=1),
            nn.Conv2d(planes_low//4, planes_low, kernel_size=1),
            nn.BatchNorm2d(planes_low),
            nn.Sigmoid(),
        )
        self.end_conv = nn.Sequential(
            # nn.Conv2d(planes_low, planes_out, 3, 1, 1),
            nn.Conv2d(planes_low, planes_out, 3, stride=1, padding=1),
            nn.BatchNorm2d(planes_out),
            nn.ReLU(True),
        )
        # self.rlfb = block.RLFB(planes_low, planes_low//4, planes_low)

        # self.fuse_conv = nn.Sequential(
        #     nn.Conv2d(planes_low*3, planes_low, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(planes_low),
        #     nn.ReLU()
        # )

    def forward(self, x_high, x_low):

        # low = torch.add(x_low, x_low2)
        # low = self.rlfb(low)
        # x_low = torch.cat((low, x_low, x_low2), dim=1)
        # x_low = self.fuse_conv(x_low)

        x_high = self.plus_conv(x_high)
        pa = self.pa(x_low)
        ca = self.ca(x_high)

        feat = x_low + x_high
        feat = self.end_conv(feat)
        feat = feat * ca
        feat = feat * pa
        return feat