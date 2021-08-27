import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, ReLU, Upsample, Sigmoid


class TESTCAE(nn.Module):

    def __init__(self):
        super(TESTCAE, self).__init__()

        self.conv1_en = Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.relu = ReLU()
        self.maxpool_en = MaxPool2d(kernel_size=2, ceil_mode=False)
        self.conv2_en = Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.conv3_en = Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=0)

        self.conv1_de = Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.up_de = Upsample(scale_factor=2)
        self.conv2_de = Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.conv3_de = Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.sig_de = Sigmoid()

    def forward(self, input):
        # encoder
        output = self.conv1_en(input)
        output = self.relu(output)
        output = self.maxpool_en(output)
        output = self.conv2_en(output)
        output = self.relu(output)
        output = self.maxpool_en(output)
        output = self.conv3_en(output)
        output = self.relu(output)
        # decoder
        output = self.conv1_de(output)
        output = self.relu(output)
        output = self.up_de(output)
        output = self.conv2_de(output)
        output = self.relu(output)
        output = self.up_de(output)
        output = self.conv3_de(output)
        output = self.sig_de(output)

        output = output.view(32, 1, 155, 256)
        """
        1245184 = 32 * 1 * 152 * 256
        """
        return output


cae_nn = TESTCAE()
