#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import cv2
from PIL import Image
import time
import sys
import getopt
import os

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



def main():
    input = torch.randn(1, 3, 3, 3)
    print(input)
    m = nn.AdaptiveAvgPool2d((2, 2))
    output = m(input)
    print(output)
    print(output.shape)

    data = torch.randn(5)
    print(data)
    print(F.softmax(data, dim=-1))
    print(F.softmax(data, dim=0).sum())  # Sums to 1 because it is a distribution!
    print(F.log_softmax(data, dim=0))  # theres also log_softmax

    layer1 = F.softmax
    layer2 = F.log_softmax

    input = np.asarray([2, 3])
    input = torch.Tensor(input)

    output1 = layer1(input, dim=0)
    output2 = layer2(input, dim=0)
    print('output1:', output1)
    print('output2:', output2)



if __name__ == "__main__":
    main()