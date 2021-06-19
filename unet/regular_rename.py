#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import cv2
from PIL import Image
import time
import sys
import getopt
import os


def main(argv):
    for arg in sys.argv[1:]:
        print(arg)

    fold_name = './xrays'

    if arg is not None:
        fold_name = arg
        # check if this is file or folder
        if os.path.isdir(fold_name) is False:
            print('Wrong input filename')
            sys.exit()

    '''
    g = os.walk(fold_name)
    for path, dir_list, file_list in g:
        for dir_name in dir_list:
            file_name = os.path.join(path, dir_name)
            # wait to implement for future
            print(file_name)
    '''
    for file_name in os.listdir(fold_name):
        index = file_name.find('.')
        start_num_str = file_name[:index]
        start_num = int(start_num_str)
        orig_file_name = fold_name + '/' + file_name
        new_file_name = fold_name + '/' + "%03d" % start_num + '.jpg'
        # rename() function
        os.rename(orig_file_name, new_file_name)
        #print(orig_file_name, " rename to ", new_file_name)


if __name__ == "__main__":
    main(sys.argv[1:])