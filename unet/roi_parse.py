#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import cv2
from PIL import Image
import time
import sys
import getopt
import os
from ThresholdingAlgo import thresholding_algo

def getRowAverageProfile(rect, cal, ip):
    double[]
    profile = new
    double[rect.height];

    int[]
    counts = new
    int[rect.height];

    double[]
    aLine;

    ip.setInterpolate(false);

    for (int x=rect.x; x < rect.x+rect.width; x++)
    {

    aLine = ip.getLine(x, rect.y, x, rect.y + rect.height - 1);

    for (int i=0; i < rect.height; i++)
    {

    if (!Double.isNaN(aLine[i])) {

    profile[i] += aLine[i];

    counts[i] + +;

    }

    }

    }

    for (int i=0; i < rect.height; i++)

    profile[i] /= counts[i];

    if (cal != null)

    xInc = cal.pixelHeight;

    return profile




def getColumnAverageProfile(rect, ip):
    double[]
    profile = new
    double[rect.width];

    int[]
    counts = new
    int[rect.width];

    double[]
    aLine;

    ip.setInterpolate(false);

    for (int y=rect.y; y < rect.y+rect.height; y++) {

        aLine = ip.getLine(rect.x, y, rect.x+rect.width-1, y);

    for (int i=0; i < rect.width; i++) {

    if (!Double.isNaN(aLine[i])) {

    profile[i] += aLine[i];

    counts[i]++;

    }

    }

    }

    for (int i=0; i < rect.width; i++)

        profile[i] /= counts[i];

    return profile


def main(argv):
    for arg in sys.argv[1:]:
        print(arg)

    file_name = "01.jpg"
    fold_name = './xrays'
    # define the final resize image resolution
    input_h = 4611
    input_w = 2258
    x_h = 448*2
    x_w = 448*4
    roi_lt_x = 0
    roi_lr_y = 0
    roi_w = 0
    roi_h = 0
    batch_mode = False

    try:
        opts, args = getopt.getopt(argv, "hf:", ["help", "file="])
    except getopt.GetoptError:
        print('Error: roi_parse.py -f <filename/folder>')
        print('   or: roi_parse.py --file=<filename/folder>')
        print('Warning: Run with default setting')

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('roi_parse.py -f <filename/folder>')
            print('or: roi_parse.py --file=<filename/folder>')
            sys.exit()
        elif opt in ("-f", "--file"):
            file_name = arg
            fold_name = arg
            # check if this is file or folder
            if os.path.isfile(file_name):
                batch_mode = False
            elif os.path.isdir(fold_name):
                batch_mode = True

    try:
        if batch_mode is False :
            img_x = Image.open(file_name)
            input_w, input_h = img_x.size
            #Image._show(img_x)
            image = np.array(img_x)
            # verify the image, display it for a while
            cv2.imshow("origin_image", image)
            cv2.waitKey(200)

            # get the input image size
            print('Input Image Resolution is : %d x %d' % (input_w, input_h))
            print("image shape: ", image.shape)
            # Getting histogram of image
            #print(len(img_x.histogram()))
            #r, g, b = img_x.split()
            #print(len(r.histogram()))
    except:
        print('Wrong input filename')
        sys.exit()

    1) use np.array image as input to calculate H/V histogram array
    2) call thresholding_algo to get peak of H/V
    H --> get 3 peak close to center w/2
      -- double check it with expert idea, Limit:  one center, width of other 2 should be close w/2 and symmatirc
    V --> Get 2 peak, one for neck, the other for Sacram
      -- double check it with expert idea, Limit:  one top 10% of H, the other 20% bottom of H
    Note to merger the peak based on the experience, one particular function to handle the output of thresholding_algo


    # Try to detect and crop the ROI
    image = cv2.resize(image, (x_h, x_w))
    cv2.imshow("label_image", image)
    cv2.waitKey(0)

    cap_image = bgr8_to_jpeg(image, quality=95)
    new_file_name = file_name[:-4] + '_roi.jpg'
    f = open(new_file_name, 'wb')
    f.write(bytearray(cap_image))
    f.close()


    # capture a single frame, more than once if desired
    if batch_mode:
        n = len(os.listdir(fold_name))
        for i in range(n):
            start_num = i + 1
            file_name = fold_name + '/' + "%03d"%start_num + '.jpg'
            # in case the number is not sequence
            for j in range(5):
                try:
                    myfile = open(file_name, 'rb')
                    data = myfile.read()
                except FileNotFoundError:
                    start_num = start_num + 1
                    print("The file " + file_name + " can't find. Try next one")
                    file_name = fold_name + '/' + "%03d" %start_num + '.jpg'

            print("file name :" + file_name)
            print(len(data))
            image = np.ndarray(buffer=data, dtype=np.uint8, shape=(input_h, input_w))

            cv2.imshow("origin_image", image)
            cv2.waitKey(20)

            # must do ROI parse here
            cv2.imshow("ROI_image", image)
            cv2.waitKey(20)

            image = cv2.resize(image, (x_w, x_h))
            cv2.imshow("label_image", image)
            cv2.waitKey(0)

            cap_image = bgr8_to_jpeg(image, quality=95)
            new_file_name = fold_name + '/' + "%03d" % start_num + '.jpg'
            f = open(new_file_name, 'wb')
            f.write(bytearray(cap_image))
            f.close()

    
def bgr8_to_jpeg(value, quality=75):
    return bytes(cv2.imencode('.jpg', value)[1])

if __name__ == "__main__":
    main(sys.argv[1:])