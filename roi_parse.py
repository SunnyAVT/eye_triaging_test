#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import cv2
from PIL import Image
import time
import sys
import getopt
import os
import pylab
import matplotlib.pyplot as plt
from ThresholdingAlgo import thresholding_algo

class rect(object):
    def __init__(self):
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0


# rect has x, y, width, height
# npdata is 2-d array image
def getRowAverageProfile(rect, npdata):
    profile = np.zeros(rect.height, dtype=np.float)
    counts = np.zeros(rect.height)

    for h in range(rect.y, rect.y+rect.height):
        aLine = npdata[h, rect.x:rect.x + rect.width, 0]
        for i in range(0, rect.width):
            if aLine[i] is not None :
                profile[h] += aLine[i]
                counts[h] += 1

    for i in range(0, rect.height):
        profile[i] /= counts[i]

    return profile


def getColumnAverageProfile(rect, npdata):
    profile = np.zeros(rect.width, dtype=np.float)
    counts = np.zeros(rect.width)

    for x in range(rect.x, rect.x + rect.width):
        aLine = npdata[rect.y:rect.y + rect.height, x, 0]

        for i in range(0, rect.height):
            if aLine[i] is not None:
                profile[x] += aLine[i]
                counts[x] += 1

    for i in range(0, rect.width):
        profile[i] /= counts[i]

    return profile

def getPeakHighLowPoint(profile_array):
    peakLow=[]
    peakHigh=[]
    peakHighIndex=[]
    peakLowIndex=[]

    for i in range(1, 255):
        if(profile_array[i]>profile_array[i-1] and profile_array[i]<profile_array[i+1]):
            print("Peak High index: &d, Value: %d" %{i, profile_array[i]} )
            peakHigh.append(profile_array[i])
            peakHighIndex.append(i)

        if (profile_array[i] < profile_array[i - 1] and profile_array[i] > profile_array[i + 1]):
            print("Peak Low index: &d, Value: %d" % {i, profile_array[i]})
            peakLow.append(profile_array[i])
            peakLowIndex.append(i)

    # Need to consider to merge the high/low point for

# the verify process very depends on the application
def HistogramHPeakVerify(signal):
    '''
    H --> get 3 peak close to center w/2
      -- double check it with expert idea, Limit:  one center, width of other 2 should be close w/2 and symmatirc
    V --> Get 2 peak, one for neck, the other for Sacram
      -- double check it with expert idea, Limit:  one top 10% of H, the other 20% bottom of H
    Note to merger the peak based on the experience, one particular function to handle the output of thresholding_algo
    '''

    profile = np.zeros(10, dtype=np.float)


    return profile

def HistogramVPeakVerify(signal):
    '''
    H --> get 3 peak close to center w/2
      -- double check it with expert idea, Limit:  one center, width of other 2 should be close w/2 and symmatirc
    V --> Get 2 peak, one for neck, the other for Sacram
      -- double check it with expert idea, Limit:  one top 10% of H, the other 20% bottom of H
    Note to merger the peak based on the experience, one particular function to handle the output of thresholding_algo
    '''

    profile = np.zeros(10, dtype=np.float)
    return profile

def showPeakPoint(input):
    signal = (input > np.roll(input, 1)) & (input > np.roll(input, -1))
    plt.plot(input)
    plt.plot(signal.nonzero()[0], input[signal], 'ro')
    plt.show()

def main(argv):
    for arg in sys.argv[1:]:
        print(arg)

    file_name = "001.jpg"
    fold_name = './xrays'
    # define the final resize image resolution
    input_h = 4611
    input_w = 2258
    x_w = 448*2
    x_h = 448*4
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

    roi = rect()  # it is full image here
    roi.x = 0
    roi.y = 0
    roi.width = input_w
    roi.height = input_h

    profile_h = getRowAverageProfile(roi, image)
    #print("profile_h= ", profile_h)
    t = np.arange(0, input_h, 1)
    plt.plot(profile_h, t)
    # draw a thick red hline at y=0 that spans the xrange
    l = plt.axhline(linewidth=4, color='r')
    plt.axis([0, 255, input_h, 0])
    plt.show()
    plt.close()

    profile_w = getColumnAverageProfile(roi, image)
    #print("profile_w= ", profile_w)
    t = np.arange(0, input_w, 1)
    plt.plot(t, profile_w)
    l = plt.axhline(y=1, color='b')
    plt.axis([0, input_w, 0, 255])
    plt.show()
    plt.close()

    '''
    1) use np.array image as input to calculate H/V histogram array
    2) call thresholding_algo to get peak of H/V
    H --> get 3 peak close to center w/2
      -- double check it with expert idea, Limit:  one center, width of other 2 should be close w/2 and symmatirc
    V --> Get 2 peak, one for neck, the other for Sacram
      -- double check it with expert idea, Limit:  one top 10% of H, the other 20% bottom of H
    Note to merger the peak based on the experience, one particular function to handle the output of thresholding_algo
    '''
    lag = int(input_h / 100)
    threshold = 5
    influence = 0.0

    # Run algo with settings from above
    y = profile_h
    result = thresholding_algo(y, lag=lag, threshold=threshold, influence=influence)

    # Plot result
    pylab.subplot(211)
    pylab.plot(np.arange(1, len(y) + 1), y)

    pylab.plot(np.arange(1, len(y) + 1),
               result["avgFilter"], color="cyan", lw=2)

    pylab.plot(np.arange(1, len(y) + 1),
               result["avgMean"], color="yellow", lw=2)

    pylab.plot(np.arange(1, len(y) + 1),
               result["avgFilter"] + threshold * result["stdFilter"], color="green", lw=2)

    pylab.plot(np.arange(1, len(y) + 1),
               result["avgFilter"] - threshold * result["stdFilter"], color="green", lw=2)

    pylab.subplot(212)
    pylab.step(np.arange(1, len(y) + 1), result["signals"], color="red", lw=2)
    pylab.ylim(-2, 2)
    pylab.show()


    lag = int(input_w/100)
    threshold = 5
    influence = 0.0

    # Run algo with settings from above
    y = profile_w
    result = thresholding_algo(y, lag=lag, threshold=threshold, influence=influence)

    # Plot result
    pylab.subplot(211)
    pylab.plot(np.arange(1, len(y) + 1), y)

    pylab.plot(np.arange(1, len(y) + 1),
               result["avgFilter"], color="cyan", lw=2)

    pylab.plot(np.arange(1, len(y) + 1),
               result["avgMean"], color="yellow", lw=2)

    pylab.plot(np.arange(1, len(y) + 1),
               result["avgFilter"] + threshold * result["stdFilter"], color="green", lw=2)

    pylab.plot(np.arange(1, len(y) + 1),
               result["avgFilter"] - threshold * result["stdFilter"], color="green", lw=2)

    pylab.subplot(212)
    pylab.step(np.arange(1, len(y) + 1), result["signals"], color="red", lw=2)
    pylab.ylim(-2, 2)
    pylab.show()

    # Try to detect and crop the ROI
    #image = cv2.resize(image, (x_h, x_w))
    image = cv2.resize(image, (x_w, x_h))
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