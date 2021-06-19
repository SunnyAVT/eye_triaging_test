#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import, print_function, division
from pymba import *
import numpy as np
import cv2
from PIL import Image
import time
import sys
import getopt

def main(argv):
    for arg in sys.argv[1:]:
        print(arg)

    file_name = ""
    image_mode = "L"
    Height = 2487
    Width = 2048

    try:
        """
            options, args = getopt.getopt(args, shortopts, longopts=[])

            参数args：一般是sys.argv[1:]。过滤掉sys.argv[0]，它是执行脚本的名字，不算做命令行参数。
            参数shortopts：短格式分析串。例如："hp:i:"，h后面没有冒号，表示后面不带参数；p和i后面带有冒号，表示后面带参数。
            参数longopts：长格式分析串列表。例如：["help", "ip=", "port="]，help后面没有等号，表示后面不带参数；ip和port后面带冒号，表示后面带参数。

            返回值options是以元组为元素的列表，每个元组的形式为：(选项串, 附加参数)，如：('-i', '192.168.0.1')
            返回值args是个列表，其中的元素是那些不含'-'或'--'的参数。
        """
        opts, args = getopt.getopt(argv, "hm:f:", ["help", "mode=", "file="])
    except getopt.GetoptError:
        print('Error: opencv_display_nh2_image.py -m <C/T/L> -f <filename>')
        print('   or: opencv_display_nh2_image.py --mode=<C/T/L> --file=<filename>')
        sys.exit(2)

        # 处理 返回值options是以元组为元素的列表。
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('opencv_display_nh2_image.py -m <C/T/L> -f <filename>')
            print('or: opencv_display_nh2_image.py --mode=<C/T/L> --file=<filename>')
            sys.exit()
        elif opt in ("-m", "--mode"):
            image_mode = arg
        elif opt in ("-f", "--file"):
            file_name = arg

    '''
    cervical spine image files
    1463 x 1755 x 2 = 5,135,130 bytes, about 5 MB
    lumbar spine image files
    2048 x 2487 x 2 = 10,186,752 bytes, about 10 MB
    '''
    if image_mode is 'C':
        Height = 1755
        Width = 1463
    else:
        Height = 2487
        Width = 2048

    #folder = 'I:/astar/nhanes_database'
    #file_name = folder + "/L" + "00160" + ".nh2"

    print("image_mode is :%s" %image_mode)
    print('Resolution is : %d x %d' %(Width, Height))
    print('file_name is :%s' %file_name)

    #image = np.empty((Height, Width), np.uint16)
    #image.data[:] = open(file_name, 'rb').read()
    #image.data[:] = open(file_name, mode='rb', encoding="utf8").read()

    start_num_str = file_name[-9:-4]
    start_num = int(start_num_str)

    # capture a single frame, more than once if desired
    for i in range(10):
        #file_name[-9:-4] = str(start_num)
        file_name = file_name[:-9] + "%05d"%start_num + file_name[-4:]

        for j in range(10):
            try:
                myfile = open(file_name, 'rb')
                data = myfile.read()
            except FileNotFoundError:
                start_num = start_num + 1
                print("The file " + file_name + " can't find. Try next one")
                file_name = file_name[:-9] + "%05d"%start_num + file_name[-4:]

        print("file name :" + file_name)
        print(len(data))
        image = np.ndarray(buffer=data, dtype=np.uint8, shape=(Height, Width * 2))

        frame_pixel_format = "nh2_Mono12"

        print('File No.{}'.format(file_name))
        camera_frame_size = len(image)
        print("Frame size: %d,  Image Resolution: %dx%d,  pixel_format: %s " % (camera_frame_size, Width, Height, frame_pixel_format))
        data_bytes = image

        if (frame_pixel_format == "Mono8" or frame_pixel_format == "BayerRG8" or frame_pixel_format == "BayerGR8"):
            frame_8bits = np.ndarray(buffer=data_bytes, dtype=np.uint8, shape=(Height, Width))

        elif (frame_pixel_format == "BayerRG12" or frame_pixel_format == "Mono10" or frame_pixel_format == "Mono12" or frame_pixel_format == "Mono14"):
            data_bytes = np.frombuffer(data_bytes, dtype=np.uint8)
            pixel_even = data_bytes[0::2]
            pixel_odd = data_bytes[1::2]

            # Convert bayer16 to bayer8 / Convert Mono12/Mono14 to Mono8
            if (frame_pixel_format == "Mono14"):
                pixel_even = np.right_shift(pixel_even, 6)
                pixel_odd = np.left_shift(pixel_odd, 2)
            elif (frame_pixel_format == "Mono10"):
                pixel_even = np.right_shift(pixel_even, 2)
                pixel_odd = np.left_shift(pixel_odd, 6)
            else:
                pixel_even = np.right_shift(pixel_even, 4)
                pixel_odd = np.left_shift(pixel_odd, 4)
            frame_8bits = np.bitwise_or(pixel_even, pixel_odd).reshape(Height, Width)

        elif (frame_pixel_format == "nh2_Mono12"):
            data_bytes = np.frombuffer(data_bytes, dtype=np.uint8)
            pixel_even = data_bytes[0::2]
            pixel_odd = data_bytes[1::2]
            pixel_even = np.left_shift(pixel_even, 4)
            #pixel_odd = np.left_shift(pixel_odd, 4)
            #pixel_odd = np.left_shift(pixel_odd, 4)
            pixel_odd = np.right_shift(pixel_odd, 4)
            #pixel_even = np.left_shift(pixel_even, 8)
            #pixel_odd1 = np.left_shift(pixel_odd, 4)
            #pixel_odd2 = np.right_shift(pixel_odd, 4)
            frame_8bits = np.bitwise_or(pixel_even, pixel_odd).reshape(Height, Width)
            #frame_8bits = np.bitwise_or(pixel_odd1, pixel_odd2).reshape(Height, Width)

        elif (frame_pixel_format == "BayerRG12Packed" or frame_pixel_format == "Mono12Packed" or frame_pixel_format == "BayerGR12Packed"):
            data_bytes = np.frombuffer(data_bytes, dtype=np.uint8)
            size = len(data_bytes)
            index = []
            for i in range(0, size, 3):
                index.append(i+1)

            data_bytes = np.delete(data_bytes, index)
            frame_8bits = data_bytes.reshape(Height, Width)

        elif (frame_pixel_format == "RGB8Packed" or frame_pixel_format == "BGR8Packed"):
            frame_8bits = np.ndarray(buffer=frame.buffer_data(), dtype=np.uint8, shape=(Height, Width*3))

        else:
            # Note: wait to do -- other format, such as YUV411Packed, YUV422Packed, YUV444Packed
            frame_8bits = np.ndarray(buffer=frame.buffer_data(), dtype=np.uint8, shape=(Height, Width))

        frame_8bits = cv2.resize(frame_8bits, (300, 360))
        #frame_8bits = cv2.resize(frame_8bits, (512, 621))
        #cv2.imshow("Frame_8bits", frame_8bits)
        cv2.imshow("Frame_{}{}".format(image_mode, "%05d"%start_num), frame_8bits)
        k = cv2.waitKey(100)

        if (frame_pixel_format == "BayerRG8" or frame_pixel_format == "BayerRG12" or frame_pixel_format == "BayerRG12Packed"):
            colorImg = cv2.cvtColor(frame_8bits, cv2.COLOR_BAYER_RG2RGB )
            cv2.imshow("Color_Image", colorImg)
        elif (frame_pixel_format == "BayerGR8" or frame_pixel_format == "BayerGR12" or frame_pixel_format == "BayerGR12Packed"):
            colorImg = cv2.cvtColor(frame_8bits, cv2.COLOR_BAYER_GR2RGB )
            cv2.imshow("Color_Image", colorImg)
        elif (frame_pixel_format == "RGB8Packed" or frame_pixel_format == "BGR8Packed"):
            RGBImg = frame_8bits.reshape(Height, Width, 3)
            colorImg = cv2.cvtColor(RGBImg, cv2.COLOR_BGR2RGB)
            cv2.imshow("Color_Image", colorImg)

        start_num = start_num + 1
        k = cv2.waitKey(100)
    k = cv2.waitKey(0)

if __name__ == "__main__":
    main(sys.argv[1:])