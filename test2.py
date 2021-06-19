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

    fold_name = './'

    if arg is not None:
        fold_name = arg
        # check if this is file or folder
        if os.path.isdir(fold_name) is False:
            print('Wrong input filename')
            sys.exit()

    folders = ['bruce', 'chuck', 'benny']
    patientIDFetch(fold_name, folders)

    '''
    g = os.walk(fold_name, topdown=True)
    for path, dir_list, file_list in g:
        print("path:", path)
        for file_name in file_list:
            print(path+file_name)
        print("Seperate --- ")
        for dir_name in dir_list:
            file_name = os.path.join(path, dir_name)
            # wait to implement for future
            print(file_name)

    for file_name in os.listdir(fold_name):
        index = file_name.find('.')
        start_num_str = 0 #file_name[:index]
        start_num = int(start_num_str)
        orig_file_name = fold_name + '/' + file_name
        new_file_name = fold_name + '/' + "%03d" % start_num + '.jpg'
        #print(orig_file_name, " rename to ", new_file_name)
    '''

# exclude_path is a path list to skip
# id_table.csv  -- id, label, ...
# id_images_table.csv -- id, folder, full_name
def patientIDFetch(fold_name, exclude_path):
    g = os.walk(fold_name, topdown=True)
    for path, dir_list, file_list in g:
        print("Current path:", path)
        x_len = len(exclude_path)
        for i in range(x_len):
            if(exclude_path[i] is path):
                break

        for dir_name in dir_list:

            # wait to implement for future
            #  check if the folder has patient ID - ISxxxxx CSxxxxx MSxxxxx
            #  CSxxxxxR is the name of Seed2, need to mark it.
            # append it to id_table.csv -- csv writer function

            # append id and path to id_images_table.csv -- csv writer function
            # id and path will match with the image file later

            directory_name = os.path.join(path, dir_name)
            print(directory_name)


        print("---- show files --- ")
        # we ignore all the files under the current folder
        if path is not fold_name:
            for file_name in file_list:
                full_image_name = os.path.join(path, file_name)
                # search id and path in id_images_table.csv -- csv reader function
                # if match, check the list is empty or not
                # empty --> tmp keep id and patch for next image
                # not empty --> update id_images_table.csv with id, path, full_image_name
                # use re search function for ID match

        print("path %s search end ..." %{path})



    for file_name in os.listdir(fold_name):
        index = file_name.find('.')
        start_num_str = 0 #file_name[:index]
        start_num = int(start_num_str)
        orig_file_name = fold_name + '/' + file_name
        new_file_name = fold_name + '/' + "%03d" % start_num + '.jpg'
        #print(orig_file_name, " rename to ", new_file_name)



if __name__ == "__main__":
    main(sys.argv[1:])