#!/usr/bin/env python
# coding: utf-8
# Copyright (C) Shenshu Technologies Co., Ltd. 2022-2022. All rights reserved.
import os
import sys
import getopt
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

def draw_box(imgName, boxTxt):
    boxValue = []
    with open(boxTxt, "r") as f:
        for line in f.readlines():
            line = line.strip('/n')
            temp = line.split()
            boxValue.append(temp)
    imgW = int(boxValue[0][0])
    imgH = int(boxValue[0][1])
    img = Image.open(imgName)
    imgOriW, imgOriH = img.size
    if imgW != imgOriW or imgH != imgOriH:
        img = img.resize((imgW, imgH))
    for i in range(1, len(boxValue)):
        txt = "{}: {}".format(boxValue[i][0], boxValue[i][1])
        d = ImageDraw.ImageDraw(img);
        d.text((int(float(boxValue[i][2])) + 2, int(float(boxValue[i][3])) + 1), txt, fill = 'blue')
        d.rectangle(((int(float(boxValue[i][2])), int(float(boxValue[i][3]))),
                            (int(float(boxValue[i][4])), int(float(boxValue[i][5])))), fill = None, outline = 'blue', width = 3)
    imgName = "out_" + os.path.basename(boxTxt).rsplit('_', 1)[0] + "_" + os.path.basename(imgName)
    img.save(imgName);
    print("image {} write success.".format(imgName))

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hi:t:", ["ifile=", "tfile="])
    except getopt.GetoptError:
        print("drawBox.py -i <image> -t <boxTxt>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-i":
            inputFile = arg
        elif opt == "-t":
            boxTxt = arg
        else:
            print("drawBox.py -i <image> -t <boxTxt>")
    draw_box(inputFile, boxTxt)

if __name__ == "__main__":
    main(sys.argv[1:])