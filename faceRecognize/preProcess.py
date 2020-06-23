# -*- coding: utf-8 -*-
"""
Created on Mon May 25 23:07:42 2020

@author: litia
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from scipy.io import loadmat, savemat

def visualMat(path):
    for root, dirs, files in os.walk(path):
        for fi in files:
            fullname = fi.split('.')[0]
            data = loadmat(os.path.join(path, fi))
            isTest=  data['isTest']
            fea = data['fea']
            gnd = data['gnd']
            n,_ = fea.shape
            for i in range(n):
                oneimg = fea[i].reshape((64, 64))
                outpath = os.path.join(path, fullname+'\\', str(i)+'_test'+str(int(isTest[i][0]))+'_'+str(gnd[i][0])+'.jpg')
                print(outpath)
                cv2.imwrite(outpath, oneimg)
                print(i)
            print(isTest.shape, fea.shape, gnd.shape)
        
        break

        
if __name__ == "__main__":
    visualMat("E:\\2020-Spring\\IMGProcecss\\Face\\PIE-dataset\\")