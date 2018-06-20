# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 09:33:04 2018

@author: PhosF
"""

import numpy as np
import yaml
import cv2

file_name = input('Enter file name:')
    
fs = cv2.FileStorage(file_name, cv2.FILE_STORAGE_READ)

if "intrinsics" in file_name:
   
    fn = fs.getNode("Camera_Matrix") #"camera_matrix" 
    print("Camera_Matrix:")
    print(fn.mat().round(1))
    print('\n')
    fn2 = fs.getNode("Distortion_Coefficients") # "distortion_coefficients"
    print("Distortion_Coefficients:")
    print(fn2.mat().round(6))
    print('\n')

if "Extrinsics" in file_name:
    fs1 = fs.getNode("Stereo_Parameters")
    fn3 = fs1.getNode( "Rotation_Matrix")
    print("Rotation_Matrix:")
    print(fn3.mat().round(6))
    print('\n')
    fn4 = fs1.getNode( "Translation_Vector")
    print("Translation_Vector:")
    print(fn4.mat().round(4))
    print('\n')
    fn5 = fs1.getNode( "Fundamental_Matrix")
    print("Fundamental_Matrix:")
    print(fn5.mat().round(6))
    print('\n')
