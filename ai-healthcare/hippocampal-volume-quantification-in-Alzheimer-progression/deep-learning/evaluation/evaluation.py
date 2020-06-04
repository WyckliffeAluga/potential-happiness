# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 00:24:58 2020

@author: wyckliffe
"""

import numpy as np
import nibabel as nib

if __name__ == '__main__' :

    lbl1 = nib.load("data/spleen1_label_auto.nii.gz").get_fdata()
    lbl2 = nib.load("data/spleen1_label_gt.nii.gz").get_fdata()


    def dsc3d(a,b):
        intersection = np.sum(a*b)
        volumes = np.sum(a) + np.sum(b)

        if volumes == 0:
            return -1

        return 2.*float(intersection) / float(volumes)

    print(f"DSC: {dsc3d(lbl1, lbl2)}")

    def sensitivity(gt,pred):
        # Sens = TP/(TP+FN)
        tp = np.sum(gt[gt==pred])
        fn = np.sum(gt[gt!=pred])

        if fn+tp == 0:
            return -1

        return (tp)/(fn+tp)

    print(f"Sensitivity: {sensitivity(lbl1, lbl2)}")