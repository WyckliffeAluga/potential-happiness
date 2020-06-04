# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 19:34:57 2020

@author: wyckliffe
"""


import pydicom
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy.ma as ma
import numpy as np
import os

# load the metadata
path = r"data"
series = np.array([[(os.path.join(dp, f), pydicom.dcmread(os.path.join(dp, f), force=True, stop_before_pixels = True)) for f in files]
                   for dp,_, files in os.walk(path) if len(files) != 0])

# count how many files
instances =[f for l in series for f in l]
print(len(instances))

# count how mnay patients
patient_ids = np.unique([inst[1].PatientID for inst in instances])
print(len(patient_ids))

# how many studies
studies = {}
for s in series:
    studies.setdefault(s[0][1].StudyIntanceUID, []).append(d)

print(len(studies))

[len([st for st in studies.values() if st[0][0][1].PatientID == p]) for p in patient_ids]


# how many series per study
series_per_study = [(len(sr), sr[0][0][1].PatientID) for sr in studies.values()]
series_per_study

# images per series
img_per_series = [len(s) for s in series]
print(img_per_series)

res = {}
spc = {}
thck = {}

for sr in series:
    dcm = sr[0][1]
    key = str(dcm.PixelSpacing)
    spc.setdefault(key, [])
    spc[key].append((dcm.PatientID, dcm.StudyDescription, dcm.StudyDate, dcm.SeriesDescription))

    key = str((dcm.Rows, dcm.Columns))
    res.setdefault(key, [])
    res[key].append((dcm.PatientID, dcm.StudyDescription, dcm.StudyDate, dcm.SeriesDescription))

    key = str(dcm.SliceThickness)
    thck.setdefault(key, [])
    thck[key].append((dcm.PatientID, dcm.StudyDescription, dcm.StudyDate, dcm.SeriesDescription))


seq1 = r"PGBM-003\10-17-1995-MR RCBV SEQUENCE-57198\34911-T1prereg-46949"
t1_slices = [pydicom.dcmread(os.path.join(path, seq1, f)) for f in os.listdir(os.path.join(path, seq1))]
t1_slices.sort(key = lambda inst: int(inst.ImagePositionPatient[2]))

seq2 = r"PGBM-003\10-17-1995-MR RCBV SEQUENCE-57198\36471-FLAIRreg-02052"
flair_slices = [pydicom.dcmread(os.path.join(path, seq2, f)) for f in os.listdir(os.path.join(path, seq2))]
flair_slices.sort(key = lambda inst: int(inst.ImagePositionPatient[2]))



t1 = np.stack([s.pixel_array for s in t1_slices])
flair = np.stack([s.pixel_array for s in flair_slices])

[ipp.ImagePositionPatient for ipp in t1_slices]

[ipp.ImagePositionPatient for ipp in flair_slices]


np.array([ipp.ImageOrientationPatient for ipp in t1_slices]) - np.array([ipp.ImageOrientationPatient for ipp in flair_slices])

plt.imshow((flair+1.0*t2)[9,:,:], cmap="gray")

plt.imshow((0.0*flair+t2)[9,:,:], cmap="gray")