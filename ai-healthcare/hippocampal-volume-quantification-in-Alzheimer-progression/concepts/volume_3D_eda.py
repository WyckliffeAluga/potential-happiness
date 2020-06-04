# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 21:28:12 2020

@author: wyckl
"""


import pydicom
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil


path = r"data"
dirs = np.array([[(os.path.join(dp, f), pydicom.dcmread(os.path.join(dp, f), stop_before_pixels = True)) for f in files]
                   for dp,_,files in os.walk(path) if len(files) != 0])

# instances
instances = dirs[0]

# series
series = np.unique([inst[1].SeriesInstanceUID for inst in instances])
print(len(series))

# studies
studies = np.unique([inst[1].StudyInstanceUID for inst in instances])
print(len(studies))

# patient IDs
# how many patientys
patient_ids = [inst[1].PatientID for inst in instances]
patients = np.unique(patient_ids)
print(f"Number of patients: {len(patients)}, patient IDs: {patients}")

# hash for all modalities for individual series

series_uids_modality_map = {uid: s[1].Modality for uid in series for s in instances if s[1].SeriesInstanceUID == uid }

# load pixel data
slices_ct = [pydicom.dcmread(inst[0]) for inst in instances \
             if inst[1].SeriesInstanceUID == "1.2.826.0.1.3680043.2.1125.1.45859137663006505718300393375464286"]

slices_mr1 = [pydicom.dcmread(inst[0]) for inst in instances \
             if inst[1].SeriesInstanceUID == list(series_uids_modality_map.items())[4][0]]

slices_mr2 = [pydicom.dcmread(inst[0]) for inst in instances \
             if inst[1].SeriesInstanceUID == list(series_uids_modality_map.items())[9][0]]

plt.figure()
plt.imshow(slices_ct[1].pixel_array, cmap="gray")

plt.figure()
plt.imshow(slices_mr1[1].pixel_array, cmap="gray")

plt.figure()
plt.imshow(slices_mr2[1].pixel_array, cmap="gray")

study_dates = sorted(np.unique([inst[1].StudyDate for inst in instances]))


print(np.unique([inst[1].StudyDate for inst in instances if inst[1].Modality == "CT"]))

slices_odd_mr = [pydicom.dcmread(inst[0]) for inst in instances \
             if inst[1].StudyDate == "20150116"]

print(np.unique([s.SeriesInstanceUID for s in slices_odd_mr]))
print(len(slices_odd_mr))


plt.imshow(slices_odd_mr[5].pixel_array, cmap="gray")


volumes = dict()

for inst in instances:
    sid = inst[1].SeriesInstanceUID
    if (sid not in volumes):
        volumes[sid] = dict()

    volumes[sid]["StudyDate"] = inst[1].StudyDate
    volumes[sid]["Width"] = inst[1].Columns
    volumes[sid]["Height"] = inst[1].Rows
    volumes[sid]["PatientId"] = inst[1].PatientID

    if ("slice_count" not in volumes[sid]):
        volumes[sid]["slice_count"] = 0
    else:
        volumes[sid]["slice_count"] += 1


for _,v in volumes.items():
    print(f"{v['Width']}x{v['Height']}x{v['slice_count']}  ")

