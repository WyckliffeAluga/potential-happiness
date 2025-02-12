# -*- coding: utf-8 -*-

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob

import matplotlib.pyplot as plt
import seaborn as sns

## Load NIH data
all_xray_df = pd.read_csv('/data/Data_Entry_2017.csv')
all_xray_df.sample(3)

## Load 'sample_labels.csv' data for pixel level assessments
sample_df = pd.read_csv('sample_labels.csv')
all_xray_df.sample(3)

ax = sns.countplot(x="View Position", data=all_xray_df)
plt.plot('View Position')

#drop unused columns
all_xray_df = all_xray_df[['Image Index','Finding Labels','Follow-up #','Patient ID','Patient Age','Patient Gender']]
labels = all_xray_df['Finding Labels'].value_counts()[:30]
fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
ax1.bar(np.arange(len(labels))+0.5, labels)
ax1.set_xticks(np.arange(len(labels))+0.5)
_ = ax1.set_xticklabels(labels.index, rotation = 90)

# gender distribution
# drop age > 100
all_xray_df = all_xray_df[all_xray_df['Patient Age']<100]
sns.FacetGrid(all_xray_df,hue='Patient Gender',height=5).map(sns.distplot,'Patient Age').add_legend()
plt.show()

from itertools import chain
labels = np.unique(list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
for c_label in labels:
    if len(c_label)>1: # leave out empty labels
        all_xray_df[c_label] = all_xray_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)
all_xray_df['pneumonia_class'] = all_xray_df['Pneumonia']

sns.countplot(x="Pneumonia", data=all_xray_df)

df = all_xray_df[all_xray_df['Finding Labels'] != 'No Finding']
data=df[df['Pneumonia'] == 1.0]
#data.groupby('Patient Gender')['Pneumonia'].value_counts().plot(kind='bar')
sns.FacetGrid(data,hue='Patient Gender',height=5).map(sns.distplot,'Patient Age').add_legend()
plt.title('Age/Gender distribution of Pneumoina Only Cases')
plt.show()

df = all_xray_df[all_xray_df['Finding Labels'] != 'No Finding']
data=df[df['Pneumonia'] == 0.0]
data.groupby('Patient Gender')['Pneumonia'].value_counts().plot(kind='bar')
plt.title('Gender Distribution of No Pneumonia Cases')

labels = data['Finding Labels'].value_counts()[:30]
fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
ax1.bar(np.arange(len(labels))+0.5, labels)
ax1.set_xticks(np.arange(len(labels))+0.5)
_ = ax1.set_xticklabels(labels.index, rotation = 90)

all_xray_df = pd.read_csv('/data/Data_Entry_2017.csv')
labels = np.unique(list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
for c_label in labels:
    if len(c_label)>1: # leave out empty labels
        all_xray_df[c_label] = all_xray_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)
all_xray_df['pneumonia_class'] = all_xray_df['Pneumonia']

df = all_xray_df[all_xray_df['Finding Labels'] != 'No Finding']
patholist_list = ['Pneumonia', 'Infiltration', 'Edema', 'Effusion', 'Atelectasis','Consolidation', 'Mass',
                 'Nodule','Pleural_Thickening', 'Cardiomegaly', 'Pneumothorax', 'Emphysema', 'Fibrosis' , 'Hernia']
remove_list = ['Image Index','Finding Labels','Follow-up #','Patient ID','Patient Age','Patient Gender', 'Unnamed: 11',
              'OriginalImagePixelSpacing[x', 'y]', 'OriginalImage[Width', 'Height]','View Position', 'pneumonia_class', 'No Finding' ]
data = df[df['Pneumonia'] == 1.0]
data = data.drop(remove_list, axis=1)
plt.figure(figsize=(12, 10))
data[data[patholist_list] == 1.0].count().sort_values(ascending=False).plot(kind='bar')
plt.grid('off')
plt.title('Frequency Distribution Label wise of Each disease that occur with Pneumonia')

#Number of disease per patient
all_xray_df = pd.read_csv('/data/Data_Entry_2017.csv')
labels = np.unique(list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
for c_label in labels:
    if len(c_label)>1: # leave out empty labels
        all_xray_df[c_label] = all_xray_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)
all_xray_df['pneumonia_class'] = all_xray_df['Pneumonia']

df = all_xray_df[all_xray_df['Finding Labels'] != 'No Finding']
df.sample()

sample_df = pd.read_csv('sample_labels.csv')
all_image_paths = {os.path.basename(x): x for x in
                   glob(os.path.join('/data','images*', '*', '*.png'))}
print('Scans found:', len(all_image_paths), ', Total Headers', sample_df.shape[0])
sample_df['path'] = sample_df['Image Index'].map(all_image_paths.get)
sample_df.sample(3)

import numpy as np
random_image = np.random.randint(0, len(sample_df))
image = plt.imread(sample_df["path"][random_image])
plt.imshow(image, cmap='gray')

from skimage import io
from skimage import data

no_finding = sample_df[sample_df['Finding Labels'] == 'No Finding']['path'].get(5600)
pneumonia = sample_df[sample_df['Finding Labels'] == 'Pneumonia' ]['path'].get(1631)
infiltration= sample_df[sample_df['Finding Labels'] == 'Infiltration']['path'].get(5578)
edema = sample_df[sample_df['Finding Labels'] == 'Edema']['path'].get(282)
effusion = sample_df[sample_df['Finding Labels'] == 'Effusion']['path'].get(5423)
atelectasis= sample_df[sample_df['Finding Labels'] == 'Atelectasis']['path'].get(29)
consolidation = sample_df[sample_df['Finding Labels'] == 'Consolidation']['path'].get(5593)
mass= sample_df[sample_df['Finding Labels'] == 'Mass']['path'].get(108)
nodule = sample_df[sample_df['Finding Labels'] == 'Nodule']['path'].get(347)
pleural_thickening = sample_df[sample_df['Finding Labels'] == 'Pleural_Thickening']['path'].get(5289)
cardiomegaly = sample_df[sample_df['Finding Labels'] == 'Cardiomegaly']['path'].get(658)
pneumothorax = sample_df[sample_df['Finding Labels'] == 'Pneumothorax']['path'].get(369)
emphysema = sample_df[sample_df['Finding Labels'] == 'Emphysema']['path'].get(482)
fibrosis = sample_df[sample_df['Finding Labels'] == 'Fibrosis']['path'].get(96)
hernia = sample_df[sample_df['Finding Labels'] == 'Hernia']['path'].get(1546)

images = {'Infiltration':infiltration, 'Edema': edema, 'Effusion' : effusion, 'Atelectasis': atelectasis,
          'Consolidation': consolidation, 'Mass':mass, 'Nodule':nodule,
                 'Pleural_Thickening': pleural_thickening, 'Cardiomegaly':cardiomegaly,
          'Pneumothorax':pneumothorax, 'Emphysema':emphysema, 'Fibrosis':fibrosis , 'Hernia':hernia}

for key, value in images.items():
    # pneumonia image
    pnemonia_image = io.imread(pneumonia)
    mean_intensity = np.mean(pnemonia_image)
    std_intensity = np.std(pnemonia_image)
    new_img = pnemonia_image.copy()
    new_img = (new_img - mean_intensity)/std_intensity
    plt.figure(figsize=(5,5))
    plt.hist(new_img.ravel(), bins = 100, color='black', label='Pneumonia')

    # other image
    image = io.imread(value)
    img_mean = np.mean(image)
    img_std = np.std(image)
    n_img = image.copy()
    n_img = (n_img  - img_mean) / img_std
    plt.hist(n_img.ravel(), bins=100, color='red',label=key)
    plt.title("Pnemonia Mean: " + str(mean_intensity) + " Pneumonia Std: " + str(std_intensity) + ' \n' + key + ' Mean: ' + str(img_mean) + key + ' Std: '+ str(img_std) )
    plt.legend()
    plt.show()

for key, value in images.items():

    fig, axs = plt.subplots(1,4, figsize=(15, 6))
    fig.subplots_adjust(hspace = .5, wspace=.001)

    axs = axs.ravel()
    pnemonia_image = io.imread(pneumonia)
    mean_intensity = np.mean(pnemonia_image)
    std_intensity = np.std(pnemonia_image)
    new_img = pnemonia_image.copy()
    new_img = (new_img - mean_intensity)/std_intensity
    axs[0].imshow(new_img, cmap='gray')
    axs[0].set_title("Pnemonia" + "\n" + "Mean: " + str(np.round(mean_intensity)) +"\n" + "Std: " + str(np.round(std_intensity)))
    axs[3].hist(new_img.ravel(), bins = 100, color='black', label='Pneumonia')

    no_finding_image = io.imread(no_finding)
    mean_intensity = np.mean(no_finding_image)
    std_intensity = np.std(no_finding_image)
    new_img = no_finding_image.copy()
    new_img = (new_img - mean_intensity)/std_intensity
    axs[1].imshow(new_img, cmap='gray')
    axs[1].set_title("No Finding" + "\n" + " Mean: " + str(np.round(mean_intensity)) +"\n" + "Std: " + str(np.round(std_intensity)))
    axs[3].hist(new_img.ravel(), bins = 100, color='blue', label='No Finding')

    # other image
    image = io.imread(value)
    img_mean = np.mean(image)
    img_std = np.std(image)
    n_img = image.copy()
    n_img = (n_img  - img_mean) / img_std
    img_mean = np.round(img_mean)
    img_std = np.round(img_std)
    axs[2].imshow(n_img, cmap='gray')
    axs[2].set_title(key + "\n" +  " Mean: " + str(img_mean) + "\n" + " Std: " + str(img_std))
    axs[3].hist(n_img.ravel(), bins = 100, color='red', label=key)
    axs[3].legend()

pathology = ['No Finding', 'Pneumonia', 'Infiltration', 'Edema', 'Effusion', 'Atelectasis','Consolidation','Mass', 'Nodule',
                 'Pleural_Thickening', 'Cardiomegaly','Pneumothorax', 'Emphysema','Fibrosis','Hernia']

for p in range(0, len(pathology)):
    image_paths = sample_df[sample_df['Finding Labels'] == pathology[p]]['path']
    fig, axs = plt.subplots(1,5, figsize=(10, 6))
    plt.title(pathology[p], loc='center')
    fig.subplots_adjust(hspace = .5, wspace=.001)
    images = dict(list(image_paths.items())[0: 5])
    image_list = []
    mean_list = []
    std_list = []

    for key, value in images.items():
        img = io.imread(value)
        img_mean = np.mean(img)
        img_std = np.std(img)
        new_img = img.copy()
        new_img = (new_img  - img_mean) / img_std
        image_list.append(new_img)
        mean_list.append(np.round(img_mean))
        std_list.append(np.round(img_std))

    for i in range(0, len(image_list)) :
        axs[i].imshow(image_list[i], cmap='gray')
        axs[i].set_title(str(pathology[p]) + '\n' + "Mean : " + str(mean_list[i]) + '\n' + "Std: " + str(std_list[i]))
        axs[i].axis('off')

"""Conclusions

* Presence of random spikes in the image distributions could suggest uneven pixels
* The general shape of the distribution of a disease suggest that the intensity values between -2 to 2 which could make the images rather similar and hard for the model to rpedict
* The means and stds of some of similar disease images vary and this could make the model predic them to be different diseases while they are the same.
* For some diseases like Effusion and Consolidation they some images have almost the same mean and std with No finding images and that could make the model struggle to differentiate No Finding
"""

