import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
import matplotlib.pyplot as plt
from itertools import chain
from random import sample

import sklearn.model_selection as train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import  Dense, Flatten, Dropout
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score, precision_recall_curve


## Load the NIH data to all_xray_df
all_xray_df = pd.read_csv('/data/Data_Entry_2017.csv')
all_image_paths = {os.path.basename(x): x for x in
                   glob(os.path.join('/data','images*', '*', '*.png'))}
print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
all_xray_df.sample(3)


labels = np.unique(list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
print(labels)

for c_label in labels:
    if len(c_label)>1: # leave out empty labels
        all_xray_df[c_label] = all_xray_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)
all_xray_df.sample(5)

all_xray_df['pneumonia_class'] = all_xray_df['Pneumonia']
all_xray_df.sample(5)


def create_splits(df):


    train_df, valid_df = train_test_split(df,
                                   test_size = 0.2,
                                   stratify = df["pneumonia_class"])

    ## making equal proportions of Pneumonia in both sets!
    # training  data
    p_inds = train_df[train_df['pneumonia_class'] == 1].index.tolist()
    np_inds = train_df[train_df['pneumonia_class'] ==0].index.tolist()

    np_sample = sample(np_inds,len(p_inds))
    train_df = train_df.loc[p_inds + np_sample]

    # validation data
    p_inds = valid_df[valid_df['pneumonia_class']==1].index.tolist()
    np_inds = valid_df[valid_df['pneumonia_class']==0].index.tolist()

    np_sample = sample(np_inds,4*len(p_inds))
    valid_df = valid_df.loc[p_inds + np_sample]

    return train_df, valid_df

train_data, val_data = create_splits(all_xray_df)



def my_image_augmentation():

    augment = ImageDataGenerator(rescale = 1. / 255.0, horizontal_flip = True,
                                vertical_flip = False, height_shift_range = 0.1,
                                width_shift_range = 0.1, rotation_range = 20 ,
                                shear_range = 0.1, zoom_range = 0.1)

    return augment


def make_train_gen():

    augment = my_image_augmentation()
    train_gen = augment.flow_from_dataframe(dataframe=train_data,
                                             directory=None,
                                             x_col = "path",
                                             y_col = "pneumonia_class",
                                             class_mode = 'raw',
                                             target_size = (224, 224),
                                             batch_size = 64
                                             )

    return train_gen


def make_val_gen():

    val_idg = ImageDataGenerator(rescale=1./255.)
    val_gen = val_idg.flow_from_dataframe(dataframe = val_data,
                                             directory=None,
                                             x_col = "path",
                                             y_col = "pneumonia_class",
                                             class_mode = 'raw',
                                             target_size = (224, 224),
                                             batch_size = 64)


    return val_gen

train_gen = make_train_gen()
val_gen = make_val_gen()

valX, valY = val_gen.next()


t_x, t_y = next(train_gen)
fig, m_axs = plt.subplots(4, 4, figsize = (16, 16))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(c_x[:,:,0], cmap = 'bone')
    if c_y == 1:
        c_ax.set_title('Pneumonia')
    else:
        c_ax.set_title('No Pneumonia')
    c_ax.axis('off')


def load_pretrained_model():

    model = VGG16(include_top=True, weights='imagenet')
    transfer_layer = model.get_layer('block5_pool')
    vgg_model = Model(inputs = model.input, outputs = transfer_layer.output)

    for layer in vgg_model.layers[0:17]:
        layer.trainable = False

    return vgg_model

def build_my_model():

    model = Sequential()
    vgg_model = load_pretrained_model()
    model.add(vgg_model)
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))


    return model

model = build_my_model()


from keras.callbacks import EarlyStopping
weight_path="{}model.best.hdf5".format('xray_class')

checkpoint = ModelCheckpoint(weight_path,
                             monitor= 'val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode= "auto",
                             save_weights_only = True)

early = EarlyStopping(monitor= "val_loss",
                      mode= "auto",
                      patience=5)

def scheduler(epoch, lr):
    if epoch < 1:
        return lr
    else:
        return lr * np.exp(-0.1)

lr_scheduler = LearningRateScheduler(scheduler)
callbacks_list = [checkpoint, early, lr_scheduler]



## train your model
from keras.optimizers import Adam
optimizer = Adam(learning_rate=1e-2)
loss = 'binary_crossentropy'
metrics = ['binary_accuracy']
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

history = model.fit_generator(train_gen,
                          validation_data = (valX, valY),
                           epochs = 20,
                            shuffle=1,
                           callbacks = callbacks_list)


weight_path = 'xray_classmodel.best.hdf5'
model.load_weights(weight_path)
pred_Y = model.predict(valX, batch_size = 32, verbose = True)

def plot_auc(t_y, p_y):

    fpr, tpr, threshold = roc_curve(valY, pred_Y)
    roc_auc = auc(fpr, tpr)

    plt.title('Plot AUC')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    return


def plot_prec_rec(val_Y, pred_Y):
    prec, rec, threshold = precision_recall_curve(val_Y, pred_Y)
    plt.title('Plot Precision Recall')
    plt.plot(prec, rec, 'b', label = 'score = %0.2f' % average_precision_score(val_Y,pred_Y))
    plt.legend(loc = 'upper right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()


def plot_history(history):

    # Todo
    n = len(history.history["loss"])
    plt.figure()
    plt.plot(np.arange(n), history.history["loss"], label="train_loss")
    plt.plot(np.arange(n), history.history["val_loss"], label="val_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    return

## plot figures

plot_auc(valY, pred_Y)
plot_prec_rec(valY, pred_Y)
plot_history(history)


## Find the threshold that optimize model's performance,
## and use that threshold to make binary classification.
from sklearn.metrics import recall_score, precision_score
def optimize_f1(t_y, p_y):
    best_threshold = None
    threshold = []
    scores = []
    best_f1 = 0.
    for t in np.arange(0.0,1,0.1):
        pred =  p_y > t
        f1 = f1_score(t_y, pred)
        if f1 > best_f1:
            best_threshold = t
            best_f1 = f1
        threshold.append(t)
        scores.append(f1)
    return best_threshold, best_f1, threshold, scores
best_threshold, best_f1, threshold, scores = optimize_f1(valY, pred_Y)
print("Threshold of %.2f gives best f1 at %.4f"%(best_threshold, best_f1))

# plot F1 scores vs threshold values
plt.figure()
plt.plot(threshold, scores)
plt.title('F1 Scores vs Threshold')
plt.ylabel('F1 Scores')
plt.xlabel('Threshold Values')
plt.plot(best_threshold, best_f1, 'ro')

y_pred = pred_Y > best_threshold
recall = recall_score(valY, y_pred, average='weighted')
print("Threshold of %.2f gives recall score of %.4f"%(best_threshold, recall))
precision = precision_score(valY, y_pred, average='micro')
print("Threshold of %.2f gives a precision score %.4f"%(best_threshold, precision))


YOUR_THRESHOLD = best_threshold
fig, m_axs = plt.subplots(8, 8, figsize = (16, 16))
i = 0
for (c_x, c_y, c_ax) in zip(valX[0:64], valY[0:64], m_axs.flatten()):
    c_ax.imshow(c_x[:,:,0], cmap = 'bone')
    if c_y == 1:
        if pred_Y[i] > YOUR_THRESHOLD:
            c_ax.set_title('1, 1')
        else:
            c_ax.set_title('1, 0')
    else:
        if pred_Y[i] > YOUR_THRESHOLD:
            c_ax.set_title('0, 1')
        else:
            c_ax.set_title('0, 0')
    c_ax.axis('off')
    i=i+1

## save model architecture to a .json:

model_json = model.to_json()
with open("my_model.json", "w") as json_file:
    json_file.write(model_json)

