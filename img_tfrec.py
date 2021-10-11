import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pandas as pd
import os
#from .processsing import Processing_data

path = os.getcwd()+"/data_tumors"

df = pd.read_csv("preprocessed_data.csv")
df = df.rename(columns={df.columns[0]:"image"})
columns = df.columns[2:]

filesnames = os.listdir(path)
files = [os.path.join(path,image) for image in filesnames]

split_ind = int(0.7 * len(filesnames))
split_ind2 = int(0.9 * len(filesnames))

train_dir = files[:split_ind]
val_dir = files[split_ind:split_ind2]
test_dir = files[split_ind2:]

feature_map= {}
for elem in list(df.columns)[1:]:
    feature_map[elem] = tf.io.FixedLenFeature([], tf.int64)   
feature_map['image'] = tf.io.FixedLenFeature([], tf.string)

def read_tfrecord(example):
    example = tf.io.parse_single_example(example, feature_map)
    image = tf.io.decode_jpeg(example["image"], channels=3)
    image = tf.image.resize(image, (150,150))
    image = tf.cast(image, tf.float32) / 255.0
    
    label = []
    
    for val in columns:
        label.append(example[val])
    
    return image, label

def load_dataset(filenames):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(read_tfrecord)
    
    return dataset

AUTOTUNE = tf.data.experimental.AUTOTUNE
def get_dataset(filenames):
    dataset = load_dataset(filenames)
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(64)
    
    return dataset

train_dataset = get_dataset(train_dir)
valid_dataset = get_dataset(val_dir)
test_dataset = get_dataset(test_dir)

image_viz, label_viz = next(iter(valid_dataset))

def show_batch(X, Y):
    fig =plt.figure(figsize=(20, 20))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(X[n])
        
        result = [x for i, x in enumerate(columns) if Y[n][i]]
        title = "+".join(result)
        
        if result == []: title = "No Finding"
        
        plt.title(title)
        plt.axis("off")
    fig.savefig('good?plot.png')
show_batch(image_viz.numpy(), label_viz.numpy())
