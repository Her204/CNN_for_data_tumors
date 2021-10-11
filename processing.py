import tensorflow as tf 
import pandas as pd
import os
import numpy as np
from tensorflow.keras.layers import Conv2D,MaxPool2D, MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

PATH = os.getcwd()

class Processing_data:

    def __init__(self,path,img_h,img_w,
                    epochs,batch_size):
        self.path = path
        self.filesnames = os.listdir(path+"/data_tumors")
        self.files = [os.path.join(path+"/data_tumors",file) for file in self.filesnames]
        self.df = pd.read_csv(os.path.join(path,"preprocessed_data.csv"))
        self.df = self.df.rename(columns={self.df.columns[0]:"image"})
        self.columns = self.df.columns[2:]
        self.img_h = img_h
        self.img_w = img_w
        self.epochs = epochs
        self.batch_size = batch_size
        feature_map = {}
        for elem in list(self.columns):
            feature_map[elem] = tf.io.FixedLenFeature([],tf.int64)
        feature_map["image"] = tf.io.FixedLenFeature([],tf.string)
        self.feature_map = feature_map
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        
    def prepro(self):
        split_ind = int(0.7 * len(self.files))
        split_ind2 = int(0.9 * len(self.files))
        
        train_dir= self.files[:split_ind]
        val_dir = self.files[split_ind:split_ind2]
        test_dir = self.files[split_ind2:]
        return [train_dir,val_dir,test_dir]

    def read_tfrecord(self,example):
        example = tf.io.parse_single_example(example, self.feature_map)
        image = tf.io.decode_jpeg(example["image"], channels=3)
        image = tf.image.resize(image, (self.img_h,self.img_w))
        image = tf.cast(image, tf.float32) / 255.0
        
        label = []
        
        for val in self. columns:
            label.append(example[val])
        
        return image, label

    def load_dataset(self,filesnames):
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        dataset = tf.data.TFRecordDataset(filesnames)
        dataset = dataset.with_options(ignore_order)
        process = Processing_data(self.path,self.img_h,
                                self.img_w,self.epochs,
                                self.batch_size)
        dataset = dataset.map(process.read_tfrecord)
        
        return dataset

    def get_dataset(self,filenames):
        process = Processing_data(self.path,self.img_h,
                                self.img_w,self.epochs,
                                self.batch_size)
        dataset = process.load_dataset(filenames)
        dataset = dataset.shuffle(2048)
        dataset = dataset.prefetch(buffer_size=self.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        
        return dataset
        
    def model(self,name):
        process = Processing_data(self.path,self.img_h,
                                self.img_w,self.epochs,
                                self.batch_size)
        pre = process.prepro()
        train_dataset = process.get_dataset(filenames=pre[0])
        val_dataset = process.get_dataset(filenames=pre[1])
        test_dataset = process.get_dataset(filenames=pre[2])
        
        train_size = sum(1 for _ in
                         tf.data.TFRecordDataset(pre[0]))
        validation_size = sum(1 for _ in
                         tf.data.TFRecordDataset(pre[1]))
        
        epoch_steps = int(np.ceil(train_size/self.batch_size))
        validation_steps = int(np.ceil(validation_size/self.batch_size))
        base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
                                                input_shape=(self.img_h,
                                                        self.img_w,3),
                                                include_top=False,
                                                classes=len(self.columns),
                                                weights="imagenet"
        )
        base_model.trainable = False
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = tf.keras.layers.Dense(len(self.columns))
        #model = Sequential([
        #    base_model,
        #    global_average_layer,
        #    prediction_layer
        #])
        eff = tf.keras.applications.EfficientNetB1(include_top = False,
                         input_shape=(self.img_h,self.img_w,3))
        for layer in eff.layers:
            layer.trainable = False
        checkpoint = ModelCheckpoint("vgg16_1.h5",monitor="val_acc",
                                    verbose=1,save_best_only=True,
                                    save_weights_only=False,
                                    mode="auto",period=1)
        early = EarlyStopping(monitor="val_acc",min_delta=0,
                            patience=20,verbose=1,mode="auto")
        model = Sequential()
        model.add(eff)
        model.add(global_average_layer)
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(14, activation='softmax'))
        model.compile(
            optimizer = Adam(0.0001),
            loss = "binary_crossentropy",
            #metrics = ["accuracy"]
            metrics =[tf.keras.metrics.AUC(name="auc")]
        )

        history = model.fit(train_dataset,
                            epochs=self.epochs,
                            #steps_per_epoch = epoch_steps,
                            batch_size=self.batch_size,
                            callbacks = [checkpoint,early],
                            validation_data = val_dataset,
                            validation_steps = validation_steps
                            )
        model.save(name)

        return model.predict(test_dataset)

preds = Processing_data(PATH,100,100,5,256).model(name="prove.h5")

print(preds)
