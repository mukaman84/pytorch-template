"""
Copyright 2019 The Mobis AA team. All Rights Reserved.
======================================
base scenario data parsing API
======================================
Author : Dongyul Lee
Issue date : 25, Jan, 2019
ver : 2.0.0

============
Descriptions
============
Packages to the decoder class with aa_layers based on keras, such as CONV, ACTIVATION, DECONV, etc.

============
depedencies
============
tensorflow=2.0-preview
python=2.7.x
=====

"""
import tensorflow as tf
from tensorflow.keras.layers import Input,Conv1D,AveragePooling1D,Flatten,Dense
from tensorflow import keras
from tensorflow.keras.activations import hard_sigmoid




class BsplineModel(keras.Model):
    def __init__(self,img_h,img_w,batch_size):
        super(BsplineModel, self).__init__()
        """
        :param imgshape:

        Here, we define the parameters of Bspline model
        """
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        # self.OutputChannels = OutputChannels

        self.conv_1 = Conv1D(filters=32, kernel_size=3, padding='same', kernel_initializer="he_normal")
        self.conv_2 = Conv1D(filters=32, kernel_size=3, padding='same', kernel_initializer="he_normal")
        self.conv_3 = Conv1D(filters=64, kernel_size=3, padding='same', kernel_initializer="he_normal")
        self.conv_4 = Conv1D(filters=64, kernel_size=3, padding='same', kernel_initializer="he_normal")
        self.Pool_4_1 = AveragePooling1D(pool_size=3, strides=[1])
        self.conv_5 = Conv1D(filters=128, kernel_size=3, padding='same', kernel_initializer="he_normal")
        self.conv_6 = Conv1D(filters=128, kernel_size=3, padding='same', kernel_initializer="he_normal")
        self.conv_7 = Conv1D(filters=256, kernel_size=3, padding='same', kernel_initializer="he_normal")
        self.conv_8 = Conv1D(filters=256, kernel_size=3, padding='same', kernel_initializer="he_normal")
        self.Pool_8_1 = AveragePooling1D(pool_size=3, strides=[1])
        self.flatten = Flatten()
        self.flatten_2 = Flatten()
        self.dense_1 = Dense(1024, activation='relu')
        self.dense_1_2 = Dense(1024, activation='relu')
        self.dense_2 = Dense(2048, activation='relu')
        self.dense_2_2 = Dense(2048, activation='relu')
        self.dense_3 = Dense(1024, activation='relu')
        self.dense_3_2 = Dense(1024, activation='relu')
        self.dense_4 = Dense(512, activation='relu')
        self.dense_4_2 = Dense(512, activation='relu')
        self.dense_5 = Dense(256, activation='relu')
        self.dense_5_2 = Dense(128, activation='relu')
        self.dense_6 = Dense(10, activation='hard_sigmoid')
        self.dense_6_2 = Dense(10, activation='hard_sigmoid')
        # self.dense_7 = hard_sigmoid()
        # self.dense_7_2 = hard_sigmoid()
        # self.max1 = tf.maximum()
        # self.max2 = tf.maximum()




    def call(self,inp):
        in_w = inp[0]
        in_h = inp[1]

        y = self.flatten(in_w)
        z = self.flatten(in_h)
        # y = self.flatten(x)
        # z = self.flatten_2(x)

        y = self.dense_1(y)
        y = self.dense_2(y)
        y = self.dense_3(y)
        y = self.dense_4(y)
        y = self.dense_5(y)
        y = self.dense_6(y)
        # y = self.dense_7(y)

        # y = self.dense_4(y)

        z = self.dense_1_2(z)
        z = self.dense_2_2(z)
        z = self.dense_3_2(z)
        z = self.dense_4_2(z)
        z = self.dense_5_2(z)
        z = self.dense_6_2(z)
        # z = self.dense_7_2(z)
        # z = self.min2(z, 1)
        # z = self.dense_4_2(z)

        # a=self.hard_sigmoid(y)
        # b=self.hard_sigmoid(z)


        return tf.stack((y*self.img_w, z*self.img_h))
        # return [y, z]



