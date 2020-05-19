# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:42:42 2020

@author: wyckliffe
"""



import numpy as np
import tensorflow as tf
import cart_pole_v3_q_learning_with_RBF

class SGDRegressor :

    def __init__ (self, D):

        print("Hello welcome to tf!!! ")
        lr = 10e-2

        # inputs, targets
        self.w = tf.Variable(tf.random.normal(shape=(D, 1), name='w'))
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')

        # maek prediction and cost
        Y_ = tf.reshape(tf.matmul(self.X, self.w), [-1])
        delta = self.Y - Y_
        cost = tf.reduce_sum(delta * delta)

        self.train_ = tf.train.GradientDescentOptimizer(lr).minimize(cost)
        self.predict_ = Y_

        # session initialization
        init = tf.global_variables_initializer()
        self.session = tf.InteractiveSession()
        self.session.run(init)

    def partial_fit(self, X, Y) :

        self.session.run(self.train_ , feed_dict={self.X: X, self.Y : Y})

    def predict_(self, X) :

        return self.session.run(self.predict_, feed_dict={self.X:X})

if __name__ == '__main__' :

    cart_pole_v3_q_learning_with_RBF.SGDRegressor = SGDRegressor
    cart_pole_v3_q_learning_with_RBF.main()


