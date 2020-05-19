# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:17:57 2020

@author: wyckliffe
"""


import numpy as np
import theano
import theano.tensor as T
import cart_pole_v3_q_learning_with_RBF

T.cxx = []

class SGDRegressor:

    def __init__(self, D) :

        print("Hello welcome to Theano!")
        w = np.random.randn(D) / np.sqrt(D) # weights
        self.w = theano.shared(w)
        self.lr = 10e-2

        X = T.matrix('X')
        Y = T.vector('Y')

        Y_ = X.dot(self.w)
        delta = Y - Y_
        cost = delta.dot(delta)
        grad = T.grad(cost, self.w)
        updates = [(self.w, self.w - self.lr * grad)]

        self.train_ = theano.function(
            inputs=[X,Y],
            updates = updates,
            )

        self.predict_ = theano.function(
            inputs = [X],
            outputs= Y_,
            )

    def partial_fit(self, X, Y) :
        self.train_(X, Y)

    def predict(self, X) :
        return self.predict_(X)


if __name__ == '__main__' :
    cart_pole_v3_q_learning_with_RBF.SGDRegressor = SGDRegressor
    cart_pole_v3_q_learning_with_RBF.main()