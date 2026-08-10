import tensorflow as tf
import numpy as np
import sys

sys.path.insert(0, '/lustre/lrspec/users/4301/hxe-for-tda/hxetda')
import hxetda
import matplotlib.pyplot as plt

class WeightedHierchicalLoss_hxetda(tf.keras.losses.Loss):
    def __init__(self, mask_list, pathlengths, class_weights, alpha = 0.5, epsilon=1e-10, 
                 name="weighted_hierchical_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.mask_list = mask_list
        self.pathlengths = pathlengths
        self.alpha = alpha
        self.epsilon = epsilon
        self.class_weights = class_weights
    
    def masked_softmax(self, vec, mask):
        """ This produces the conditional probability of each branch from a parent
        denotated as p(c_(h)|c_(h+1)) in the paper"""
        masked_vec = vec * mask
        exps = tf.exp(masked_vec)
        masked_exps = exps * mask
        masked_sums = tf.reduce_sum(masked_exps, axis=1, keepdims=True) + self.epsilon
        final_vec = masked_exps/masked_sums + ((1-mask) * vec)
        return final_vec
    
    def call(self, y_true, y_pred):
        final_sum = 0
        # Convert tensors to float32
        y_pred = tf.cast(y_pred, tf.float32)
        # Set first column to 1.0 (can't do in-place assignment)
        output = tf.concat([tf.ones_like(y_pred[:, :1]), y_pred[:, 1:]], axis=1)
    
        for i,mask in enumerate(self.mask_list):
            output = self.masked_softmax(output, mask)
            
        output = tf.math.log(output)  # this gives us log[p(c_(h)|c_(h+1))]
    
        # this is log[p(c_(h)|c_(h+1))] * λ(c) 
        # λ(c) weights the different levels of the tree
        output = output * np.exp(-self.alpha * (self.pathlengths - 1)) 
    
        # this gives us W(c_h) * {log[p(c_h|c_(h+1))] * λ(c)}
        # W(c_h) weights each class by its class fraction 
        output = output * self.class_weights 

        # finally ({W(c_h) * log[p(c_h|c_(h+1))] * λ(c)} * y_true)
        output = tf.reduce_sum((output * y_true), axis=1)
    
        # finally we get the negative of all of this as the loss
        # I dont know why its the mean and not just the sum but it fundementally doesnt matter
        return -tf.reduce_mean(output)
        
    
    def get_config(self):
        config = {
            'alpha': self.alpha,
            'epsilon': self.epsilon,
            'mask_list': self.mask_list,
            'class_weights': self.sample_weights,
        }
        base_config = super().get_config()
        return {**base_config, **config}

def class_weights_hxetda(y_true):
    """this is going to create the weight vector for classes using the hxetda papers metric
    W(c) =  N_All/(N_Labels * N_c)

    note: their actual implementation does vary from this version"""

    N_All = y_true.sum() # N_All is the total number of events in the dataset
    N_c = y_true.sum(axis = 0) # N_Labels is the number of unique classes
    N_Labels = y_true.shape[1] # N_c is the number of events of class c
    W = np.ones(int(N_Labels)) * N_All / N_c
    W = W / N_Labels
    return W