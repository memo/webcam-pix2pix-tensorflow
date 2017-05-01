#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: memo

Loads a model and runs predictions on it based on JSON metadata
"""


from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import json
import os



# TODO add checks
def get_info_from_dict(model_info, key):
    '''
    get input/output tensor info from dict
    model_info : contents of model_info json
    key : 'input' or 'output'
    '''
    d = model_info[key]
    t_shape = d['shape']
    t_range = d['range']
    t_opname = d['opname']
    t_opname = t_opname if t_opname[-2:]==':0' else t_opname + ':0' # tensorflow needs ':0' at end
    return t_shape, t_range, t_opname



class Predictor:
    def __init__(self, json_path):
        self.json_path = json_path
        
        with open(json_path) as f:    
            model_info = json.load(f)
        
            
        # TODO add checks
        self.name = model_info['name'] # name of the model (for GUI)
        
        self.ckpt_path = model_info['ckpt_path'] # path to saved model (meta + checkpoints). Loads latest if points to a folder, otherwise loads specific checkpoint
        
        # if path points to a folder, use latest checkpoint
        if (os.path.exists(self.ckpt_path) and os.path.isdir(self.ckpt_path)):
            self.ckpt_path = tf.train.latest_checkpoint(self.ckpt_path)        
        
        # get tensor info on inputs and outputs
        self.input_shape, self.input_range, self.input_op = get_info_from_dict(model_info, 'input')
        self.output_shape, self.output_range, self.output_op = get_info_from_dict(model_info, 'output')

        
        # init tensorflow stuff        
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        
        # load model
#        with self.sess.as_default():
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(self.ckpt_path + '.meta')
            saver.restore(self.sess, self.ckpt_path)



    def predict(self, img, input_range=(0., 1.), output_range=(0., 1.)):
        if input_range != self.input_range:
            img = np.interp(img, input_range, self.input_range)
        
         # add batch of 1 if input data is lacking batch dimension
        img = np.expand_dims(img, 0) if len(img.shape) < 4 else img
        model_out = self.sess.run( [self.output_op], { self.input_op: img })[0]
        
        if output_range != self.output_range:
            model_out = np.interp(model_out, self.output_range, output_range)
            
        return model_out
