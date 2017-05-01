#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: memo

Main app
"""

from __future__ import print_function
from __future__ import division

import numpy as np
import time

import params
import gui

import msa.utils
from msa.capturer import Capturer
from msa.predictor import Predictor
from msa.framestats import FrameStats


#%%
capture = None # msa.capturer.Capturer, video capture wrapper
predictor = None # msa.predictor.Predictor, model for prediction

img_cap = np.empty([]) # captured image before processing
img_in = np.empty([]) # processed capture image
img_out = np.empty([]) # output from prediction model


#%% init gui and params

gui.init_app()

pyqt_params = gui.init_params(params.params_list, target_obj=params, w=320)

# reading & writing to pyqtgraph.parametertree seems to be slow,
# so going to cache in an object for direct access
gui.params_to_obj(pyqt_params, target_obj=params, create_missing=True, verbose=True)

# create main window
gui.init_window(x=320, w=(gui.screen_size().width()-320), h=(gui.screen_size().width()-320)*0.4)


#%%

# load predictor model
predictor = Predictor(json_path = './models/gart_canny_256.json')


# init capture device
def init_capture(capture, output_shape):
    if capture:
        capture.close()
        
    capture_shape = (params.Capture.Init.height, params.Capture.Init.width)
    capture = Capturer(sleep_s = params.Capture.sleep_s,
                       device_id = params.Capture.Init.device_id,
                       capture_shape = capture_shape,
                       capture_fps = params.Capture.Init.fps,
                       output_shape = output_shape
                       )
    
    capture.update()
    
    if params.Capture.Init.use_thread:
        capture.start()
    
    return capture


capture = init_capture(capture, output_shape=predictor.input_shape)


# keep track of frame count and frame rate
frame_stats = FrameStats('Main')


# main update loop
while not params.Main.quit:
    
    # reinit capture device if parameters have changed
    if params.Capture.Init.reinitialise:
        params.child('Capture').child('Init').child('reinitialise').setValue(False)
        capture = init_capture(capture, output_shape=predictor.input_shape)
        
        
    capture.enabled = params.Capture.enabled
    if params.Capture.enabled:
        # update capture parameters from GUI
        capture.output_shape = predictor.input_shape
        capture.verbose = params.Main.verbose
        capture.freeze = params.Capture.freeze
        capture.sleep_s = params.Capture.sleep_s
        for p in msa.utils.get_members(params.Capture.Processing):
            setattr(capture, p, getattr(params.Capture.Processing, p))
        
        # run capture if multithreading is disabled
        if params.Capture.Init.use_thread == False:
            capture.update()
            
        img_cap = np.copy(capture.img) # create copy to avoid thread issues


    # interpolate (temporal blur) on input image
    img_in = msa.utils.np_lerp( img_in, img_cap, 1 - params.Prediction.pre_time_lerp)

    # run prediction
    if params.Prediction.enabled and predictor:
        img_predicted = predictor.predict(img_in)[0]
    else:
        img_predicted = capture.img0

    # interpolate (temporal blur) on output image
    img_out = msa.utils.np_lerp(img_out, img_predicted, 1 - params.Prediction.post_time_lerp)

    # update frame states
    frame_stats.verbose = params.Main.verbose
    frame_stats.update()
    
    # update gui
    gui.update_image(0, capture.img0)
    gui.update_image(1, img_in)
    gui.update_image(2, img_out)
    gui.update_stats(frame_stats.str + "   |   " + capture.frame_stats.str)
    gui.process_events()

    time.sleep(params.Main.sleep_s)


# cleanup
capture.close()
gui.close()

capture = None
predictor = None
    
print('Finished')