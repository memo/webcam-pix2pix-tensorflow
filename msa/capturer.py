#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: memo

Wrapper for opencv VideoCapture
Captures from webcam and does all preprocessing
runs in separate thread
"""

from __future__ import print_function
from __future__ import division

import cv2
import threading
import time
import numpy as np

from msa.framestats import FrameStats


class Capturer(threading.Thread):
    def __init__(self, sleep_s, device_id=0, capture_shape=None, capture_fps=0, output_shape=(0, 0, 3)):
        print('Capturer.init with', device_id, capture_shape, capture_fps, output_shape)
        threading.Thread.__init__(self)
        
        self.sleep_s = sleep_s
        self.cvcap = cv2.VideoCapture(device_id)

        if capture_shape != None:
            self.cvcap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_shape[1])
            self.cvcap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_shape[0])
            
        if capture_fps != 0:
            self.cvcap.set(cv2.CAP_PROP_FPS, capture_fps)          
        
        if self.cvcap:
            self.fps = self.cvcap.get(cv2.CAP_PROP_FPS)
            self.width = int(self.cvcap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cvcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.aspect_ratio = self.width * 1.0 / self.height
            print('   Initialized at {}x{} at {}fps'.format(self.width, self.height, self.fps))
        else:
            self.enabled = False
            raise Exception('Could not initialise capture device')
            return


        self.frame_stats = FrameStats('Capturer')
        
        self.output_shape = output_shape
        
        self.verbose = False
        self.enabled = True
        self.thread_running = False
        
        self.freeze = False
        self.frame_diff = False
        self.flip_h = False
        self.flip_v = False
        
        self.grayscale = False
        self.pre_blur = 0
        self.pre_median = 0
        self.pre_thresh = 0
        self.adaptive_thresh = False
        self.adaptive_thresh_block = 7
        self.adaptive_thresh_c = 2
        self.invert = False
        self.canny = False
        self.canny_t1 = 100
        self.canny_t2 = 200
        self.accum_w1 = 0
        self.accum_w2 = 0
        self.post_blur = 0
        self.post_thresh = 0
    
    
        
    def close(self):
        print('Capturer.close')
        self.stop_thread()
        self.cvcap.release()
    
        
    
    # TODO: minimise and optimise RGB<->GRAY and FLOAT<->UINT8 conversions
    def update(self):
        if not self.enabled:
            return
        
        ret, raw = self.cvcap.read()
           
        if ret:
            if self.freeze:
                raw = self.raw
            else:
                # flip
                if self.flip_h or self.flip_v:
                    if self.flip_h and self.flip_v: flip_mode = -1
                    elif self.flip_h: flip_mode = 1
                    else: flip_mode = 0
                    raw = cv2.flip(raw, flip_mode)
                
                # normalise to float 0...1
                raw = np.float32(raw) / 255.
             

            if self.frame_diff:
                img = cv2.absdiff(raw, self.raw)
            else:
                img = raw

            self.raw = raw # save original captured image before any processing
            
            # convert from greyscale if need be
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if self.output_shape[-1] == 1 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # crop to square. TODO make this more flexible
            if self.output_shape[0] > 0 and self.output_shape[0]==self.output_shape[1]:
                wdiff = int((self.width-self.height)/2)
                img = img[0:self.height, wdiff:wdiff+self.height]
            
            # resize
            if self.output_shape[0] > 0 and self.output_shape[1] > 0:
                img = cv2.resize(img, tuple(self.output_shape[:2]), interpolation=cv2.INTER_AREA)

                            
            self.img0 = img # save captured image before any processing (but after cropping and resizing)
            
            
            # processing
            
            if self.grayscale:
                img = cv2.cvtColor( cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
            
            if self.pre_blur > 0:
                w = self.pre_blur * 2 + 1
                img = cv2.GaussianBlur(img, (w,w), 0)
                
            if self.pre_median > 0:
                img = cv2.cvtColor(np.uint8(img * 255.), cv2.COLOR_RGB2GRAY) # grayscale uint8
                img = cv2.medianBlur(img, self.pre_median*2+1)
                img = cv2.cvtColor(np.float32(img) / 255., cv2.COLOR_GRAY2RGB) # float RGB


            if self.pre_thresh > 0:
                _, img = cv2.threshold(img, self.pre_thresh / 255., 1., cv2.THRESH_TOZERO)

            if self.adaptive_thresh:
                img = cv2.cvtColor(np.uint8(img * 255.), cv2.COLOR_RGB2GRAY) # grayscale uint8
                img = cv2.adaptiveThreshold(img,
                                            maxValue = 255,
                                            adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            blockSize = self.adaptive_thresh_block*2+1,
                                            thresholdType = cv2.THRESH_BINARY_INV,
                                            C = self.adaptive_thresh_c)
                img = cv2.cvtColor(np.float32(img) / 255., cv2.COLOR_GRAY2RGB) # float RGB

            if self.invert:
                img = cv2.absdiff(np.ones_like(img), img)

            if self.canny:
                img = cv2.cvtColor(np.uint8(img * 255.), cv2.COLOR_RGB2GRAY) # grayscale uint8
                img = cv2.Canny(img, self.canny_t1, self.canny_t2)
                img = cv2.cvtColor(np.float32(img) / 255., cv2.COLOR_GRAY2RGB) # float RGB
                            
            if self.post_blur > 0:
                w = self.post_blur * 2 + 1
                img = cv2.GaussianBlur(img, (w,w), 0)
                            
            if self.accum_w1 > 0 and self.accum_w2 > 0:
                img = self.img * self.accum_w1 + img * self.accum_w2
                
            if self.post_thresh > 0:
                _, img = cv2.threshold(img, self.post_thresh / 255., 1., cv2.THRESH_BINARY)


            self.img = img
            
        
        self.frame_stats.verbose = self.verbose
        self.frame_stats.update()
        
        

    def run(self):
        print('Capturer.run')
        self.thread_running = True
        while self.thread_running:
            self.update()
            time.sleep(self.sleep_s)
            
            
            
    def stop_thread(self):
        print('Capturer.stop_thread')        
        self.thread_running = False
        
