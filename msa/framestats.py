#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: memo

Keeps track of frame count, frame time delta, fps etc.
"""

from __future__ import print_function
from __future__ import division

import time

class FrameStats:
    def __init__(self, name):
        self.reset()
        self.name = name
        self.verbose = True
        self.fps_smoothing = 0.9
        
    def update(self):            
        now_time = time.time() # time.perf_counter() # for python 3.5
        self.time_delta = now_time - self.last_time
        self.last_time = now_time
        fps = 1.0/self.time_delta if self.time_delta > 0 else 0
        self.fps += (fps - self.fps) * (1. - self.fps_smoothing)
        self.str = "[ {} : frame_number:{} time_delta:{:.4f}s {:.2f}fps ]".format(self.name, self.frame_number, self.time_delta, self.fps)
        if self.verbose:
            print(self.str)
        self.frame_number += 1
        
    def reset(self):
        self.fps = 0      
        self.frame_number = 0;
        self.time_delta =0
        self.last_time = time.time() # time.perf_counter() # for python 3.5
        
