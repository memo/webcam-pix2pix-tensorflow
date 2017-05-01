#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: memo
"""

from __future__ import print_function
from __future__ import division

import numpy as np
import os


def np_lerp(a, b, t):
    if a.shape == b.shape and t > 0:
        a = a * (1. - t) + b * t
    else:
        a = b
    return a


def np_weighted_sum(a, b, wa, wb):
    if a.shape == b.shape and wa > 0:
        a = a * wa + b * wb
    elif wb > 0:
        a = b
    return a


def get_members(obj):
    '''returns public variable names of a class or object (i.e. not private or functions)'''
    return [n for n in dir(obj) if not callable(getattr(obj, n)) and not n.startswith("__")]


def get_members_and_info(obj):
    '''returns public variable (name, type, value) of a class or object (i.e. not private or functions)'''
    return [(n, type(getattr(obj, n)), getattr(obj, n)) for n in dir(obj) if not callable(getattr(obj, n)) and not n.startswith("__")]


def get_file_list(path, extensions=['jpg', 'jpeg', 'png']):
    '''returns a (flat) list of paths of all files of (certain types) recursively under a path'''
    paths = [os.path.join(root, name)
             for root, dirs, files in os.walk(path)
             for name in files
             if name.lower().endswith(tuple(extensions))]
    return paths
    
