#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: memo

Handles GUI and display
"""


from __future__ import print_function
from __future__ import division

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui


#update = None


_app = None
_update_timer = None
_ptree = None
_params = None #
_params_target_obj = None

_window_view = None
_window_layout = None
_window_imgs = []
_window_stats_label = None

_windows = []



def screen_size():
    return _app.desktop().screenGeometry()
    
    

def init_app():
    global _app
    print('gui | Initializing QApplication')
    _app = QtGui.QApplication([])

    
    
def process_events():
    if _app != None:
        _app.processEvents()



def toggle_param(p):
    p.setValue( not p.value() )



# reading & writing to pyqtgraph.parametertree seems to be slow,
# so going to cache in an object for direct access
def params_to_obj(pyqt_params, target_obj, create_missing=False, verbose=False):
    '''copy values from pyqtgraph params to an object'''
    class DummyParamObj:
        pass
    
    if verbose:
        print('params_to_obj {} to {}'.format(pyqt_params.name(), target_obj))
        
    for p in pyqt_params.children():
        if len(p.children()) > 0:
            obj_has_member = p.name() in dir(target_obj)
            if create_missing and not obj_has_member:
                if verbose:
                    print('   creating member', p.name())

                setattr(target_obj, p.name(), DummyParamObj()) # quite a hack to create a dummy object
                obj_has_member = True
            
                
            if obj_has_member:
                params_to_obj(p, getattr(target_obj, p.name()), create_missing, verbose)
            elif verbose:
                print('   skipping missing member', p.name())
                
        else:
            if verbose:
                print('   {}.{} = {}'.format(target_obj, p.name(), p.value()))
            setattr(target_obj, p.name(), p.value())
        


# TODO: only update what changed
def on_params_changed(param, changes):
    '''If anything changes in the parameter tree, update _params_target_obj object'''
    print("gui.on_params_changed", param, changes)
    params_to_obj(_params, _params_target_obj)
#    for param, change, data in changes:
#        path = p.childPath(param)
#        if path is not None:
#            childName = '.'.join(path)
#        else:
#            childName = param.name()
#        print('  parameter: %s'% childName)
#        print('  change:    %s'% change)
#        print('  data:      %s'% str(data))
#        print('  ----------')
    
        

def init_params(params_list, target_obj, x=0, y=0, w=320, h=1080, title='Tweak Me Control Freak'):
    global _ptree, _params, _params_target_obj, _windows
    print('gui.init_params')
    _params_target_obj = target_obj
    _params = pg.parametertree.Parameter.create(name='params', type='group', children=params_list)
    _ptree = pg.parametertree.ParameterTree()
    _ptree.setParameters(_params, showTop=False)
    _ptree.setWindowTitle(title)
    _ptree.setGeometry(x, y, w, h)
    _ptree.show()
    _params.sigTreeStateChanged.connect(on_params_changed) 
    
    _windows.append(_ptree)
    return _params



def _add_image_to_layout(layout, row=None, col=None, rowspan=1, colspan=1, title=''):
    i = pg.ImageItem(border='w')
    i.setOpts(axisOrder='row-major')
    
    l = layout.addLayout()
    l.addLabel(title)
    l.nextRow()

    v = l.addViewBox(lockAspect=True, invertY=True, row=row, col=col, rowspan=rowspan, colspan=colspan)
    v.addItem(i)
    return {'img':i, 'view':v, 'layout':l}
    


def init_window(x=0, y=0, w=1440, h=1080, title='Hello World'):
    global _window_view, _window_layout, _window_imgs, _window_stats_label, _windows
    view = pg.GraphicsView()
    layout = pg.GraphicsLayout(border=(100,100,100))
    view.setCentralItem(layout)
    view.setWindowTitle(title)
    view.setGeometry(x, y, w, h)
    view.show()
    
    imgs = []
    imgs.append( _add_image_to_layout(layout, title='capture') )
    imgs.append( _add_image_to_layout(layout, title='processed') )
    imgs.append( _add_image_to_layout(layout, title='prediction') )
    
    layout.nextRow()

    stats_label = pg.LabelItem()
    layout.addItem(stats_label, colspan=3)
    
    _window_view = view
    _window_layout = layout
    _window_imgs = imgs
    _window_stats_label = stats_label
    
    _windows.append(view)
    


def update_image(index, img_data, enabled=True, autoLevels=False, levels=(0., 1.)):
    if enabled:
        _window_imgs[index]['img'].setImage(img_data, autoLevels=autoLevels, levels=levels)
    else:
        _window_imgs[index]['img'].clear()
        
        
        
def update_stats(text):
    _window_stats_label.setText(text)

#    
#def start(sleep_s):
#    global _app, _update_timer
#    print('gui.start')
#    if update != None:
#        print('gui | Starting Update')
#        _update_timer = QtCore.QTimer()
#        _update_timer.timeout.connect(update)
#        _update_timer.start(sleep_s * 1000)
#    
#    print('gui | Starting QApplication')
#    _app.exec_()
#    
#    

def close():
    global _app, _update_timer, _windows
    _update_timer = None
    _app.closeAllWindows()
    for w in _windows:
        w.close()
