import os
import keras.backend as K
from keras.models import load_model, model_from_json
import numpy as np
import json


def switch_backend(backend):
    if backend == 'theano':
        # os.environ['KERAS_BACKEND'] = 'theano'
        image_data_format = 'channels_first'
        K.set_image_data_format(image_data_format)
    elif backend == 'tf':
        # os.environ['KERAS_BACKEND'] = 'tensorflow'
        image_data_format = 'channels_last'
        K.set_image_data_format(image_data_format)
    else:
        raise ValueError('Unknown backend')


def get_th_model():
    switch_backend('theano')
    
    th_model = load_model('model_th.hdf5')
    
    th_weights = th_model.get_weights()
    th_json = th_model.to_json()
    
    return th_json, th_weights


def jsmodel_th_to_tf(js_str):
    js_obj = json.loads(js_str)
    
    js_obj['backend'] = 'tensorflow'

    p_flatten = False
    dense_to_tp = 0

    for n, l in enumerate(js_obj['config']):
        lc = l['config']
        # print(n, lc['name'])
        if 'data_format' in lc:
            lc['data_format'] = 'channels_last'

        if l['class_name'] == 'Flatten':
            p_flatten = True

        if p_flatten and l['class_name'] == 'Dense':
            p_flatten = False
            dense_to_tp = n

        if 'batch_input_shape' in lc:
            b_sh = lc['batch_input_shape']
            lc['batch_input_shape'] = [b_sh[0], b_sh[2], b_sh[3], b_sh[1]]
    
    return json.dumps(js_obj), dense_to_tp


th_json, th_weights = get_th_model()
tf_json, dense_to_tp = jsmodel_th_to_tf(th_json)

switch_backend('tf')

model_tf = model_from_json(tf_json)
tf_weights = th_weights.copy()

for n in range(len(tf_weights)):
    w_sh = tf_weights[n].shape
    # Change convolution to crosscorrelation and move channel to last place
    if len(w_sh) == 4:
        tf_weights[n] = (tf_weights[n].reshape((-1, w_sh[2], w_sh[3]))[::-1, :, :]).reshape((w_sh))

model_tf.set_weights(tf_weights)

# Transpose weights in the first dense layer after a flatten layer    
layer = model_tf.layers[dense_to_tp]
l_w = layer.get_weights()
n_sh = l_w[0].shape[-1]
c_sh = K.int_shape(model_tf.layers[dense_to_tp - 2].output)  # Shape of the last non-flatten
l_w[0] = l_w[0].reshape((c_sh[-1], c_sh[1], c_sh[2], n_sh)).transpose([1, 2, 0, 3]).reshape((-1, n_sh))

layer.set_weights(l_w)

model_tf.save('model_tf.hdf5')