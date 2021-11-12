# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import numpy as np

from keras.models import Model
from keras import backend as K
import keras
import cv2
import utils



def conv_output(model, layer_name, img):
    """Get the output of conv layer.

    Args:
           model: keras model.
           layer_name: name of layer in the model.
           img: processed input image.

    Returns:
           intermediate_output: feature map.
    """
    # this is the placeholder for the input images
    input_img = model.input

    try:
        # this is the placeholder for the conv output
        out_conv = model.get_layer(layer_name).output
    except:
        raise Exception('Not layer named {}!'.format(layer_name))

    # get the intermediate layer model
    intermediate_layer_model = Model(inputs=input_img, outputs=out_conv)

    # get the output of intermediate layer model
    intermediate_output = intermediate_layer_model.predict(img)

    return intermediate_output[0]


def conv_filter(model, layer_name, img):
    """Get the filter of conv layer.

    Args:
           model: keras model.
           layer_name: name of layer in the model.
           img: processed input image.

    Returns:
           filters.
    """
    # this is the placeholder for the input images
    input_img = model.input

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

    try:
        layer_output = layer_dict[layer_name].output
    except:
        raise Exception('Not layer named {}!'.format(layer_name))

    kept_filters = []
    for i in range(layer_output.shape[-1]):
        loss = K.mean(layer_output[:, :, :, i])
        # compute the gradient of the input picture with this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = utils.normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # step size for gradient ascent
        step = 1.
        # run gradient ascent for 20 steps
        fimg = img.copy()*1.0

        for j in range(40):
            loss_value, grads_value = iterate([fimg])
            fimg += grads_value * step

        # decode the resulting input image
        fimg = utils.deprocess_image(fimg[0])
        kept_filters.append((fimg, loss_value))

        # sort filter result
        kept_filters.sort(key=lambda x: x[1], reverse=True)

    return np.array([f[0] for f in kept_filters])


def output_heatmap(model, last_conv_layer, img):
    """Get the heatmap for image.

    Args:
           model: keras model.
           last_conv_layer: name of last conv layer in the model.
           img: processed input image.

    Returns:
           heatmap: heatmap.
    """
    # predict the image class
    preds = model.predict(img)
    # find the class index
    index = np.argmax(preds[0])
    # This is the entry in the prediction vector
    target_output = model.output[:, index]

    # get the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer)

    # compute the gradient of the output feature map with this target class
    grads = K.gradients(target_output, last_conv_layer.output)[0]

    # mean the gradient over a specific feature map channel
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # this function returns the output of last_conv_layer and grads
    # given the input picture
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([img])

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the target class

    for i in range(conv_layer_output_value.shape[-1]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(conv_layer_output_value, axis=-1)

    if np.sum(np.maximum(heatmap, 0)) == 0:
        heatmap = np.minimum(heatmap, 0)
        heatmap /= np.min(heatmap)
    else:
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
    return heatmap

def merge_heatmap_img(img,heatmap):
    """

    :param img: 80x80x1 或者80x80x3
    :param heatmap:
    :return:
    """


    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + img

    cv2.imwrite('cam.jpg', superimposed_img)

if __name__ == '__main__':
    img_path = r'CNN-images/trainImages/4_50.png'
    layer_name = 'layer_conv3'
    last_conv_layer = 'layer_conv3'

    model = keras.models.load_model('epoch-380_loss-0.0345_val_acc-0.9858.h5')

    img, pimg = utils.read_img(img_path, (1,80, 80,-1))

    #cout = conv_output(model, 'layer_conv1', pimg)
    #utils.vis_conv(cout, 2, 'layer_conv1', 'conv')
    #cout = conv_output(model, 'layer_conv2', pimg)
    #utils.vis_conv(cout, 2, 'layer_conv2', 'conv')
    #cout = conv_output(model, 'layer_conv3', pimg)
    #utils.vis_conv(cout, 2, 'layer_conv3', 'conv')

    #
    pimg = np.random.random((1, 80, 80, 1)) * 20 + 128.
    fout = conv_filter(model, 'layer_conv1', pimg)
    utils.vis_conv(fout, 2, 'layer_conv1', 'filter')
    fout = conv_filter(model, 'layer_conv2', pimg)
    utils.vis_conv(fout, 2, 'layer_conv2', 'filter')
    fout = conv_filter(model, 'layer_conv3', pimg)
    utils.vis_conv(fout, 2, 'layer_conv3', 'filter')
    #
    #heatmap = output_heatmap(model, last_conv_layer, pimg)
   # merge_heatmap_img(img,heatmap)
    # #
    #
   # utils.vis_heatmap(img, heatmap)
