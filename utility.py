
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from time import time
from matplotlib import pyplot as plt


# In[3]:


def plot_training_history(r):
    #绘制折线图 loss
    plt.plot(r.history['loss'],label='loss')
    plt.plot(r.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()
    
    #绘制折线图 accuracies
    plt.plot(r.history['acc'], label='acc')
    plt.plot(r.history['val_acc'], label='val_acc')
    plt.legend()
    plt.show()
    

def plot_images(images,cls_true,cls_pred=None):
    name = ['transverse','longitudinal', 'crocodile','normal']
    assert len(images) == len(cls_true) == 9
    
    # 创建3x3的子图
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # 获取第i个图像并重塑阵列。

        image = images[i].reshape(80, 80)

        # 保证噪声的像素值介于0-1之间
        # image = np.clip(image, 0.0, 1.0)

        # 绘图
        ax.imshow(image,
                  cmap='gray',
                  interpolation='nearest')

        # 显示真实的和预测的类
        if cls_pred is None:
            xlabel = "True:{0}".format(name[cls_true[i]])
        else:
            xlabel = "True:{0}, Pred:{1}".format(name[cls_true[i]], name[cls_pred[i]])

        # 将类设置为x轴上的标签
        ax.set_xlabel(xlabel)

        # 删除刻度
        ax.set_xticks([])
        ax.set_yticks([])

    # 在一个Notebook的cell里
    plt.show()
    

class clean_data(object):
    _train = True
    
    def __init__(self, filename, train=True):
        self._train = train
        self._train_df = pd.read_csv(filename)
        self._train_df['feature'] = self._train_df['feature'].map(lambda x: np.array(list(map(float, x.split()))))
        self._image_size = self._train_df.feature[0].size
        self._image_shape = (int(np.sqrt(self._image_size)), int(np.sqrt(self._image_size)))
        self._dataNum = self._train_df.size
        self._feature = np.array(self._train_df.feature.map(lambda x: x.reshape(self._image_shape)).values.tolist())
        if self._train:
            self._label = self._train_df.label.values
            self._labelNum = self._train_df['label'].unique().size
            self._onehot = pd.get_dummies(self._train_df.label).values
            
    @property
    def distribution(self):
        return self._distribution

    @property
    def image_size(self):
        return self._image_size

    @property
    def image_shape(self):
        return self._image_shape

    @property
    def dataNum(self):
        return self._dataNum

    @property
    def feature(self):
        return self._feature

    if _train:
        @property
        def label(self):
            return self._label

        @property
        def labelNum(self):
            return self._labelNum

        @property
        def onehot(self):
            return self._onehot

