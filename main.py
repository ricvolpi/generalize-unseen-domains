import tensorflow as tf
from model import Model
from trainOps import TrainOps
import glob
import os
import cPickle

import numpy.random as npr
import numpy as np

flags = tf.app.flags
flags.DEFINE_string('gpu', '0', "GPU to used")
flags.DEFINE_string('exp_dir', 'exp_dir', "Experiment directory")
flags.DEFINE_string('mode', 'mode', "Experiment directory")
FLAGS = flags.FLAGS

def main(_):

    GPU_ID = FLAGS.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152 on stackoverflow
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

    EXP_DIR = FLAGS.exp_dir

    model = Model()
    trainOps = TrainOps(model, EXP_DIR)
    trainOps.load_exp_config()

    if FLAGS.mode=='train':
	print 'Training'
	trainOps.train()       

    elif FLAGS.mode=='test':
	print 'Testing'
	trainOps.test('svhn')       

if __name__ == '__main__':
    tf.app.run()



    






