import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import numpy.random as npr
from ConfigParser import *
import os
import cPickle
import scipy.io
import sys
import glob
from numpy.linalg import norm
from scipy import misc

sys.path.append('../functions')

import utils

class TrainOps(object):

    def __init__(self, model, exp_dir):

	self.model = model
	self.exp_dir = exp_dir

	self.config = tf.ConfigProto()
	self.config.gpu_options.allow_growth=True

	self.data_dir = './data/'
		    	
    def load_exp_config(self):

	config = ConfigParser()
	config.read(self.exp_dir + '/exp_configuration')

	self.source_dataset = config.get('EXPERIMENT_SETTINGS', 'source_dataset')
	self.target_dataset = config.get('EXPERIMENT_SETTINGS', 'target_dataset')
	self.no_images = config.getint('EXPERIMENT_SETTINGS', 'no_images')

	self.log_dir = os.path.join(self.exp_dir,'logs')
	self.model_save_path = os.path.join(self.exp_dir,'model')

	if not os.path.exists(self.log_dir):
	    os.makedirs(self.log_dir)

	if not os.path.exists(self.model_save_path):
	    os.makedirs(self.model_save_path)

	self.add_train_iters = config.getint('MAIN_SETTINGS', 'add_train_iters')
	self.k = config.getint('MAIN_SETTINGS', 'k')	
	self.batch_size = config.getint('MAIN_SETTINGS', 'batch_size')
	self.model.batch_size = self.batch_size
	self.model.gamma = config.getfloat('MAIN_SETTINGS', 'gamma')
	self.model.learning_rate_min = config.getfloat('MAIN_SETTINGS', 'learning_rate_min')
	self.model.learning_rate_max = config.getfloat('MAIN_SETTINGS', 'learning_rate_max')
	self.T_adv = config.getint('MAIN_SETTINGS', 'T_adv')
	self.T_min = config.getint('MAIN_SETTINGS', 'T_min')

	self.train_iters = int(self.k * self.T_min + 1) 
	
    def load_svhn(self, split='train'):

	print ('Loading SVHN dataset.')

	image_file = 'train_32x32.mat' if split=='train' else 'test_32x32.mat'

	image_dir = os.path.join(self.data_dir, 'svhn', image_file)
	svhn = scipy.io.loadmat(image_dir)
	images = np.transpose(svhn['X'], [3, 0, 1, 2]) / 255.
	labels = svhn['y'].reshape(-1)
	labels[np.where(labels==10)] = 0
	return images, labels

    def load_mnist(self, split='train'):

	print ('Loading MNIST dataset.')
	image_file = 'train.pkl' if split=='train' else 'test.pkl'
	image_dir = os.path.join(self.data_dir, 'mnist', image_file)
	with open(image_dir, 'rb') as f:
	    mnist = cPickle.load(f)
	images = mnist['X'] 
	labels = mnist['y']

	images = images / 255.

	images = np.stack((images,images,images), axis=3) # grayscale to rgb

	return np.squeeze(images[:self.no_images]), labels[:self.no_images]

    def load_test_data(self, target):

	if target=='svhn':
	    self.target_test_images, self.target_test_labels = self.load_svhn(split='test')
	elif target=='mnist':
	    self.target_test_images, self.target_test_labels = self.load_mnist(split='test')

	return self.target_test_images,self.target_test_labels

    def train(self): 

	# build a graph
	print 'Building model'
	self.model.mode='train_encoder'
	self.model.build_model()
	print 'Built'

	print 'Loading data.'
	source_train_images, source_train_labels = self.load_mnist(split='train')
	source_test_images, source_test_labels = self.load_mnist(split='test')
	target_test_images, target_test_labels = self.load_test_data(target=self.target_dataset)
	print 'Loaded'
	
	with tf.Session(config=self.config) as sess:
	    tf.global_variables_initializer().run()
	    saver = tf.train.Saver()

	    summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())

	    print 'Training'
	    for t in range(self.train_iters + self.add_train_iters):

		#train_iters is defined by k in the load_config method	
		if ((t+1) % self.T_min == 0) and (t < self.train_iters): #if T_min iterations are passed and T_adv > 0
		    print 'Generating adversarial images.'
		    for start, end in zip(range(0, self.no_images, self.batch_size), range(self.batch_size, self.no_images, self.batch_size)): #going through the dataset
			feed_dict = {self.model.z: source_train_images[start:end], self.model.labels: source_train_labels[start:end]} 

			#assigning the current batch of images to the variable to learn z_hat
			sess.run(self.model.z_hat_assign_op, feed_dict) 
			for n in range(self.T_adv): #running T_adv gradient ascent steps
			    sess.run(self.model.max_train_op, feed_dict)
			    
			#tmp variable with the learned images
			learnt_imgs_tmp = sess.run(self.model.z_hat, feed_dict)

			#stacking the learned images and corresponding labels to the original dataset
			source_train_images = np.vstack((source_train_images, learnt_imgs_tmp))
			source_train_labels = np.hstack((source_train_labels, source_train_labels[start:end]))
		    
		    #shuffling the dataset
		    rnd_indices = range(len(source_train_images))
		    npr.shuffle(rnd_indices)
		    source_train_images = source_train_images[rnd_indices]
		    source_train_labels = source_train_labels[rnd_indices]
		    
		i = t % int(source_train_images.shape[0] / self.batch_size)

		#current batch of images and labels
		batch_z = source_train_images[i*self.batch_size:(i+1)*self.batch_size]
		batch_labels = source_train_labels[i*self.batch_size:(i+1)*self.batch_size]

		feed_dict = {self.model.z: batch_z, self.model.labels: batch_labels} 

		#running a step of gradient descent
		sess.run([self.model.min_train_op, self.model.min_loss], feed_dict) 

		#evaluating the model
		if t % 50 == 0:

		    summary, min_l, max_l, acc = sess.run([self.model.summary_op, self.model.min_loss, self.model.max_loss, self.model.accuracy], feed_dict)

		    train_rand_idxs = np.random.permutation(source_train_images.shape[0])[:100]
		    test_rand_idxs = np.random.permutation(target_test_images.shape[0])[:100]

		    train_acc, train_min_loss = sess.run(fetches=[self.model.accuracy, self.model.min_loss], 
					   feed_dict={self.model.z: source_train_images[train_rand_idxs], 
						      self.model.labels: source_train_labels[train_rand_idxs]})
		    test_acc, test_min_loss = sess.run(fetches=[self.model.accuracy, self.model.min_loss], 
					   feed_dict={self.model.z: target_test_images[test_rand_idxs], 
						      self.model.labels: target_test_labels[test_rand_idxs]})
												      
		    summary_writer.add_summary(summary, t)
		    print ('Step: [%d/%d] train_min_loss: [%.4f] train_acc: [%.4f] test_min_loss: [%.4f] test_acc: [%.4f]'%(t+1, self.train_iters, train_min_loss, train_acc, test_min_loss, test_acc))

	    print 'Saving'
	    saver.save(sess, os.path.join(self.model_save_path, 'encoder'))

    def test(self, target):

	test_images, test_labels = self.load_test_data(target=self.target_dataset)

	# build a graph
	print 'Building model'
	self.model.mode='train_encoder'
	self.model.build_model()
	print 'Built'

	with tf.Session() as sess:

	    tf.global_variables_initializer().run()

	    print ('Loading pre-trained model.')
	    variables_to_restore = slim.get_model_variables(scope='encoder')
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, os.path.join(self.model_save_path,'encoder'))


	    N = 100 #set accordingly to GPU memory
	    target_accuracy = 0
	    target_loss = 0

	    print 'Calculating accuracy'

	    for test_images_batch, test_labels_batch in zip(np.array_split(test_images, N), np.array_split(test_labels, N)):
		feed_dict = {self.model.z: test_images_batch, self.model.labels: test_labels_batch} 
		target_accuracy_tmp, target_loss_tmp = sess.run([self.model.accuracy, self.model.min_loss], feed_dict) 
		target_accuracy += target_accuracy_tmp/float(N)
		target_loss += target_loss_tmp/float(N)

	print ('Target accuracy: [%.4f] target loss: [%.4f]'%(target_accuracy, target_loss))
	
if __name__=='__main__':

    print '...'


