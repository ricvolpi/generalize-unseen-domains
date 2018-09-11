import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import lrelu

class Model(object):
    """Tensorflow model
    """
    def __init__(self, mode='train'):

	self.no_classes = 10
	self.img_size = 32
	self.no_channels = 3

    def encoder(self, images, reuse=False, return_feat=False):

	images = tf.reshape(images, [-1, self.img_size, self.img_size, self.no_channels])

	with tf.variable_scope('encoder', reuse=reuse):
	    with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
		with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='VALID'):

		    net = slim.conv2d(images, 64, 5, scope='conv1')
		    net = slim.max_pool2d(net, 2, stride=2, scope='pool1')
		    net = slim.conv2d(net, 128, 5, scope='conv2')
		    net = slim.max_pool2d(net, 2, stride=2, scope='pool2')
		    net = tf.contrib.layers.flatten(net)
		    net = slim.fully_connected(net, 1024, scope='fc1')
		    net = slim.fully_connected(net, 1024, scope='fc2')
		    if return_feat:
			return net	
		    net = slim.fully_connected(net, self.no_classes, activation_fn=None, scope='fco')
		    return net
		
    def build_model(self):

	#images placeholder
	self.z = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, self.no_channels], 'z')
	#labels placeholder
	self.labels = tf.placeholder(tf.int64, [None], 'labels')
	
	#images-for-gradient-ascent variable
	self.z_hat = tf.get_variable('z_hat', [self.batch_size, self.img_size, self.img_size, self.no_channels])
	#op to assign the value fed to self.z to the variable
	self.z_hat_assign_op = self.z_hat.assign(self.z)

	self.logits = self.encoder(self.z)
	self.logits_hat = self.encoder(self.z_hat, reuse=True)

	#this is just for the softmax method
	self.softmax_logits = tf.nn.softmax(self.logits)
	self.extracted_features = self.encoder(self.z, return_feat=True, reuse=True)

	#for evaluation
	self.pred = tf.argmax(self.logits, 1)
	self.correct_pred = tf.equal(self.pred, self.labels)
	self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

	#variables for the minimizer are the net weights, variables for the maxmizer are the images' pixels
	t_vars = tf.trainable_variables()
	min_vars = [var for var in t_vars if 'z_hat' not in var.name]
	max_vars = [var for var in t_vars if 'z_hat' in var.name]

	#loss for the minimizer
	self.min_loss = slim.losses.sparse_softmax_cross_entropy(self.logits, self.labels)

	#first term of the loss for the maximizer (== loss for the minimizer)
	self.max_loss_1 = slim.losses.sparse_softmax_cross_entropy(self.logits_hat, self.labels)	    

	#second term of the loss for the maximizer
	self.max_loss_2 = slim.losses.mean_squared_error(self.encoder(self.z, reuse=True, return_feat=True), self.encoder(self.z_hat, reuse=True, return_feat=True))

	#final loss for the maximizer
	self.max_loss = self.max_loss_1 - self.gamma * self.max_loss_2

	#we use Adam for the minimizer and vanilla SGD for the maximizer 
	self.min_optimizer = tf.train.AdamOptimizer(self.learning_rate_min) 
	self.max_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate_max) 

	#minimizer
	self.min_train_op = slim.learning.create_train_op(self.min_loss, self.min_optimizer, variables_to_train = min_vars)
	#maximizer (-)
	self.max_train_op = slim.learning.create_train_op(-self.max_loss, self.max_optimizer, variables_to_train = max_vars)

	min_loss_summary = tf.summary.scalar('min_loss', self.min_loss)
	max_loss_summary = tf.summary.scalar('max_loss', self.max_loss)

	accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
	self.summary_op = tf.summary.merge([min_loss_summary, max_loss_summary, accuracy_summary])		


