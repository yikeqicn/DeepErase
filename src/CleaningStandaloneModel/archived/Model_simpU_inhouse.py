import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import cv2

import os, sys
import numpy as np
import math
from datetime import datetime
import time
from PIL import Image
from math import ceil
from tensorflow.python.ops import gen_nn_ops
# modules
from utils import log_images,_variable_with_weight_decay, _variable_on_cpu, _add_loss_summaries, _activation_summary, print_hist_summery, get_hist, per_class_acc, writeImage
from datasets import ArtPrint
from torch.utils.data import DataLoader, ConcatDataset, random_split#, SequentialSampler #yike: add SequentialSampler
from os.path import join, basename, dirname
#from Inputs import *

### initializers###
def msra_initializer(kl, dl):
    """
    kl for kernel size, dl for filter number
    """
    stddev = math.sqrt(2. / (kl**2 * dl))
    return tf.truncated_normal_initializer(stddev=stddev)

def orthogonal_initializer(scale = 1.1):
    ''' From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    def _initializer(shape, dtype=tf.float32, partition_info=None):
      flat_shape = (shape[0], np.prod(shape[1:]))
      a = np.random.normal(0.0, 1.0, flat_shape)
      u, _, v = np.linalg.svd(a, full_matrices=False)
      # pick the one with the correct shape
      q = u if u.shape == flat_shape else v
      q = q.reshape(shape) #this needs to be corrected to float32
      return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer

### graph units ###

def conv_layer_with_bn(inputT, shape, train_phase, activation=True, name=None):
    in_channel = shape[2]
    out_channel = shape[3]
    k_size = shape[0]
    with tf.variable_scope(name) as scope:
      kernel = _variable_with_weight_decay('ort_weights', shape=shape, initializer=orthogonal_initializer(), wd=None)
      conv = tf.nn.conv2d(inputT, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [out_channel], tf.constant_initializer(0.0))
      bias = tf.nn.bias_add(conv, biases)
      if activation is True:
        conv_out = tf.nn.relu(batch_norm_layer(bias, train_phase, scope.name))
      else:
        conv_out = batch_norm_layer(bias, train_phase, scope.name)
    return conv_out

def get_deconv_filter(f_shape):
  """
    reference: https://github.com/MarvinTeichmann/tensorflow-fcn
  """
  width = f_shape[0]
  heigh = f_shape[0]
  f = ceil(width/2.0)
  c = (2 * f - 1 - f % 2) / (2.0 * f)
  bilinear = np.zeros([f_shape[0], f_shape[1]])
  for x in range(width):
      for y in range(heigh):
          value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
          bilinear[x, y] = value
  weights = np.zeros(f_shape)
  for i in range(f_shape[2]):
      weights[:, :, i, i] = bilinear

  init = tf.constant_initializer(value=weights,
                                 dtype=tf.float32)
  return tf.get_variable(name="up_filter", initializer=init,
                         shape=weights.shape)

def deconv_layer(inputT, f_shape, output_shape, stride=2, name=None):
  # output_shape = [b, w, h, c]
  # sess_temp = tf.InteractiveSession()
  sess_temp = tf.global_variables_initializer()
  strides = [1, stride, stride, 1]
  with tf.variable_scope(name):
    weights = get_deconv_filter(f_shape)
    deconv = tf.nn.conv2d_transpose(inputT, weights, output_shape,
                                        strides=strides, padding='SAME')
  return deconv

def batch_norm_layer(inputT, is_training, scope):
  return tf.cond(is_training,
          lambda: tf.contrib.layers.batch_norm(inputT, is_training=True,
                           center=False, updates_collections=None, scope=scope+"_bn"),
          lambda: tf.contrib.layers.batch_norm(inputT, is_training=False,
                           updates_collections=None, center=False, scope=scope+"_bn", reuse = True))


### model ###


class Model:
    # Constants describing the training process. probably move to hyperprameters at main beginning
    MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
    NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
    LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
    
    def __init__(self, args, experiment, loss_weight, mustRestore=False):
      "init model: SegNet model"
      self.args = args
      self.experiment=experiment     
      self.mustRestore = mustRestore
      
      ###model hyperparameters###
      self.num_classes=args.num_class  
      # self.FilePaths = FilePaths
      self.batch_size = args.batch_size
      self.lrInit = args.lrInit
      self.loss_weight=loss_weight
      
      ###input### -- try to only set up graph once, combine train and test, by yike
      tf.reset_default_graph() # yike reset default graph
      self.input_images= tf.placeholder( tf.float32, shape=[None, self.args.image_h, self.args.image_w, self.args.image_c]) # try my best to make runtime batch_size flexible
      self.input_labels= tf.placeholder(tf.int64, shape=[None, self.args.image_h, self.args.image_w, 1])
      self.phase_train= tf.placeholder(tf.bool, name='phase_train')
      self.global_step=tf.Variable(0,trainable=False)
      self.learning_rate=tf.placeholder(tf.float32, shape=[])
    
      ###graph### -- combine 
      self.logit= self.setup_graph(self.input_images, self.phase_train)
      self.loss=self.cal_loss(self.logit,self.input_labels) 
      self.pred=tf.argmax(self.logit,axis=3)
      self.train_op=self.train(self.loss,self.learning_rate,self.global_step)

      
      ###session and saver###
      #self.saver=tf.train.Saver(max_to_keep=1)
      #self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
      (self.sess,self.saver) =self.initTF()
    
    ### 1. loss factory ###
    def weighted_loss(self, logits, labels): # num_classes, head=None):
      """ median-frequency re-weighting """
      with tf.name_scope('loss'):
           #print('w_llll')
           logits = tf.reshape(logits, (-1, self.num_classes))
           #print(logits.get_shape())
           epsilon = tf.constant(value=1e-10)

           logits = logits + epsilon

           # consturct one-hot label array
           label_flat = tf.reshape(labels, (-1, 1))
           #print(label_flat.get_shape())

           # should be [batch ,num_classes]
           labels = tf.reshape(tf.one_hot(label_flat, depth=self.num_classes), (-1, self.num_classes))
           # print(labels.get_shape())

           softmax = tf.nn.softmax(logits)
           #print(softmax.get_shape())
#        print(epsilon.get_shape())

#        print((labels * tf.log(softmax + epsilon)).get_shape())
#        print(head.shape)
#        print(tf.multiply(labels * tf.log(softmax + epsilon), head))
        
           cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon), self.loss_weight), axis=[1])
#        print(cross_entropy.get_shape()) # yike head -> self.loss_weight

           cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
#        print(cross_entropy_mean.get_shape())
           tf.add_to_collection('losses', cross_entropy_mean)
           
           loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
           print('loss: '+str(loss.get_shape()))      

      return loss
    
    def cal_loss(self,logits,labels):
       labels = tf.cast(labels, tf.int32)
       return self.weighted_loss(logits, labels)
    #self.weighted_loss(logits, labels, num_classes=NUM_CLASSES, head=loss_weight)
    
    ###2. train optimizer factory ###    
    def train(self,total_loss, lr, global_step):
       # all of them are tensor
       #total_sample = 274 yike: ok to comment out?
       #num_batches_per_epoch = 274/1 yike: ok to comment out?

       loss_averages_op = _add_loss_summaries(total_loss)
       # Compute gradients.
       with tf.control_dependencies([loss_averages_op]):
         #print('try...')
         opt = tf.train.AdamOptimizer(lr)
         print('toto_loss_shape: '+str(total_loss))
         opt.compute_gradients(total_loss)
         grads = opt.compute_gradients(total_loss)
         #print(grads)
         apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

         # Add histograms for trainable variables.
         for var in tf.trainable_variables():
           tf.summary.histogram(var.op.name, var)

         # Add histograms for gradients.
         for grad, var in grads:
           if grad is not None:
             tf.summary.histogram(var.op.name + '/gradients', grad)

         # Track the moving averages of all trainable variables.
         variable_averages = tf.train.ExponentialMovingAverage(Model.MOVING_AVERAGE_DECAY, global_step)
         variables_averages_op = variable_averages.apply(tf.trainable_variables())

         with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
           train_op = tf.no_op(name='train')

       return train_op
    
    
    ###3. graph factory ###
    
    def setup_graph(self, images, phase_train): # previous inference() labels,inference, batch_size -- in order to get batch_size at running time 
       #rather than using fixed batch_size in graph set up, revise it in inference:
       batchsize=tf.shape(images)[0] # yike !!!
       print('GGG')
       print(images.get_shape())
       # norm1
       norm1 = tf.nn.lrn(images, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,name='norm1')
       print(norm1.get_shape())
       # conv1
       conv1 = conv_layer_with_bn(norm1, [7, 7, images.get_shape().as_list()[3], 64], phase_train, name="conv1") # yike: 7 too large? how about 3?
       print(conv1.get_shape())
       # pool1
       pool1, pool1_indices = tf.nn.max_pool_with_argmax(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool1')
       print('111111')
       print(pool1.get_shape())
       print(pool1_indices.get_shape())
       # conv2
       conv2 = conv_layer_with_bn(pool1, [7, 7, 64, 64], phase_train, name="conv2")
    

       # pool2
       pool2, pool2_indices = tf.nn.max_pool_with_argmax(conv2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME', name='pool2')
       print('22222')
       print(pool2.get_shape())
       print(pool2_indices.get_shape())


       # conv3
       conv3 = conv_layer_with_bn(pool2, [7, 7, 64, 64], phase_train, name="conv3")

       # pool3
       pool3, pool3_indices = tf.nn.max_pool_with_argmax(conv3, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool3')

       print('33333')
       print(pool3.get_shape())
       print(pool3_indices.get_shape())

       # conv4
       conv4 = conv_layer_with_bn(pool3, [7, 7, 64, 64], phase_train, name="conv4")

       # pool4
       pool4, pool4_indices = tf.nn.max_pool_with_argmax(conv4, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool4')
       print('44444')
       print(pool4.get_shape())
       print(pool4_indices.get_shape())

       """ End of encoder """
       """ start upsample """
       # upsample4
       # Need to change when using different dataset out_w, out_h
       # upsample4 = upsample_with_pool_indices(pool4, pool4_indices, pool4.get_shape(), out_w=45, out_h=60, scale=2, name='upsample4')
       pool3_shape=pool3.get_shape()
       upsample4 = deconv_layer(pool4, [2, 2, 64, 64], tf.stack([batchsize, pool3_shape[1],pool3_shape[2], 64]), 2, "up4") #45, 60,
       #concat 4 yike
       #combined4=tf.concat(axis=3,values=(upsample4,pool3))  
       combined4=tf.concat(axis=3,values=(upsample4,conv4))  

        #print(tf.stack([batchsize, 45, 60, 64]))
       # decode 4
       conv_decode4 = conv_layer_with_bn(combined4, [7, 7, 128, 64], phase_train, False, name="conv_decode4")
       print('d4444444')
       print(conv_decode4.get_shape())
       # upsample 3
       # upsample3 = upsample_with_pool_indices(conv_decode4, pool3_indices, conv_decode4.get_shape(), scale=2, name='upsample3')
       pool2_shape=pool2.get_shape()
       upsample3= deconv_layer(conv_decode4, [2, 2, 64, 64], tf.stack([batchsize, pool2_shape[1],pool2_shape[2], 64]), 2, "up3") #90, 120
       #concat 3 yike
#       combined3=tf.concat(axis=3,values=(upsample3,pool2))  
       combined3=tf.concat(axis=3,values=(upsample3,conv3))  

        # decode 3
       conv_decode3 = conv_layer_with_bn(combined3, [7, 7, 128, 64], phase_train, False, name="conv_decode3")
       print('d333333')
       print(conv_decode3.get_shape())
       # upsample2
       # upsample2 = upsample_with_pool_indices(conv_decode3, pool2_indices, conv_decode3.get_shape(), scale=2, name='upsample2')
       pool1_shape=pool1.get_shape()
       upsample2= deconv_layer(conv_decode3, [2, 2, 64, 64], tf.stack([batchsize, pool1_shape[1],pool1_shape[2], 64]), 2, "up2") #180, 240
       #concat 2 yike
       #combined2=tf.concat(axis=3,values=(upsample2,pool1))   
       combined2=tf.concat(axis=3,values=(upsample2,conv2))   
       # decode 2
       conv_decode2 = conv_layer_with_bn(combined2, [7, 7, 128, 64], phase_train, False, name="conv_decode2")
       print('d22222')
       print(conv_decode2.get_shape()) 
       # upsample1
       # upsample1 = upsample_with_pool_indices(conv_decode2, pool1_indices, conv_decode2.get_shape(), scale=2, name='upsample1')
       upsample1=deconv_layer(conv_decode2, [2, 2, 64, 64], tf.stack([batchsize,self.args.image_h,self.args.image_w , 64]), 2, "up1") # IMAGE_HEIGHT, IMAGE_WIDTH yike !!!! deconv_layer(conv_decode2, [2, 2, 64, 64], [batch_size, 360, 480, 64], 2, "up1")
    
       #concat 1 yike
       #combined2=tf.concat(axis=3,values=(upsample2,pool1))   
       combined1=tf.concat(axis=3,values=(upsample1,conv1))   

       # decode4
       conv_decode1 = conv_layer_with_bn(combined1, [7, 7, 128, 64], phase_train, False, name="conv_decode1")
       print('d111111')
       print(conv_decode1.get_shape())
    
       """ end of Decode """
       """ Start Classify """
       # output predicted class number (6)
       with tf.variable_scope('conv_classifier') as scope:
         kernel = _variable_with_weight_decay('weights',
                                           shape=[1, 1, 64, self.num_classes],
                                           initializer=msra_initializer(1, 64),
                                           wd=0.0005)
         conv = tf.nn.conv2d(conv_decode1, kernel, [1, 1, 1, 1], padding='SAME')
         print('cv')
         print(conv.get_shape())
         biases = _variable_on_cpu('biases', [self.num_classes], tf.constant_initializer(0.0))
         print(biases.get_shape())
         logit= tf.nn.bias_add(conv, biases, name=scope.name)
         #conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)
         #print(conv_classifier.get_shape())
         #logit = conv_classifier
         #print('LLL')
         #print(labels)
         #print(conv_classifier)
    
         #loss = cal_loss(conv_classifier, labels)
         print(logit.get_shape())

       return logit # loss

    ###4. initialization###
    def initTF(self):
       "initialize TF"
       print('Python: ' + sys.version)
       print('Tensorflow: ' + tf.__version__)

       sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
       saver = tf.train.Saver(max_to_keep=1)  # saver saves model to file

       # Restore from saved model in current checkpoint directory
       latestSnapshot = tf.train.latest_checkpoint(self.args.ckptpath)  # is there a saved model?
       if self.mustRestore and not latestSnapshot: # if model must be restored (for inference), there must be a snapshot
         raise Exception('No saved model found in: ' + self.args.ckptpath)

       if latestSnapshot: # load saved model if available
         saver.restore(sess, latestSnapshot)
         print('Init with stored values from ' + latestSnapshot)
       else:
         sess.run(tf.global_variables_initializer())
         print('Ran global_variables_initializer')

         # initialize params from other model (transfer learning)
       if self.args.transfer:
         utils.maybe_download(source_url=self.args.urlTransferFrom,
                         filename=join(self.args.ckptpath, 'transferFrom'),
                         target_directory=None,
                         filetype='folder',
                         force=True)
         saverTransfer = tf.train.Saver(tf.trainable_variables()[:-1])  # load all variables except from logit (classification) layer
         saverTransfer.restore(sess, glob(join(self.args.ckptpath, 'transferFrom', 'model*'))[0].split('.')[0])
         print('Loaded variable values (except logit layer) from ' + self.args.urlTransferFrom)

       return (sess, saver) 
    
    ###5. training ###
    def trainBatch(self, images, labels):
       "feed a batch into the NN to train it"
    
       #sparse = self.toSparse(labels)
       #lrnrate = self.lrInit if self.batchesTrained < self.args.lrDrop1 else (
       #self.lrInit*1e-1 if self.batchesTrained < self.args.lrDrop2 else self.lrInit*1e-2)  # decay learning rate
       train_step=self.global_step.eval(session=self.sess)
       """ fix lr """ ## To Ronny, change the schedule?
       #lr = self.lrInit
       lr = self.lrInit if train_step < self.args.lrDrop1 else (
            self.lrInit*1e-1 if train_step < self.args.lrDrop2 else self.lrInit*1e-2) #yike
       (_, lossVal) = self.sess.run([self.train_op, self.loss],
                                  {self.input_images: images,
                                   self.input_labels: labels,
                                   self.learning_rate: lr, 
                                   self.phase_train: True})
       #self.batchesTrained += 1
       return lossVal
    
    def training(self, loader, validateloader=None,testloader=None):
       "train NN"
       epoch = 0  # number of training epochs since start
       best_accuracy=0.0
       step = 0
       while True:
         epoch += 1; print('Epoch:', epoch, ' Training...')
         # train
         counter = 0
         #step = 0 
         for idx, (images, labels) in enumerate(loader):
            images=images.numpy()
            labels=labels.numpy()
            loss_value=self.trainBatch(images,labels)
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            step+=1
            
            if idx % 100 ==0:
              print('TRAIN: Batch:', idx/len(loader), 'Loss:', loss_value)
              self.experiment.log_metric('train/loss', loss_value, step)
              logits=self.sess.run(self.logit, feed_dict={self.input_images: images, # check in, comment out in formal run
                                   #self.input_labels: labels,
                                   #self.learning_rate: lr, 
                                   self.phase_train: False})
              train_acc,train_acc_classes=per_class_acc(logits,labels)  # check in, comment out in formal run

         # train log:
         self.experiment.log_metric('train/acc',train_acc,step)
         self.experiment.log_metric('train/cap_0',train_acc_classes[0],step)
         self.experiment.log_metric('train/cap_1',train_acc_classes[1],step)
                
         #validate:
         if validateloader !=None: 
           avg_batch_loss,acc_total,cap_0,cap_1=self.validate(validateloader,epoch)
         else:
           avg_batch_loss,acc_total,cap_0,cap_1=self.validate(loader,epoch)
         self.experiment.log_metric('valid/acc',acc_total,step)
         self.experiment.log_metric('valid/cap_0',cap_0,step)
         self.experiment.log_metric('valid/cap_1',cap_1,step)
         self.experiment.log_metric('valid/loss',avg_batch_loss,step)   
         

         #test:
         if testloader !=None:
           acc_total,cap_0,cap_1=self.validate(testloader,epoch,is_testing=True)
           self.experiment.log_metric('test/acc',acc_total,step)
           self.experiment.log_metric('test/cap_0',cap_0,step)
           self.experiment.log_metric('test/cap_1',cap_1,step)
           
         # log best metrics
         if acc_total > best_accuracy: # if best validation accuracy so far, save model parameters
           print('Character error rate improved, save model')
           best_accuracy = acc_total
           noImprovementSince = 0
           self.save(epoch)
           open(join(self.args.ckptpath, 'accuracy.txt'), 'w').write('Validation accuracy, class 0, class 1 capture rates of saved model: %f%%, %f%% and %f%% ' % (acc_total * 100.0, cap_0 * 100.0, cap_1 * 100.0))
           self.experiment.log_metric('best/acc',acc_total,step)
           self.experiment.log_metric('best/cap_0',cap_0,step)
           self.experiment.log_metric('best/cap_1',cap_1,step)         
         else:
           print('Character error rate not improved')
           noImprovementSince += 1

         # stop training
         if epoch>=self.args.max_epoch: print('Done with training at epoch', epoch, 'sigoptObservation='+str(best_accuracy)); break            
            
            
    ###6. testing / validate ###
    def validate(self, loader, epoch, is_testing=False):
       "validate NN"
       if not is_testing: print('Validating NN')
       else: print('Testing NN')
       total_val_loss = 0.0
       #num_batches=len(loader)
       hist= np.zeros((self.num_classes, self.num_classes))
        
       image_upload_count=0 
       for idx, (images, labels) in enumerate(loader):
         images=images.numpy()
         labels=labels.numpy()
         val_loss,val_logit=self.sess.run([self.loss,self.logit],feed_dict=
                                {self.input_images: images, # check in, comment out in formal run
                                self.input_labels: labels,
                                self.phase_train: False}) #self.loss,val_loss,
          
         total_val_loss+=val_loss
         hist+=get_hist(val_logit,labels)
       #val_loss=total_val_loss / len(validateloader)*batch_size

         if epoch==self.args.max_epoch and image_upload_count<100: # decide how many images to upload
            pred=val_logit.argmax(3)
            images=np.squeeze(images,axis=3)
            image_upload_count=log_images(images,pred,image_upload_count,self.experiment,self.args.ckptpath)
            
       avg_batch_loss=total_val_loss/idx     
       cls_sample_nums=hist.sum(1).astype(float)
       capture_array=np.diag(hist)
       acc_total = capture_array.sum() / hist.sum()
       capture_rate_ls=[]
       for cls in range(self.num_classes):
         if cls_sample_nums[cls]==0:
            capture_rate=0.0
         else:
            capture_rate=capture_array[cls]/cls_sample_nums[cls]
         capture_rate_ls.append(capture_rate)
       #iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
       #mean_iu=np.nanmean(iu)
       print('VALID: Total accuracy: %f%%. Class 0 capture: %f%%. Class 1 capture: %f%%' % (acc_total * 100.0, capture_rate_ls[0] * 100.0, capture_rate_ls[1]*100.0)) 
       return avg_batch_loss,acc_total,capture_rate_ls[0],capture_rate_ls[1]
    
    ###7. infer ###
    def inferBatch(self, imgs): # modify to compatible to torch. previous def inferBatch(self, batch)
       "feed a batch into the NN to recngnize the texts"
  
       bt_size=len(imgs) # yike !!!!!!!!

       pred = self.sess.run(self.pred,feed_dict=
                                {self.input_images: images, # check in, comment out in formal run
                                self.input_labels: labels,
                                self.phase_train: False}) #yike self.batchsize!!!!!!!!!
       return pred
    ###8. save  ###
    def save(self, epoch):
       "save model to file"
       self.saver.save(self.sess, join(self.args.ckptpath, 'model'), global_step=epoch)
