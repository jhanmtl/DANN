import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib

from sklearn.datasets import make_moons


def make_trans_moons(theta=40, nb=100, noise=.05):
    from math import cos, sin, pi
    
    X, y = make_moons(nb, noise=noise, random_state=1) 
    Xt, yt = make_moons(nb, noise=noise, random_state=2)
    
    trans = -np.mean(X, axis=0) 
    X  = 2*(X+trans)
    Xt = 2*(Xt+trans)
    
    theta = -theta*pi/180
    rotation = np.array( [  [cos(theta), sin(theta)], [-sin(theta), cos(theta)] ] )
    Xt = np.dot(Xt, rotation.T)
    
    return X, y, Xt, yt


@tf.custom_gradient
def reverse_gradient(x):
  def grad(dy):
    return -1*dy
  return x,grad

class GRL(tf.keras.layers.Layer):
  def __init__(self):
    super(GRL,self).__init__()
  
  def call(self,x):
    return reverse_gradient(x)

class DANN():
  def __init__(self,alpha=5,classifier_lr=1e-3,extractor_lr=1e-3,discriminator_lr=1e-3,grl=True):
    super(DANN,self).__init__()
    
    self.extractor=tf.keras.models.Sequential([tf.keras.layers.Input(shape=(2)),
                                               tf.keras.layers.Dense(32,activation='relu'),
                                               tf.keras.layers.Dense(32,activation='relu')
                                              ])
    
    self.classifier=tf.keras.models.Sequential([
                                                tf.keras.layers.Dense(64,activation='relu'),
                                                tf.keras.layers.Dense(64,activation='relu'),
                                                tf.keras.layers.Dense(64,activation='relu'),
                                                tf.keras.layers.Dense(1,activation='sigmoid')])
    
    self.grl=grl

    if grl:
      self.discriminator=tf.keras.models.Sequential([GRL(),
                                                    tf.keras.layers.Dense(32,activation='relu'),
                                                    tf.keras.layers.Dense(32,activation='relu'),
                                                    tf.keras.layers.Dense(1,activation='sigmoid')])
    else:
      self.discriminator=tf.keras.models.Sequential([
                                                    tf.keras.layers.Dense(32,activation='relu'),
                                                    tf.keras.layers.Dense(32,activation='relu'),
                                                    tf.keras.layers.Dense(1,activation='sigmoid')])

    self.class_predictor=tf.keras.models.Sequential([self.extractor,self.classifier])
    self.domain_predictor=tf.keras.models.Sequential([self.extractor,self.discriminator])

    self.alpha=alpha

    self.classifier_lr=classifier_lr
    self.extractor_lr=extractor_lr
    self.discriminator_lr=discriminator_lr

    self.classifier_opt=tf.keras.optimizers.Adam(self.classifier_lr)
    self.extractor_opt=tf.keras.optimizers.Adam(self.extractor_lr)
    self.discriminator_opt=tf.keras.optimizers.Adam(self.discriminator_lr)

    self.bce_loss=tf.keras.losses.BinaryCrossentropy()

    self.class_acc=tf.keras.metrics.BinaryAccuracy()
    self.class_loss=tf.keras.metrics.Mean()

    self.domain_acc=tf.keras.metrics.BinaryAccuracy()
    self.domain_loss=tf.keras.metrics.Mean()
  
  @tf.function
  def train_cross_domain(self,source_x,source_y,target_x):

    self.class_acc.reset_states()
    self.class_loss.reset_states()
    self.domain_acc.reset_states()
    self.domain_loss.reset_states()

    source_domain_label=np.zeros(len(source_x))
    target_domain_label=np.ones(len(target_x))
    domain_labels=np.concatenate((source_domain_label,target_domain_label))
    
    domain_labels=tf.convert_to_tensor(domain_labels,dtype=tf.int64)
    combined_x=tf.concat((source_x,target_x),axis=0)

    shuffle_idx=tf.range(len(domain_labels))
    shuffle_idx=tf.random.shuffle(shuffle_idx)

    domain_labels=tf.gather(domain_labels,shuffle_idx)
    combined_x=tf.gather(combined_x,shuffle_idx,axis=0)

    with tf.GradientTape(persistent=True) as tape:

      class_pred=self.class_predictor(source_x,training=True)
      class_loss=self.bce_loss(source_y,class_pred)

      domain_pred=self.domain_predictor(combined_x,training=True)
      domain_loss=self.bce_loss(domain_labels,domain_pred)
      domain_loss*=self.alpha

    classifier_gradient=tape.gradient(class_loss,self.class_predictor.trainable_variables)

    extractor_gradient_by_class_loss=tape.gradient(class_loss,self.extractor.trainable_variables)
    extractor_gradient_by_domain_loss=tape.gradient(domain_loss,self.extractor.trainable_variables)

    discriminator_gradient=tape.gradient(domain_loss,self.discriminator.trainable_variables)

    self.classifier_opt.apply_gradients(zip(classifier_gradient,self.class_predictor.trainable_variables))

    self.extractor_opt.apply_gradients(zip(extractor_gradient_by_domain_loss,self.extractor.trainable_variables))
    self.extractor_opt.apply_gradients(zip(extractor_gradient_by_class_loss,self.extractor.trainable_variables))

    self.discriminator_opt.apply_gradients(zip(discriminator_gradient,self.discriminator.trainable_variables))

    self.class_loss(class_loss)
    self.class_acc(source_y,class_pred)

    self.domain_loss(domain_loss)
    self.domain_acc(domain_labels,domain_pred)
  
  @tf.function
  def train_source_domain(self,source_x,source_y,target_x):

    self.class_acc.reset_states()
    self.class_loss.reset_states()
    self.domain_acc.reset_states()
    self.domain_loss.reset_states()

    source_domain_label=np.zeros(len(source_x))
    target_domain_label=np.ones(len(target_x))
    domain_labels=np.concatenate((source_domain_label,target_domain_label))
    
    domain_labels=tf.convert_to_tensor(domain_labels,dtype=tf.int64)
    combined_x=tf.concat((source_x,target_x),axis=0)

    shuffle_idx=tf.range(len(domain_labels))
    shuffle_idx=tf.random.shuffle(shuffle_idx)

    domain_labels=tf.gather(domain_labels,shuffle_idx)
    combined_x=tf.gather(combined_x,shuffle_idx,axis=0)

    with tf.GradientTape(persistent=True) as tape:

      class_pred=self.class_predictor(source_x,training=True)
      class_loss=self.bce_loss(source_y,class_pred)

      domain_pred=self.domain_predictor(combined_x,training=True)
      domain_loss=self.bce_loss(domain_labels,domain_pred)
      domain_loss*=self.alpha

    classifier_gradient=tape.gradient(class_loss,self.class_predictor.trainable_variables)
    extractor_gradient_by_class_loss=tape.gradient(class_loss,self.extractor.trainable_variables)

    self.classifier_opt.apply_gradients(zip(classifier_gradient,self.class_predictor.trainable_variables))
    self.extractor_opt.apply_gradients(zip(extractor_gradient_by_class_loss,self.extractor.trainable_variables))

    self.class_loss(class_loss)
    self.class_acc(source_y,class_pred)

    self.domain_loss(domain_loss)
    self.domain_acc(domain_labels,domain_pred)
  
  def view_decision_boundary(self,x,y,props=None,additional_x=None):
    gridx_min=min(np.min(x[:,0]),np.min(x[:,0]))-1
    gridx_max=max(np.max(x[:,0]),np.max(x[:,0]))+1

    gridy_min=min(np.min(x[:,1]),np.min(x[:,1]))-1
    gridy_max=max(np.max(x[:,1]),np.max(x[:,1]))+1

    gridx_vec=np.arange(gridx_min,gridx_max,0.1)
    gridy_vec=np.arange(gridy_min,gridy_max,0.1)

    grid_pts=np.meshgrid(gridx_vec,gridy_vec)

    xx=grid_pts[0]
    yy=grid_pts[1]

    grid_ptx=xx.flatten()
    grid_ptx=np.reshape(grid_ptx,(len(grid_ptx),1))

    grid_pty=yy.flatten()
    grid_pty=np.reshape(grid_pty,(len(grid_pty),1))

    grid=np.concatenate((grid_ptx,grid_pty),axis=1)

    grid_pred=self.class_predictor.predict(grid)

    zz=grid_pred.reshape(xx.shape)

    plt.figure(figsize=(16,9))
    plt.axis('off')
    plt.contourf(xx,yy,zz,
                cmap=matplotlib.colors.ListedColormap([props['class0_srfcolor'],props['class1_srfcolor']]))

    mask0=y==0
    plt.scatter(x[mask0][:,0],x[mask0][:,1],marker=props['class0_marker'],c=props['class0_color'])

    mask1=y==1
    plt.scatter(x[mask1][:,0],x[mask1][:,1],marker=props['class1_marker'],c=props['class1_color'])

    if additional_x is not None:
      plt.scatter(additional_x[:,0],additional_x[:,1],c='k')
    
    plt.tight_layout()
  
  def evaluate_classifier_acc(self,x,ytrue,threshold=0.5):
    scores=self.class_predictor.predict(x)
    ypred=scores>threshold
    ypred=ypred.astype(int)
    ypred=ypred.flatten()
    acc=np.mean(ypred==ytrue)
    return acc
  
  def evaluate_discriminator_acc(self,x,label,threshold=0.5):
    scores=self.domain_predictor.predict(x)
    ypred=scores>threshold
    ypred=ypred.astype(int)
    ypred=ypred.flatten()
    acc=np.mean(ypred==label)
    return acc