import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib

from sklearn.datasets import make_moons


class FIDO():
  def __init__(self,alpha,classifier_lr=1e-3,extractor_lr=1e-3,reconstructor_lr=1e-3):
    super(FIDO,self).__init__()

    dunit=1024
    self.extractor=tf.keras.models.Sequential([tf.keras.layers.Input(shape=(2)),
                                               tf.keras.layers.Dense(512,activation='relu'),
                                               tf.keras.layers.Dense(dunit,activation='relu'),
                                               tf.keras.layers.Dense(dunit,activation='relu'),
                                               tf.keras.layers.Dense(dunit,activation='relu'),
                                               tf.keras.layers.Dense(dunit,activation='relu'),
                                               tf.keras.layers.Dense(dunit,activation='relu'),
                                               tf.keras.layers.Dense(dunit,activation='relu'),
                                              ])
    
    self.classifier=tf.keras.models.Sequential([
                                                # tf.keras.layers.Dense(1024,activation='relu'),
                                                # tf.keras.layers.Dense(256,activation='relu'),
                                                tf.keras.layers.Dense(128,activation='relu'),
                                                tf.keras.layers.Dense(64,activation='relu'),
                                                tf.keras.layers.Dense(1,activation='sigmoid')])
    
    self.reconstructor=tf.keras.models.Sequential([
                                                  #  tf.keras.layers.Dense(128,activation='relu'),
                                                  #  tf.keras.layers.Dense(128,activation='relu'),
                                                  #  tf.keras.layers.Dense(128,activation='relu'),
                                                  #  tf.keras.layers.Dense(32,activation='relu'),
                                                   tf.keras.layers.Dense(2,activation=tf.keras.activations.linear)
                                                   ])
    
    self.class_predictor=tf.keras.models.Sequential([self.extractor,self.classifier])
    self.data_reconstructor=tf.keras.models.Sequential([self.extractor,self.reconstructor])

    self.classifier_lr=classifier_lr
    self.extractor_lr=extractor_lr
    self.reconstructor_lr=reconstructor_lr

    self.classifier_opt=tf.keras.optimizers.Adam(self.classifier_lr)
    self.extractor_opt=tf.keras.optimizers.Adam(self.extractor_lr)
    self.reconstructor_opt=tf.keras.optimizers.Adam(self.reconstructor_lr)

    self.bce_loss=tf.keras.losses.BinaryCrossentropy()
    self.distance_loss=tf.keras.losses.MeanAbsoluteError()

    self.class_acc=tf.keras.metrics.BinaryAccuracy()
    self.class_loss=tf.keras.metrics.Mean()
    self.rec_loss=tf.keras.metrics.Mean()

    self.alpha=alpha
  
  @tf.function
  def train_cross_domain(self,source_x,source_y,target_x):
    self.class_acc.reset_states()
    self.class_loss.reset_states()
    self.rec_loss.reset_states()

    combined_x=tf.concat((source_x,target_x),axis=0)
    combined_x=tf.random.shuffle(combined_x)

    source_y=tf.convert_to_tensor(source_y,dtype=tf.int64)

    with tf.GradientTape(persistent=True) as tape:
      class_pred=self.class_predictor(source_x)
      class_loss=self.bce_loss(source_y,class_pred)

      reconstructed_x=self.data_reconstructor(combined_x)
      reconstruction_loss=self.distance_loss(combined_x,reconstructed_x)
      reconstruction_loss*=self.alpha

    classifier_gradient=tape.gradient(class_loss,self.class_predictor.trainable_variables)
    reconstructor_gradient=tape.gradient(reconstruction_loss,self.data_reconstructor.trainable_variables)

    self.classifier_opt.apply_gradients(zip(classifier_gradient,self.class_predictor.trainable_variables))
    self.reconstructor_opt.apply_gradients(zip(reconstructor_gradient,self.data_reconstructor.trainable_variables))
    
    self.class_loss(class_loss)
    self.class_acc(source_y,class_pred)
    self.rec_loss(reconstruction_loss)

  def evaluate_classifier_acc(self,x,ytrue,threshold=0.5):
    scores=self.class_predictor.predict(x)
    ypred=scores>threshold
    ypred=ypred.astype(int)
    ypred=ypred.flatten()
    acc=np.mean(ypred==ytrue)
    return acc    

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