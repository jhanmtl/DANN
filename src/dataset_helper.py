import pandas as pd
import numpy as np
import tensorflow as tf
import os

def load_subjects(npy_dir,
                  seen_subj,
                  seen_device,
                  seen_sensor,
                  unseen_subj,
                  unseen_device,
                  unseen_sensor,
                  act_int_lookup,
                  act_subset=None):

  seen_x,seen_y=load_npy(npy_dir,seen_subj,seen_device,seen_sensor)
  unseen_x,unseen_y=load_npy(npy_dir,unseen_subj,unseen_device,unseen_sensor)

  if act_subset is not None:
    int_subset=[act_int_lookup[i] for i in act_subset]
    
    seen_mask=[i in int_subset for i in seen_y]
    unseen_mask=[i in int_subset for i in unseen_y]   

    seen_x=seen_x[seen_mask]
    seen_y=seen_y[seen_mask]

    unseen_x=unseen_x[unseen_mask]
    unseen_y=unseen_y[unseen_mask]    

    old_labels=np.unique(seen_y)
    label_dict={old_labels[i]:i for i in range(len(old_labels))}

    remapped_seen_y=[label_dict[i] for i in seen_y]
    remapped_unseen_y=[label_dict[i] for i in unseen_y]

    seen_y=np.array(remapped_seen_y)
    unseen_y=np.array(remapped_unseen_y) 

  num_class=len(np.unique(seen_y))
  return (seen_x,seen_y),(unseen_x,unseen_y),num_class

  
def load_npy(datadir,subj,device,sensor):
  datapath=os.path.join(datadir,"{}_{}_{}_data.npy".format(subj,device,sensor))
  labelpath=os.path.join(datadir,"{}_{}_{}_label.npy".format(subj,device,sensor))

  x=np.load(datapath)
  y=np.load(labelpath)

  return x,y

def read_csv(fpath,letter_act_lookup,act_int_lookup):
  df=pd.read_csv(fpath,header=None,lineterminator=";")
  df=df[:-1]

  df.columns=['subj','act','time','x','y','z']

  for key in letter_act_lookup:
    df=df.replace(key,letter_act_lookup[key])

  for key in act_int_lookup:
      df['act']=df['act'].replace(key,act_int_lookup[key])
  
  return df

def split_by_act(df,act):
  subdf=df[df['act']==act]
  subdf=subdf[['act','x','y','z']]

  x=subdf.drop(columns='act')
  x=x.values

  y=subdf['act'].values

  return x,y


def _bytes_feature(value):
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() 
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def timeseries_dataset_to_tfrecord(ds,writepath):

  tfwriter=tf.io.TFRecordWriter(writepath)

  for data,label in ds:
    for window_data,window_label in zip(data,label):

      serialized_data=tf.io.serialize_tensor(window_data)
      tf_example=tf.train.Example(
          features=tf.train.Features(
              feature={
                  'data':_bytes_feature(serialized_data),
                  'label':_int64_feature(window_label)
              }

          )
      )

      tfwriter.write(tf_example.SerializeToString())

def decode_timeseries_tfrecord(serialized_example):
  feature_map = {
                'data': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64,default_value=0),
                }
  features=tf.io.parse_single_example(serialized_example,feature_map)

  serialized_data=features['data']
  
  data=tf.io.parse_tensor(serialized_data,out_type=tf.float64)
  data=tf.dtypes.cast(data,tf.float64)
  
  label=features['label']

  return data,label


def consistency_check(original_ds,recordpath,batchsize,tol=1e-9):
  reload_ds=tf.data.TFRecordDataset([recordpath])
  reload_ds=reload_ds.map(decode_timeseries_tfrecord)
  reload_ds=reload_ds.batch(batchsize)

  for (original_data,original_label),(reload_data,reload_label) in zip(original_ds,reload_ds):
    data_diff=tf.reduce_sum(original_data-reload_data).numpy()
    data_diff=np.abs(data_diff).item()

    label_diff=tf.reduce_sum(original_label-reload_label).numpy().item()
    label_diff=np.abs(data_diff).item()

    assert data_diff<tol
    assert label_diff<tol
  








