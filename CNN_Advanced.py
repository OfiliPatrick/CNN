import tensorflow as tf
import numpy as np
import os
import random
import sys
import cv2
import tensorflow_hub as hub
def images_converted(image_path, labs):
    image=tf.io.read_file(image_path)
    image=tf.image.decode_jpeg(image)
    image=tf.image.resize(image, size=(224, 224))
    image=tf.reshape(image, shape=(224, 224, 3))
    image=tf.cast(image, dtype=tf.uint8)
    return image, labels


def image_batcher (image_list, labels):
    training_datasets=tf.data.Dataset.from_tensor_slices((image_list, labels))
    training_datasets=training_datasets.map(images_converted)
    training_datasets=training_datasets.repeat()
    training_datasets=training_datasets.batch(32)
    training_datasets=training_datasets.prefetch(buffer_size=-1)
    return training_datasets

def image_lister (parent_directory, storage_box=None):
    if storage_box==None:
        main_box=[]
    else:
        main_box=storage_box
    for _, _, files in os.walk(parent_directory):
        for i in range (len(files)):
            main_box.append(
                os.path.join(
                    parent_directory,
                    files[i]
                )
            )
    return main_box

def image_processor (image_path, storage_area_labels, storage_area_path=None):
  os.makedirs('images_folder5')
  sub_dirs=os.listdir(image_path)

  if storage_area_path==None:
    storage=[]
  else:
    storage=storage_area

  for i in range (len(sub_dirs)):
    nums=i
    sub_path=os.path.join(
        image_path,
        sub_dirs[i]
    )

    for _, _, files in os.walk(sub_path):

      for m in range (len(files)):
        img=cv2.imread(image_path+'/'+sub_dirs[i]+'/'+files[m])
        cv2.imwrite(f'images_folder5/image_folder{m}.jpg', img)
        storage.append(f'images_folder5/image_folder{m}.jpg')
        storage_area_labels.append(nums)
  return (
      storage_area_labels,
      storage
  )
gt_boxes=[]
(labels, paths)=image_processor(image_path='#Your train_data',storage_area_labels=gt_boxes)

pre=image_batcher(image_list=paths, labels=labels)
MODULE_HANDLE = 'https://tfhub.dev/google/efficientnet/b5/feature-vector/1'

model = tf.keras.Sequential([
    hub.KerasLayer(MODULE_HANDLE, input_shape=(224, 224, 3)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

optimizers=tf.keras.optimizers.Adam()
loss=tf.keras.losses.binary_crossentropy
accuracy=tf.keras.metrics.BinaryAccuracy()

def training_one_step (model, optimizer, x, y, main_loss):
  with tf.GradientTape() as tape:
    print (3)
    loss=main_loss(y_true=y, y_pred=model(x))
    print (4)
  grads=tape.gradient(loss, model.trainable_weights)
  optimizers.apply_gradients(zip(grads, model.trainable_weights))
  return loss

@tf.function()
def training (model, optimizer, epochs, train_ds, acc, loss):
  for i in range (epochs):
    print (1)
    for (x, y) in enumerate(train_ds):
      print (2)
      loss=training_one_step(model=model, optimizer=optimizer, main_loss=loss, x=x, y=y)
      print (loss)

training(model=model, optimizer=optimizers, epochs=2, train_ds=train_ds, acc=accuracy, loss=loss)

model('#Please pass your prediction image over here')
