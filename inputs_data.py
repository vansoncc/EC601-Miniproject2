import tensorflow as tf
import os
import numpy as np

def get_files(file_dir):
    daisy = []
    label_daisy = []
    rose = []
    label_rose = []
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if 'daisy' in name[0]:
            daisy.append(file_dir + file)
            label_daisy.append(0)
        if 'rose' in name[0]:
            rose.append(file_dir + file)
            label_rose.append(1)
        else:
            pass
        image_list = np.hstack((daisy,rose))
        label_list = np.hstack((label_daisy,label_rose))

    # Random the image sequence for training
    temp = np.array([image_list,label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(i) for i in label_list]
    return image_list,label_list


# Generate the batch image_w = Image Width, image_H = Image Height, batch_size = how many image in a batch,
def get_batch(image,label,image_W,image_H,batch_size,capacity):
    image = tf.cast(image,tf.string)
    label = tf.cast(label, tf.int32)

    input_queue = tf.train.slice_input_producer([image,label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents,channels =3)
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],batch_size = batch_size,
                                                num_threads = 64, capacity = capacity)

    label_batch = tf.reshape(label_batch , [batch_size])
    image_batch = tf.cast(image_batch,tf.float32)
    return  image_batch, label_batch

