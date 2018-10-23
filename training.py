import os
import numpy as np
import tensorflow as tf
import inputs_data
import model
from PIL import Image
import matplotlib.pyplot as plt

#  File dir
train_dir = '/Users/vanson/Downloads/Boston University/EC601/tensorflow/data2/train/'
test_dir = '/Users/vanson/Downloads/Boston University/EC601/tensorflow/data2/test/'
train_logs_dir = '/Users/vanson/Downloads/Boston University/EC601/tensorflow/logs2/train/'

# The parameter
N_CLASSES = 2
IMG_W = 208
IMG_H = 208
BATCH_SIZE = 16
CAPACITY = 256
MAX_STEP = 1500
learning_rate = 0.0001


def run_training():


    #Put the image into the batch
    train, train_label = inputs_data.get_files(train_dir)
    train_batch, train_label_batch = inputs_data.get_batch(train,
                                                          train_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE,
                                                          CAPACITY)

    train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = model.losses(train_logits, train_label_batch)
    train_op = model.trainning(train_loss, learning_rate)
    train__acc = model.evaluation(train_logits, train_label_batch)
    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(train_logs_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                    break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])

            if step % 50 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 200 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(train_logs_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()

# train
# run_training()


def get_one_image(file_dir):


    test =[]
    for file in os.listdir(file_dir):
        test.append(file_dir + file)
    print('There are %d test pictures\n' %(len(test)))

    n = len(test)
    ind = np.random.randint(0, n)
    print(ind)
    img_test = test[ind]

    image = Image.open(img_test)
    plt.imshow(image)
    # plt.show()
    image = image.resize([208, 208])
    image = np.array(image)
    return image

def test_one_image():


    test_image = get_one_image(test_dir)

    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2

        image = tf.cast(test_image, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 208, 208, 3])
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)

        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[208, 208, 3])

        saver = tf.train.Saver()

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(train_logs_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            prediction = sess.run(logit, feed_dict={x: test_image})
            max_index = np.argmax(prediction)
            if max_index==0:
                print('This is a daisy with possibility %.6f' %prediction[:, 0])
            else:
                print('This is a rose with possibility %.6f' %prediction[:, 1])

        plt.show()


def evaluate_all_image():

    N_CLASSES = 2
    print('-------------------------')
    test, test_label = inputs_data.get_files(test_dir)
    BATCH_SIZE = len(test)
    print('There are %d test images totally..' % BATCH_SIZE)
    print('-------------------------')
    test_batch, test_label_batch = inputs_data.get_batch(test,
                                                        test_label,
                                                        IMG_W,
                                                        IMG_H,
                                                        BATCH_SIZE,
                                                        CAPACITY)

    logits = model.inference(test_batch, BATCH_SIZE, N_CLASSES)
    testloss = model.losses(logits, test_label_batch)
    testacc = model.evaluation(logits, test_label_batch)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        print("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(train_logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %s' % global_step)
        else:
            print('No checkpoint file found')
        print('-------------------------')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        test_loss, test_acc = sess.run([testloss, testacc])
        print('The model\'s loss is %.2f' % test_loss)
        correct = int(BATCH_SIZE * test_acc)
        print('Correct : %d' % correct)
        print('Wrong : %d' % (BATCH_SIZE - correct))
        print('The accuracy in test images are %.2f%%' % (test_acc * 100.0))
    coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    # run_training()
    # test_one_image()
    evaluate_all_image()
