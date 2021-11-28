from model import model
from model import lr
from model import get_data
from model import augment_training_data
import tensorflow as tf2
import numpy as np
import math
import tensorflow.compat.v1 as tf
from statistics import mean


x_train, y_train, x_val, y_val, x_test, y_test = get_data()

data_augmentation = False

if(data_augmentation==True):
    x_train, y_train = augment_training_data(x_train, y_train)

x, y, output, y_pred_cls, global_step, learning_rate,is_training, fc, flat = model()
global_accuracy = 0
epoch_start = 0


# PARAMS
_BATCH_SIZE = 128
_EPOCH = 60
_SAVE_PATH = "./"

# Loss function
loss = tf.reduce_mean(tf2.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))

#optimizer
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)


# Accuracy calculation
correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Save Model
merged = tf.summary.merge_all()
saver = tf.train.Saver()
sess = tf.Session()
train_writer = tf.summary.FileWriter(_SAVE_PATH, sess.graph)



sess.run(tf.global_variables_initializer())

def train(epoch, average_loss, average_training):
    global epoch_start
    batch_size = int(math.ceil(len(x_train) / _BATCH_SIZE))
    i_global = 0

    for s in range(batch_size):
        batch_xs = x_train[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE].reshape(-1, 32*32*3)
        batch_ys = y_train[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]

        i_global, _, batch_loss, batch_acc, fully_connected, flatten = sess.run(
            [global_step, optimizer, loss, accuracy, fc, flat],
            feed_dict={x: batch_xs, y: batch_ys, learning_rate: lr(epoch), is_training: True})

        average_training.append(batch_acc)
        average_loss.append(batch_loss)

    if(epoch % 15 == 0):

        batch_size_test = int(math.ceil(len(x_train) / 500))

        for i in range(batch_size_test):

            batch_tests = x_test[i*500: (i+1)*500].reshape(-1, 32*32*3)
            batch_ytes = y_test[i*500: (i+1)*500]

            _,_,_,_, fully_connected, flatten = sess.run(
                [global_step, optimizer, loss, accuracy, fc, flat],
                feed_dict={x: batch_tests, y: batch_ytes, learning_rate: lr(epoch), is_training: False})

            np.save('./flatten{}_{}.npy'.format(i,epoch+1), flatten)
            np.save('./fully{}_connected_{}.npy'.format(i,epoch+1), fully_connected)

    validation_and_save(i_global, epoch)

    return mean(average_loss), mean(average_training)



def validation_and_save(_global_step, epoch):
    global global_accuracy
    global epoch_start

    i = 0
    predicted_class = np.zeros(shape=len(x_val), dtype=np.int)
    while i < len(x_val):
        j = min(i + _BATCH_SIZE, len(x_val))
        batch_xs = x_val[i:j, :].reshape(-1, 32*32*3)
        batch_ys = y_val[i:j, :]
        predicted_class[i:j] = sess.run(
            y_pred_cls,
            feed_dict={x: batch_xs, y: batch_ys, learning_rate: lr(epoch), is_training: False}
        )
        i = j


    counter = 0
    labels = np.argmax(y_val, axis=1)

    for i in range(len(labels)):
        if (labels[i]==predicted_class[i]):
            counter+=1
    
    acc = counter/len(y_val)

    if global_accuracy != 0 and global_accuracy < acc:

        summary = tf.Summary(value=[
            tf.Summary.Value(tag="Accuracy/validation", simple_value=acc),
        ])
        train_writer.add_summary(summary, _global_step)

        saver.save(sess, save_path=_SAVE_PATH, global_step=_global_step)

        mes = "This epoch has higher validation accuracy. Save session."
        print(mes.format(acc, global_accuracy))
        global_accuracy = acc

    elif global_accuracy == 0:
        global_accuracy = acc

    print("###########################################################################################################")
    print("Validation Accuracy: {}".format(acc))


def main():

    for i in range(_EPOCH):
        average_loss = []
        average_training = []
        print("\nEpoch: {}/{}\n".format((i+1), _EPOCH))
        epoch_loss, epoch_training = train(i, average_loss, average_training)
        print("Average Epoch Loss: {}, Average Traning Accuracy: {}".format(epoch_loss, epoch_training))




if __name__ == "__main__":
    main()


sess.close()