import numpy as np
from tensorflow.compat import v1 as tf
from sklearn.manifold import TSNE
from model import model
from model import get_data
import matplotlib.pyplot as plt


x_train, y_train, x_val, y_val, x_test, y_test = get_data()
x, y, output, y_pred_cls, global_step, learning_rate, is_training, fc, flat = model()


_BATCH_SIZE = 128
_CLASS_SIZE = 10
_SAVE_PATH = "./"


saver = tf.train.Saver()
sess = tf.Session()



last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
saver.restore(sess, save_path=last_chk_path)
print("Restored checkpoint from:", last_chk_path)


def prepare_data(PATH):
    layer_output = np.load(PATH)
    tsne_model = TSNE(n_components =2)
    new_values = tsne_model.fit_transform(layer_output)
    return new_values[:,0], new_values[:,1]

def scatter_tsne(comp1,comp2,y_test,classes):
    plt.figure(figsize=(20,15))
    color_map = plt.cm.get_cmap('Accent')


    #plot labels
    labels = np.array(classes)[y_test]
    class_num = set()
    for x1,x2,c,l in zip(comp1,comp2,color_map(y_test),labels):
        plt.scatter(x1,x2,c=[c],label=l)
        class_num.add(l)
        

    plt.title('Flatten Output At epoch 60')
    plt.xlabel('Component One')
    plt.ylabel('Component Two')
    plt.savefig('./tsne_plot_flatten_60.png') 
    plt.show()

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# comp1 = []
# comp2 = []
# for i in range(20):
#     temp1, temp2 = prepare_data('./flatten{}_31.npy'.format(i))
#     comp1.append(temp1)
#     comp2.append(temp2)

# comp1 = np.concatenate(comp1)
# comp2 = np.concatenate(comp2)
# scatter_tsne(comp1,comp2,y_test,classes)


def main():
    i = 0
    predicted_class = np.zeros(shape=len(x_test), dtype=np.int)
    while i < len(x_test):
        j = min(i + _BATCH_SIZE, len(x_test))
        batch_xs = x_test[i:j, :].reshape(-1, 32*32*3)
        batch_ys = y_test[i:j, :]
        fully, flatten, predicted_class[i:j] = sess.run([fc, flat, y_pred_cls], feed_dict={is_training: False, x: batch_xs, y: batch_ys})
        i = j

    correct = (np.argmax(y_test, axis=1) == predicted_class)
    acc = correct.mean() * 100
    correct_numbers = correct.sum()
    print()
    print("Test Accuracy: {0:.2f}% )".format(acc))


if __name__ == "__main__":
    main()


sess.close()