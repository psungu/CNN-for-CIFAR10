import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
from sklearn.utils import shuffle
from statistics import mean
from keras.preprocessing.image import ImageDataGenerator



def load_pickle(f):

    return  pickle.load(f, encoding='latin1')

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']

        X = X.reshape(10000,32,32,3)
        #Y = to_categorical(Y)
        Y =np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=45000, num_validation=5000, num_test=10000):
    # Load the raw CIFAR-10 data
    
    cifar10_dir = './cifar10_data'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    X_train, y_train = shuffle(X_train, y_train)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    x_train = X_train.astype('float32')
    x_test = X_test.astype('float32')
    X_val= X_val.astype('float32')

    x_train /= 255
    x_test /= 255
    X_val /= 255

    return x_train, y_train, X_val, y_val, x_test, y_test

leaky_relu_alpha = 0.2
dropout_rate = 0


def conv2d( inputs , filters , stride_size ):
    out = tf.nn.conv2d( inputs , filters , strides=[ 1 , stride_size , stride_size , 1 ] , padding='SAME' ) 
    return tf.nn.relu(out)

def maxpool( inputs , pool_size , stride_size ):
    return tf.nn.max_pool2d( inputs , ksize=[ 1 , pool_size , pool_size , 1 ] , padding='VALID' , strides=[ 1 , stride_size , stride_size , 1 ] )

def dense( inputs , weights ):
    x = tf.nn.relu(tf.matmul( inputs , weights ))
    return tf.nn.dropout( x , rate=dropout_rate )


initializer = tf.initializers.glorot_uniform()

def get_weight( shape , name ):
    return tf.Variable( initializer( shape ) , name=name , trainable=True , dtype=tf.float32 )

shapes = [
    [ 3 , 3 , 3 , 128 ] , 
    [ 3 , 3 , 128 , 64 ] ,
    [ 3 , 3 , 64 , 32 ] ,
    [ 512 , 64 ],
    [ 64 , 10]
]

print(tf.__version__) 

weights = []
for i in range( len( shapes ) ):
    weights.append( get_weight( shapes[ i ] , 'weight{}'.format( i ) ) )



def model( x ) :

    x = tf.cast( x , dtype=tf.float32 )
    c1 = conv2d( x , weights[ 0 ] , stride_size=1 ) 
    p1 = maxpool( c1 , pool_size=2 , stride_size=2 )
    
    c2 = conv2d( p1 , weights[ 1 ] , stride_size=1 )
    p2 = maxpool( c2 , pool_size=2 , stride_size=2 )
    
    c3 = conv2d( p2 , weights[ 2 ] , stride_size=1 ) 
    p3 = maxpool( c3 , pool_size=2 , stride_size=2 )

    flatten = tf.reshape( p3 , shape=( tf.shape( p3 )[0] , -1 ))

    d1 = dense( flatten , weights[ 3 ] )
    logits = tf.matmul( d1 , weights[ 4 ] )

    return tf.nn.softmax( logits )


def loss( pred , target ):
    
    return tf.compat.v1.losses.softmax_cross_entropy(target, pred)


optimizer = tf.compat.v1.train.AdamOptimizer()

def train_step(model, inputs, outputs):
    with tf.GradientTape() as tape:
        prediction = model(inputs)
        current_loss = loss(prediction, outputs)
    grads = tape.gradient(current_loss , weights)
    optimizer.apply_gradients( zip(grads , weights ))


num_epochs = 30
batch_size = 32

x_train, y_train, x_val, y_val, x_test, y_test = get_CIFAR10_data()

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)


datagen = ImageDataGenerator(
        rotation_range = 40,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        brightness_range = (0.5, 1.5))

datagen.fit(x_train)        

for epoch in range(num_epochs):

    average_training_accuracy = []

    for  (image, label) in train_dataset:
        

        with tf.GradientTape() as tape:
            predict = model(image)
            current_loss = loss(predict, tf.one_hot( label , depth=10))
        grads = tape.gradient(current_loss , weights)
        optimizer.apply_gradients( zip(grads , weights ))
        
        count = 0
        predict_labels = np.argmax(predict, axis=1)
        for i in range(len(label)):
            if (predict_labels[i]==label[i]):
                count+=1

        average_training_accuracy.append(count/len(label))
            
    print(f'Epoch {epoch}, Training Accuracy {mean(average_training_accuracy)}')

    

    counter = 0
    prediction = model(x_val)
    predicted_labels = np.argmax(prediction, axis=1)
    for i in range(len(predicted_labels)):
        if (predicted_labels[i]==y_val[i]):
            counter+=1

    print(f'Epoch {epoch}, Validation Accuracy {counter/len(y_val)}')



#Evaluation
counter = 0
prediction = model(x_test)
predicted_labels = np.argmax(prediction, axis=1)
for i in range(len(predicted_labels)):
    if (predicted_labels[i]==y_test[i]):
        counter+=1

print(f'Epoch {epoch}, Test Accuracy {counter/len(y_test)}')



filename = 'model.pk'
pickle.dump(weights, open(filename, 'wb'))