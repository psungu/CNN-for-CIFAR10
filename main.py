import time
from model import build_model
from model import get_CIFAR10_data
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
import numpy as np



x_train, y_train, x_val, y_val, x_test, y_test = get_CIFAR10_data()

datagen = ImageDataGenerator(
        rotation_range = 40,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        brightness_range = (0.5, 1.5))

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dicti = pickle.load(fo, encoding='bytes')
    return dicti
    

batch_size = 64

# train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)


test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = val_dataset.batch(batch_size)


datagen.fit(x_train)

# Get model
model = build_model()
epochs = 45


# Instantiate an optimizer to train the model.
optimizer = keras.optimizers.Adam()
# Instantiate a loss function.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Prepare the metrics.
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()


for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()
    batches = 0
    # Iterate over the batches of the dataset.
    for  (x_batch_train, y_batch_train) in datagen.flow(x_train, y_train, batch_size=64, seed=28):

        if batches >= len(x_train) / batch_size:
            break

        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)
            
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Update training metric.
        train_acc_metric.update_state(y_batch_train, logits)
        batches += 1


        if batches % 200 == 0:
            print("Training loss (for one batch) at step %d: %.4f"
                % (batches, float(loss_value)))

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()

# Run a validation loop at the end of each epoch.
    counter = 0
    prediction = model(x_val, training=False)
    predicted_labels = np.argmax(prediction, axis=1)
    for i in range(len(predicted_labels)):
        if (predicted_labels[i]==y_val[i]):
            counter+=1

    print(f'Epoch {epoch}, Validation Accuracy {counter/len(y_val)}')



counter = 0
prediction = model(x_test, training=False)
predicted_labels = np.argmax(prediction, axis=1)
for i in range(len(predicted_labels)):
    if (predicted_labels[i]==y_test[i]):
        counter+=1

print(f'Epoch {epoch}, Test Accuracy {counter/len(y_test)}')
