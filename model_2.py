
import tensorflow as tf
from tensorflow.python.keras.datasets import cifar10
from model import get_CIFAR10_data
import numpy as np
from model import build_model
from keras.preprocessing.image import ImageDataGenerator



def main():
    # Load CIFAR 10 dataset
    x_train, y_train, x_val, y_val, x_test, y_test = get_CIFAR10_data()
    
    datagen = ImageDataGenerator(
            # rotation_range = 40,
            # shear_range = 0.2,
            # zoom_range = 0.2,
            horizontal_flip = True,
            brightness_range = (0.5, 1.5))

    datagen.fit(x_train)

    batch_size = 32
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(x_train, tf.float32),
         tf.cast(y_train, tf.int64)))

    train_dataset = train_dataset.batch(batch_size)


    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(x_val, tf.float32),
         tf.cast(y_val, tf.int64)))


    test_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(x_test, tf.float32),
         tf.cast(y_test, tf.int64)))

    # Construct model

    model = build_model()

    # Training
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    loss_metric = tf.keras.metrics.Mean(name='train_loss')
    accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    accuracy_validation = tf.keras.metrics.SparseCategoricalAccuracy(name='validation_accuracy')

    train_loss_history = []
    accuracy_history = []

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            pred_loss = loss_fn(labels, predictions)

        grads = tape.gradient(pred_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        loss_metric.update_state(pred_loss)
        accuracy_metric.update_state(labels, predictions)


    for epoch in range(30):
        loss_metric.reset_states()
        accuracy_metric.reset_states()
        batches = 0
        for images, labels in datagen.flow(x_train, y_train, batch_size=64, seed=28):
            train_step(images, labels)
            batches += 1

            if batches >= len(x_train) / batch_size:
                break


        train_loss_history.append(loss_metric.result())
        accuracy_history.append(accuracy_metric.result())
        print(f'Epoch {epoch}, Loss {loss_metric.result()}, Accuracy {accuracy_metric.result()}')

        
        counter = 0
        prediction = model(x_val, training=False)
        predicted_labels = np.argmax(prediction, axis=1)
        for i in range(len(predicted_labels)):
            if (predicted_labels[i]==y_val[i]):
                counter+=1

        print(f'Epoch {epoch}, Validation Accuracy {counter/len(y_val)}')

    print("accuracy: ", accuracy_history)
    print("train loss: ", train_loss_history)

    counter = 0
    prediction = model(x_test, training=False)
    predicted_labels = np.argmax(prediction, axis=1)
    for i in range(len(predicted_labels)):
        if (predicted_labels[i]==y_test[i]):
            counter+=1

    print(f'Epoch {epoch}, Test Accuracy {counter/len(y_test)}')


if __name__ == '__main__':
    main()