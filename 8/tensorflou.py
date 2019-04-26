import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def initer():
    imdb = keras.datasets.imdb

    word_index = wordindexer(imdb)    

    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
        value=word_index["<PAD>"],
        padding='post',
        maxlen=256)

    test_data = keras.preprocessing.sequence.pad_sequences(test_data,
        value=word_index["<PAD>"],
        padding='post',
        maxlen=256)


    return imdb, train_data, train_labels, test_data, test_labels

def transform_into_text():
    imdb = initer()[0]

    word_index = wordindexer(imdb)
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    return reverse_word_index

def decode_review(text):
    reverse_word_index = transform_into_text()
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

def wordindexer(imdb):
    # A dictionary mapping words to an integer index
    word_index = imdb.get_word_index()

    # The first indices are reserved
    word_index = {k:(v+3) for k,v in word_index.items()} 
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3

    return word_index

def model_former():
    # vocabulary count used for the movie reviews (10,000 words)
    vocab_size = 10000

    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    model.summary()

    # configure the model to use an optimizer and a loss function
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

    return model


def main():
    imdb, train_data, train_labels, test_data, test_labels = initer()
    
    # print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
    print(train_data[0])
    model = model_former()

    # I use the 10000 first value of the training dataset, the test_data will be used to evaluate the accuracy
    x_val = train_data[:10000]
    partial_x_train = train_data[10000:]

    y_val = train_labels[:10000]
    partial_y_train = train_labels[10000:]

    # 40 iterations (epochs)
    history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

    results = model.evaluate(test_data, test_labels)
    # print(results)

    history_dict = history.history
    history_dict.keys()

    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(15,9))
    plt.subplot(211)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'ro', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(212)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()



if __name__ == "__main__":
    main()


""" 
    model = tf.keras.Sequential()
    
    # Adds a densely-connected layer with 64 units to the model:
    layers.Dense(64, activation='relu', input_shape=(32,)),
    # Add another:
    layers.Dense(64, activation='relu'),
    # Add a softmax layer with 10 output units:
    layers.Dense(10, activation='softmax')])

    model.compile(optimizer=tf.optimizers.Adam(0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Configure a model for mean-squared error regression.
    model.compile(optimizer=tf.optimizers.Adam(0.01),
                loss='mse',       # mean squared error
                metrics=['mae'])  # mean absolute error

    # Configure a model for categorical classification.
    model.compile(optimizer=tf.optimizers.RMSprop(0.01),
                loss=tf.keras.losses.categorical_crossentropy,
                metrics=[tf.keras.metrics.categorical_accuracy])

    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))

    val_data = np.random.random((100, 32))
    val_labels = np.random.random((100, 10))

    # Instantiates a toy dataset instance:
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.batch(32).repeat()

    val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
    val_dataset = val_dataset.batch(32).repeat()

    # Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
    model.fit(dataset, epochs=10, steps_per_epoch=30,
            validation_data=val_dataset,
            validation_steps=3)


    model.evaluate(data, labels, batch_size=32)

    model.evaluate(dataset, steps=30)
"""