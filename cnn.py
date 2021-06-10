import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from adam import Adam
import configuration as cf
import matplotlib.pyplot as plt
import pandas as pd


class ModelError(Exception):
    """An error that may be called in class HandwrittenNumeralRecognition"""
    pass


class HandwrittenNumeralRecognition:
    def __init__(self):
        self.__load_dataset()
        self.optimizer = Adam(cf.learning_rate, cf.beta_1, cf.beta_2, cf.epsilon)
        self.model = None
        self.history = None
        self.loss = None
        self.accuracy = None
        return

    def __load_dataset(self):
        # Read data
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Reshape the data to 4-dimension
        x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
        x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

        # Normalization
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0

        # Record the dataset
        self.__x_train = x_train
        self.__x_test = x_test
        self.__y_train = y_train
        self.__y_test = y_test
        return

    def create_default_model(self, activation1='relu', activation2='softmax'):
        """
        Creat a CNN model, which is the default model in the class HandwrittenNumeralRecognition.
        The default network is a sequential model:
            Input layer:
            A tensor of shape (28, 28, 1)

            Conv2D layer 1:
            Default: 8 kernels of size 5*5 with activation 'relu', padding: 'same'

            MaxPooling2D layer 1:
            Default: pooling size 2*2

            Conv2D layer 2:
            Default: 16 kernels of size 5*5 with activation 'relu', padding: 'same'

            MaxPooling2D layer 2:
            Default: pooling size 2*2

            Fully connected layer 1:
            Default: reshape the input into (128, 1) with activation 'relu'

            Dropout regularization:
            Default: dropout rate: 0.25

            Fully connected layer 2 (output layer):
            Default: reshape the input into (10, 1) with activation 'softmax'

        :param activation1: The activation function of the first fully connected layer. Default: 'relu'
        :param activation2: The activation function of the second fully connected layer. Default: 'softmax'
        :return: A CNN model of tensorflow
        """
        self.model = Sequential()

        # First module of convolution, contains one conv layer and one pooling layer
        self.model.add(Conv2D(name='Conv2D_1',
                              filters=cf.kernel_nums1,
                              kernel_size=(cf.kernel_size, cf.kernel_size),
                              padding=cf.padding,
                              activation=activation1))
        self.model.add(MaxPooling2D(name='MaxPooling2D_1',
                                    pool_size=(cf.pooling_size, cf.pooling_size)))

        # Second module of convolution, contains one conv layer and one pooling layer
        self.model.add(Conv2D(name='Conv2D_2',
                              filters=cf.kernel_nums2,
                              kernel_size=(cf.kernel_size, cf.kernel_size),
                              padding=cf.padding,
                              activation=activation1))
        self.model.add(MaxPooling2D(name='MaxPooling2D_2',
                                    pool_size=(cf.pooling_size, cf.pooling_size)))

        # Flatten layer, shift the data into one dimension
        self.model.add(Flatten())

        # First connection complete layer
        self.model.add(Dense(cf.dense_size, activation1))

        # Regularization layer, using dropout
        self.model.add(Dropout(cf.drop_rate))

        # Output layer
        self.model.add(Dense(cf.label_nums, activation2))
        return

    def create_model(self, model):
        """Create the model from 'configuration.py'."""
        self.model = model
        if not self.model:
            raise ModelError("You have not build a model in configuration.py!")
        return

    def show_summary(self):
        """Print the summary of the model, but it can be called only after the model is fitted."""
        if self.model:
            self.model.summary()
            return
        raise ModelError("Please create a model first")

    def compile(self):
        """
        Compile the model, by using the parameters in 'configuration.py'.

        Default parameters:
            loss = 'sparse_categorical_crossentropy'.
        """
        if self.model:
            self.model.compile(loss=cf.loss_func,
                               optimizer=self.optimizer,
                               metrics=['accuracy'])
            return
        raise ModelError("Please create a model first")

    def fit(self):
        """
        Fit the model, by using the parameters in 'configuration.py'.

        Default parameters:
            epochs = 10,
            batch_size = 32.
        """
        if self.model:
            self.history = self.model.fit(x=self.__x_train,
                                          y=self.__y_train,
                                          validation_split=0.2,
                                          epochs=cf.epochs,
                                          batch_size=cf.batch_size,
                                          verbose=1)
            return
        raise ModelError("Please create a model first")

    def save_model(self, name):
        """
        Save the model as a file h5.

        :param name: The file name of the model (without extensions)
        """
        self.model.save('./model/' + name + '.h5')
        return

    def evaluate(self):
        """Evaluate the model by the test set. Output the accuracy."""
        print("\n", "Evaluation of model:")
        score = self.model.evaluate(self.__x_test, self.__y_test, verbose=1)
        self.loss, self.accuracy = score[0], score[1]
        return

    def visualization(self, name, save_fig=False):
        """
        Draw the figure which illustrates the accuracy and the loss in the
        process of training. You must guarantee that there is a folder named
        'fig' in the root directory.

        :param name: The file name of the figure (without extensions)
        :param save_fig: Boolean, if true, the figure will saved in the folder 'fig'
        """
        plt.figure(figsize=(17, 8))
        plt.suptitle(name, font='Times New Roman', fontsize=20)
        plt.subplot(121)
        plt.plot(np.arange(cf.epochs) + 1, self.history.history['accuracy'],
                 marker='o', label='train_accuracy')
        plt.plot(np.arange(cf.epochs) + 1, self.history.history['val_accuracy'],
                 marker='o', label='test_accuracy')
        plt.ylim(0.9, 1)
        plt.xlim(0.5, cf.epochs + 0.5)
        plt.title('Accuracy of the training process', fontsize=16)
        plt.xticks(list(range(1, cf.epochs + 1)))
        plt.yticks(np.linspace(0.9, 1, 11))
        plt.xlabel('Epoch number', fontsize=12)
        plt.grid(linestyle='--')
        plt.legend()
        plt.subplot(122)
        plt.plot(np.arange(cf.epochs) + 1, self.history.history['loss'],
                 marker='o', label='train_accuracy')
        plt.plot(np.arange(cf.epochs) + 1, self.history.history['val_loss'],
                 marker='o', label='test_accuracy')
        plt.ylim(0, 0.3)
        plt.xlim(0.5, cf.epochs + 0.5)
        plt.title('Loss of the training process', fontsize=16)
        plt.xticks(list(range(1, cf.epochs + 1)))
        plt.xlabel('Epoch number', fontsize=12)
        plt.grid(linestyle='--')
        plt.legend()
        if save_fig:
            plt.savefig('./fig/' + name + '.png')
        plt.show()
        return

    def write_config(self, name):
        """
        Save the model's config as a file csv.

        :param name: the file name
        """
        config_dict = dict(
            kernel_size=cf.kernel_size,
            kernel_nums1=cf.kernel_nums1,
            kernel_nums2=cf.kernel_nums2,
            padding=cf.padding,
            strides=cf.strides,
            pooling_size=cf.pooling_size,
            dense_size=cf.dense_size,
            drop_rate=cf.drop_rate,
            label_nums=cf.label_nums,
            loss_func=cf.loss_func,
            epochs=cf.epochs,
            batch_size=cf.batch_size,
            learning_rate=cf.learning_rate,
            beta_1=cf.beta_1,
            beta_2=cf.beta_2,
            epsilon=cf.epsilon,
            loss=self.loss,
            accuracy=self.accuracy
        )
        temp_db = pd.DataFrame(data=config_dict.values(),
                               index=config_dict.keys(), columns=['value'])
        temp_db.to_csv('./model/' + name + '_config.csv')
        print("Successfully stored the configuration!")
        return
