from cnn import HandwrittenNumeralRecognition
import tensorflow as tf
model_dir = './model/'


def test(name):
    """The test function for reproducing the result."""

    # Create an object from HNR. The constructor of the class will initialize
    # the MNIST dataset and optimizer.
    hnr_demo = HandwrittenNumeralRecognition()
    # Load the model from a h5 file. Parameter <<name>> is type of string.
    model = tf.keras.models.load_model(model_dir + name + '.h5')
    hnr_demo.create_model(model)
    # Show the model's summary.
    hnr_demo.show_summary()
    # Show the result of evaluation
    hnr_demo.evaluate()
    return


if __name__ == '__main__':
    test('model_0')
