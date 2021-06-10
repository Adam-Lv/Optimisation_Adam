from cnn import HandwrittenNumeralRecognition


def main(name):
    """The process for solving the problem. It's a series of events."""

    # Create an object from HNR. The constructor of the class will initialize
    # the MNIST dataset and optimizer.
    hnr_demo = HandwrittenNumeralRecognition()
    # Create the default CNN model
    hnr_demo.create_default_model()
    # Compile the model
    hnr_demo.compile()
    # Fit the model
    hnr_demo.fit()
    # Show a summary of the model. It will show the framework of the model
    hnr_demo.show_summary()
    # Evaluate the model by the whole test set
    hnr_demo.evaluate()
    # Save the configuration and the evaluation result (loss and accuracy) of the model
    hnr_demo.write_config(name)
    # Draw a picture of the training process
    hnr_demo.visualization(name, save_fig=True)
    # Save the model as a h5 file
    hnr_demo.save_model(name)
    return


if __name__ == '__main__':
    main('model_0')
