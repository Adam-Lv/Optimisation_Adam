This program contains five python scripts and two folders. The functions will be illustrated below.

<< cnn.py >>
It's the head file of the program. In the script, there are two classes, ModelError and
HandwrittenNumeralRecognition (noted as HNR). ModelError is an error which may raised in the class
HNR, HNR is the base class of the program, all useful functions are encapsulated in this class.

<< adam.py >>
This is the optimizer Adam of the convolution neutral network of HNR.

<< configuration.py >>
All parameters of the model. Change the value in this program is exactly change the value of the model.

<< main.py >>
The main function of the program. It needs a parameter 'name' which represents the name of the model.
In order to run the program, you just need to make a name for the model and execute this function.

<< fig >>
This folder is the set of all output figures of the program. The name of the figure is the same as the
model's name.

<< model >>
This folder contains some csv files and h5 files.
csv files are the parameters of the model. The terminate accuracy and loss are also stored in these files.
h5 files are the trained models of tensorflow. You can use "tf.keras.models.load_model(model_dir)" to load
these models to reproduce my results.

<< test.py >>
This script is used to reproduce the results (terminate loss and accuracy). You just need to input the cores-
-bonding model name (e.x. 'model_0') in test function.


You have two choices to run the program:
** 1. Use "main.py":
      You should firstly set the parameters in "configuration.py", and you need to make a name for the model
      as the parameter of "main" function. It is recommended to use the name different from all names from
      "model" folder.
      You will obtain a picture in "fig", a configuration file in "model" and a h5 file in "model". All these
      files are the same name as the parameter called in "main" function.
** 2. Use "test.py":
      If you use this method, you can only obtain the result loss and accuracy of a model from folder "model".
      And you should only pass in the model's name which is already exists in the folder "model".