from keras.models import load_model
import cnn


"""
    In order to see the the information about 
    the neural network performe this command:
    tensorboard --logdir=logs
"""

model = cnn.create_model()
cnn.training(model, augmentation=True)
