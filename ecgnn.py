import cnn

"""
    In order to see the the information about 
    the neural network performe this command:
    tensorboard --logdir=logs
"""

cnn.training(augmentation=True)
cnn.predict_model()

