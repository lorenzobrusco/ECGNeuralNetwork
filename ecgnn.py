from keras.models import load_model
import cnn

model = cnn.create_model()
cnn.training(model, shuffle=True, augmentation=True)
# cnn.training_old(model, shuffle=True)
