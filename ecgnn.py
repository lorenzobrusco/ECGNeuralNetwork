from keras.models import load_model
import cnn

model = cnn.create_model()
cnn.training(model)
#model = load_model('model.h5')


