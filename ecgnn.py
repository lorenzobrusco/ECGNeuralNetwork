import convert_signal_img as csi
from keras.models import load_model
import cnn



# csi.create_img_from_sign(size=_size, augmentation=False)
model = cnn.create_model()
cnn.training(model)
#model = load_model('model.h5')


