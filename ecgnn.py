import convert_signal_img as csi
from keras.models import load_model
import cnn

_size = (32, 32)

# csi.create_img_from_sign(size=_size, augmentation=False)
#model = cnn.create_model(_size)
#cnn.training(model, _size)
model = load_model('model.h5')
print(model.summary())

