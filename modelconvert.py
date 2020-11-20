import tensorflow as tf

modeltest = tf.keras.models.load_model('models/mypilot2')

modeltest.save('models/mypilot2.h5')
