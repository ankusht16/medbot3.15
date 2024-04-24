import tensorflow as tf
model=tf.keras.models.load_model('chatbot_model.h5')
model.summary()
