import os
import tensorflow as tf

def recall(y_true, y_pred):
    return tf.keras.backend.sum(y_true-y_pred)
def precision(y_true, y_pred):
    return tf.keras.backend.sum(y_true-y_pred)
def localization_loss(y_true, y_pred):
    return tf.keras.backend.sum(y_true-y_pred)
def confidence_loss(y_true, y_pred):
    return tf.keras.backend.sum(y_true-y_pred)
def detection_loss(y_true, y_pred):
    return tf.keras.backend.sum(y_true-y_pred)

# NOTES:
# This function is tested on TF 1.14
# It has issues in TF 2.0-alpha 
# see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/python/lite.py
def keras_to_tflite_v1(keras_file, tflite_file, custom_objects):
    converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file, custom_objects=custom_objects)
    tflite_model = converter.convert()
    open(tflite_file, "wb").write(tflite_model)

# NOTES:
# This function is tested on TF 1.14
# It has issues in TF 2.0-alpha 
def keras_to_tflite_v2(keras_file, tflite_file, custom_objects):
    tf.keras.backend.clear_session()
    tf.keras.backend.set_learning_phase(False)
    keras_model = tf.keras.models.load_model(keras_file, custom_objects=custom_objects)
    sess = tf.keras.backend.get_session()
    converter = tf.lite.TFLiteConverter.from_session(sess, keras_model.inputs, keras_model.outputs)
    tflite_model = converter.convert()
    open(tflite_file, "wb").write(tflite_model)



if __name__ == "__main__":
    keras_file = os.path.dirname(os.path.abspath(__file__)) + "/../datasets/widerface/face_model_v4_520.h5"
    tflite_file = os.path.dirname(os.path.abspath(__file__)) + "/../datasets/widerface/face_model_v4_520.tflite"
    custom_objects = {
        "detection_loss": detection_loss, 
        "confidence_loss": confidence_loss, 
        "localization_loss": localization_loss,
        "precision": precision,
        "recall": recall
    }
    keras_to_tflite_v1(keras_file, tflite_file, custom_objects)
