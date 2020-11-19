import os
import sys
import getopt
import tensorflow as tf
import tensorflow.keras as keras

# NOTES:
# This function is tested on TF 1.14
# It has issues in TF 2.0
# see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/python/lite.py
# import tensorflow.keras.backend as K
# def recall(y_true, y_pred):
#     return K.sum(y_true-y_pred)
# def precision(y_true, y_pred):
#     return K.sum(y_true-y_pred)
# def localization_loss(y_true, y_pred):
#     return K.sum(y_true-y_pred)
# def confidence_loss(y_true, y_pred):
#     return K.sum(y_true-y_pred)
# def detection_loss(y_true, y_pred):
#     return K.sum(y_true-y_pred)
# def keras_to_tflite_v1(keras_file, tflite_file, custom_objects):
#     print("convert %s to %s ..." % (keras_file, tflite_file))
#     converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file, custom_objects=custom_objects)
#     tflite_model = converter.convert()
#     open(tflite_file, "wb").write(tflite_model)
#     print("finished")

# NOTES:
# This function is tested on TF 2.0
# see https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter#from_keras_model
def keras_to_tflite_v2(keras_file, tflite_file):
    print("convert %s to %s ..." % (keras_file, tflite_file))
    model = keras.models.load_model(keras_file, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open(tflite_file, "wb").write(tflite_model)
    print("finished")

def usage():
    print("Usage: %s -i <keras-h5-file> -o <tflite-file>" % sys.argv[0])
    sys.exit(0)

def parse_arguments():
    try:
	    opts, _ = getopt.getopt(sys.argv[1:], "hi:o:", ["help", "input=" "output="])
    except getopt.GetoptError as err:
        print(err)
        usage()
        
    keras_file = os.path.dirname(os.path.abspath(__file__)) + "/models/face_model_v1_2100.h5"
    tflite_file = os.path.dirname(os.path.abspath(__file__)) + "/models/face_model_v1_2100.tflite" 
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
        elif o in ("-i", "--input"):
            keras_file = a
        elif o in ("-o", "--output"):
            tflite_file = a
    return keras_file, tflite_file

if __name__ == "__main__":
    keras_file, tflite_file = parse_arguments()
    keras_to_tflite_v2(keras_file, tflite_file)