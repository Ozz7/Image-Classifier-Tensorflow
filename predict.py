# PROGRAMMER: Ajay Sukumar
# DATE CREATED:  03/06/2020
from PIL import Image
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import os
from time import time
import json
import numpy as np
from tensorflow.keras import regularizers
import collections
import warnings
warnings.filterwarnings('ignore')
tfds.disable_progress_bar()
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path' , type = str , help = 'Path to the test image')
    parser.add_argument('saved_model' , type = str , help = 'Path to the saved model')
    parser.add_argument('--top_k' , type = int , help = 'top K most likely classes', default = '5')
    parser.add_argument('--category_names' , type = str, help = 'path to json mapping of categories to flower names', default = './label_map.json')
    return parser.parse_args()

def get_model(model_path):
    loaded_model = tf.keras.models.load_model((model_path),custom_objects={'KerasLayer':hub.KerasLayer})
    return loaded_model

def process_image(image):
    image = np.squeeze(image)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    return image

def load_class_names(category_names):
    with open(category_names, 'r') as f:
        class_names = json.load(f)

    class_names = collections.OrderedDict(sorted(class_names.items()))
    return class_names

def prediction(in_args, model):
    class_names = load_class_names(in_args.category_names)
    im = Image.open(in_args.image_path)
    image = np.asarray(im)
    processed_image = process_image(image).numpy()
    ps = model.predict(np.expand_dims(processed_image, axis = 0))
    top_k_values, top_k_indices = tf.nn.top_k(ps, k=in_args.top_k)
    top_classes = [class_names[str(value+1)] for value in top_k_indices.cpu().numpy()[0]]
    return top_k_values.numpy()[0], top_classes

def show_predictions(probs, class_names):
    for i in range(len(probs)):
        print("Probability that image is of type {} is {:.3f}.".format(
            class_names[i], probs[i]))

def main():
    start_time = time()
    in_args = get_input_args()
    print('\nLoading the saved model...\n')
    model = get_model(in_args.saved_model)
    print("\nSuccessfully loaded\n")
    print(model.summary())
    print("\nFinding the class of input image...")
    probs, class_names = prediction(in_args,model)
    show_predictions(probs, class_names)
    end_time = time()
    tot_time = end_time - start_time
    print("\n** Total Runtime:",
          str(int((tot_time / 3600))) + ":" + str(
              int((tot_time % 3600) / 60)) + ":"
          + str(int((tot_time % 3600) % 60)))
    print("DONE...\n\n")
    
    
if __name__ == '__main__':
    main()