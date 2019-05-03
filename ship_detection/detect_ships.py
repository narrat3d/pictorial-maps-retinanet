import argparse
import os
import sys
import keras
import numpy as np
import tensorflow as tf

from PIL import Image, ImageDraw
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image,\
    resize_image
from ship_detection import config
import json


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def infer(inference_model, image_path):
    detected_bounding_boxes = []
    detection_scores = []

    # originally: 
    image = read_image_bgr(image_path)
        
    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image, min_side=config.IMAGE_MIN_SIDE, max_side=config.IMAGE_MAX_SIDE)
    image_array = np.expand_dims(image, axis=0)

    # process image
    result = inference_model.predict_on_batch(image_array)
    boxes, scores, labels = result
    
    # correct for image scale
    boxes /= scale

    # visualize detections
    for box, score, _ in zip(boxes[0], scores[0], labels[0]):
        if score == 0.0:
            break

        box_min_x = box[0].item()
        box_min_y = box[1].item()
        box_max_x = box[2].item()
        box_max_y = box[3].item()
        
        detected_bounding_boxes.append([round(box_min_x), round(box_min_y), round(box_max_x), round(box_max_y)])
        detection_scores.append(score.item())
    
    return detected_bounding_boxes, detection_scores          


def draw_bounding_box(image_draw, bounding_box, color):
    min_x, min_y, max_x, max_y = bounding_box
     
    for stroke_width in range(1, 9):
        image_draw.rectangle([min_x - stroke_width, min_y - stroke_width, 
                              max_x + stroke_width, max_y + stroke_width], outline=color)
        

def visualize_detections(image_path, bounding_boxes, scores, color):
    image_pil = Image.open(image_path)
    image_draw = ImageDraw.Draw(image_pil)

    bounding_boxes_array = []

    for bounding_box, score in zip(bounding_boxes, scores):
        if score > config.INFERENCE_THRESHOLD:
            draw_bounding_box(image_draw, bounding_box, color)
            
            bounding_boxes_array.append({
                "minX": bounding_box[0], 
                "minY": bounding_box[1], 
                "maxX": bounding_box[2],
                "maxY": bounding_box[3],
                "score": score
            })
                                         
        else :
            break
    
    return image_pil, bounding_boxes_array


def detect_ships(image_input_folder, image_output_folder, inference_model_path):
    keras.backend.tensorflow_backend.set_session(get_session())

    inference_model = models.load_model(inference_model_path, backbone_name=config.BACKBONE_NAME)
    inference_model.layers.pop()

    for image_name in os.listdir(image_input_folder):
        print (image_name)
        
        image_path = os.path.join(image_input_folder, image_name)
    
        bounding_boxes, scores = infer(inference_model, image_path)
        image_pil, bounding_boxes_array = visualize_detections(image_path, bounding_boxes, scores, (0,255,255,0))

        image_name_without_ext = os.path.splitext(image_name)[0]

        image_pil.save(os.path.join(image_output_folder, image_name_without_ext + ".png"))
        
        json_file = open(os.path.join(image_output_folder, image_name_without_ext + ".json"), 'w')
        json.dump(bounding_boxes_array, json_file)
            


def parse_args(args):
    parser = argparse.ArgumentParser(description='Detect ships on historic maps.')
    parser.add_argument('input_folder', type=str, help='Input folder with images.')
    parser.add_argument('output_folder', type=str, help='Folder where images and JSON files with bounding boxes will be copied.')
    args = parser.parse_args(args)
    
    if (not os.path.exists(args.input_folder)):
        raise Exception("The specified input folder does not exist.")
    
    if (not os.path.exists(args.output_folder)):
        raise Exception("The specified output folder does not exist.")
    
    return (args.input_folder, args.output_folder)


def main(args=None):
    if args is None:
        args = sys.argv[1:]
        
    (input_folder, output_folder) = parse_args(args)
    
    detect_ships(input_folder, output_folder, config.SHIP_DETECTION_WEIGHTS_PATH)


if __name__ == '__main__':
    main()