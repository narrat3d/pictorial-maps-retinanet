import json
import keras
import os

from keras_retinanet import models
from ship_detection import config
from ship_detection.detect_ships import infer, visualize_detections, get_session


def get_bounding_boxes(coco_ground_truth_path, coco_category_id, inference_model_path, 
                       coco_results_path, image_input_folder, image_output_folder, save_image=False):
    coco_image_ids = {}
    coco_image_bboxes = {}
    coco_results = []
    
    with open(coco_ground_truth_path) as jsonfile:
        coco_gt = json.load(jsonfile)
        
        for image in coco_gt["images"]:
            coco_image_ids[image["file_name"]] = image["id"]
        
        for annotation in coco_gt["annotations"]:
            bboxes = coco_image_bboxes.setdefault(annotation["image_id"], []) 
            bboxes.append(annotation["bbox"])
    
    inference_model = models.load_model(inference_model_path, backbone_name=config.BACKBONE_NAME)
    inference_model.layers.pop()
    
    for image_name in os.listdir(image_input_folder):
        print (image_name)
        
        image_path = os.path.join(image_input_folder, image_name)
        bounding_boxes, scores = infer(inference_model, image_path)
         
        image_id = coco_image_ids[image_name]
        
        for box, score in zip(bounding_boxes, scores):    
            box_min_x = box[0]
            box_min_y = box[1]
            box_max_x = box[2]
            box_max_y = box[3]
    
            coco_results.append({
                "image_id": image_id, 
                "category_id" : coco_category_id, 
                "bbox" : [box_min_x, box_min_y, box_max_x - box_min_x, box_max_y - box_min_y], 
                "score" : score
            })
        
        if (save_image):
            if (not os.path.exists(image_output_folder)):
                os.mkdir(image_output_folder)
            
            image_name_without_ext = os.path.splitext(image_name)[0]
            
            image_pil, _ = visualize_detections(image_path, bounding_boxes, scores, (0,255,255,0))
            image_pil.save(os.path.join(image_output_folder, image_name_without_ext + ".png"))
            
            gt_boxes = coco_image_bboxes.get(image_id, [])
            
            gt_bounding_boxes = list(map(lambda box: [box[0], box[1], box[0] + box[2], box[1] + box[3]], gt_boxes))
            gt_scores = [1.0] * len(gt_bounding_boxes)
            
            image_pil, _ = visualize_detections(image_path, gt_bounding_boxes, gt_scores, (255,0,255,0))
            image_pil.save(os.path.join(image_output_folder, image_name_without_ext + "_gt.png"))     
    
    with open(coco_results_path, "w") as jsonfile:
        json.dump(coco_results, jsonfile)


if __name__ == '__main__':
    keras.backend.tensorflow_backend.set_session(get_session())
    
    log_folder = config.LOG_FOLDER
    folder_names = os.listdir(log_folder)
    
    for folder_name in folder_names:
        inference_model_path = os.path.join(log_folder, folder_name, config.INFERENCE_MODEL_FILE_NAME)
        coco_results_path = os.path.join(log_folder, folder_name, config.COCO_RESULTS_FILE_NAME)
        image_output_folder = os.path.join(log_folder, folder_name, "images")
        
        if (os.path.exists(inference_model_path) and not os.path.exists(coco_results_path)):
            get_bounding_boxes(config.COCO_GROUND_TRUTH_PATH, config.COCO_CATEGORY_ID, inference_model_path, 
                               coco_results_path, config.EVAL_IMAGE_INPUT_FOLDER, image_output_folder, save_image=True)