import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ship_detection import config
import json

aggregated_results = {}

def calculate_results(coco_ground_truth_path, coco_results_path):
    cocoGt=COCO(coco_ground_truth_path)
    cocoDt=cocoGt.loadRes(coco_results_path)
    
    imgIds = sorted(cocoGt.getImgIds())
    
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    return cocoEval.stats

def filter_coco_results(coco_results_path, coco_filtered_results_path):
    coco_results = json.load(open(coco_results_path))
    
    filtered_coco_results = list(filter(lambda coco_result: coco_result["score"] > 0.3,
                                 coco_results))

    json.dump(filtered_coco_results, open(coco_filtered_results_path, "w"))

    
if __name__ == '__main__':
    coco_results_root_folder = config.LOG_FOLDER # r"E:\CNN\logs\faster_rcnn"
    folder_names = os.listdir(coco_results_root_folder)
    
    for folder_name in folder_names:
        coco_results_path = os.path.join(coco_results_root_folder, folder_name, config.COCO_RESULTS_FILE_NAME)
        
        if (os.path.exists(coco_results_path)):
            # coco_filtered_results_path = coco_results_path.replace(".json", "_filtered.json")
            # filter_coco_results(coco_results_path, coco_filtered_results_path)
            # results = calculate_results(config.COCO_GROUND_TRUTH_PATH, coco_filtered_results_path)
            results = calculate_results(config.COCO_GROUND_TRUTH_PATH, coco_results_path)
            
            # http://cocodataset.org/#detection-eval
            # ap = results[0]
            # ap_50 = results[1]
            # ap_75 = results[2]
            # ap_small = results[3]
            # ap_medium = results[4]
            # ap_large = results[5]
            
            for config_key in config.CONFIG_KEYS:
                if (folder_name.find(config_key) != -1):
                    results_for_scales = aggregated_results.setdefault(config_key, {})
                    
                    parts = folder_name.split("_%s_" % config_key)
                    scales_string = parts[1]
                    
                    results_array = results_for_scales.setdefault(scales_string, [])
                    results_array.append(results)
    
    
    for config_key, results_for_scales in aggregated_results.items():
        print (config_key)
        
        for scales_string, results_array in results_for_scales.items():
            scales_comma = ", ".join(scales_string.split("_"))
            
            average_results = np.mean(np.array(results_array), axis=0).tolist()
            average_results = average_results[0:6]
            average_results_string = "\t".join(map(lambda result: str(round(result * 100, 2)) + "%", average_results))
            
            print ("%s\t%s" % (scales_comma, average_results_string))