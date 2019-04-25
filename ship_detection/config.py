import os

LOG_FOLDER = r"E:\CNN\logs\retinanet"

DATA_FOLDER = r'E:\CNN\object_detection\retinanet'

COCO_WEIGHTS_PATH = r"E:\CNN\models\retinanet\resnet50_coco_best_v2.1.0.h5"

SHIP_DETECTION_WEIGHTS_PATH = r"E:\CNN\models\retinanet\resnet50_ships_0.5_1.0_1.5_small_inf.h5"

COCO_GROUND_TRUTH_PATH = os.path.join(DATA_FOLDER, "coco_ships_eval.json")

EVAL_IMAGE_INPUT_FOLDER = os.path.join(DATA_FOLDER, "images", "eval")

TRAINING_ANNOTATIONS_PATH = os.path.join(DATA_FOLDER, "training.csv")
VALIDATION_ANNOTATIONS_PATH = os.path.join(DATA_FOLDER, "validation.csv")
ANNOTATION_LABELS_PATH = os.path.join(DATA_FOLDER, "annotations.txt") 

RUN_NRS = ["1st"] # , "2nd", "3rd"

CONFIG_KEYS = ["large"] # , "small"

CONFIGS = {
    "small": {
        "resnet_blocks": [3, 4, 6],
        "resnet_outputs": lambda outputs: outputs,
        "sizes": [16, 32, 64, 128, 256],
        "strides": [4, 8, 16, 32, 64],
        "pyramid_levels": [2, 3, 4, 5, 6],
        "scales": None
    },
    "large": {
        "resnet_blocks": None, # use defaults
        "resnet_outputs": lambda outputs: outputs[1:],
        "sizes": [32, 64, 128, 256, 512],
        "strides": [8, 16, 32, 64, 128],
        "pyramid_levels": [3, 4, 5, 6, 7],
        "scales": None
    }
}

SCALE_ARRAYS = [
    # [2**0, 2**(1/3), 2**(2/3)],
    [0.5, 1.0, 1.5],
    # [0.25, 0.5, 1.0],
    # [0.25, 0.5, 1.0, 2.0],
    # [0.125, 0.25, 0.5, 1.0],
    # [0.0625, 0.125, 0.25, 0.5, 1.0]
]


EPOCHS = 2
BATCH_SIZE = 1
STEPS = 294
IMAGE_MIN_SIDE = 800
IMAGE_MAX_SIDE = 1200
BACKBONE_NAME = "resnet50"

COCO_CATEGORY_ID = 1
COCO_RESULTS_FILE_NAME = "coco_results.json"
MODEL_FILE_NAME = "%s_csv.h5" % BACKBONE_NAME
INFERENCE_MODEL_FILE_NAME = "%s_csv_inf.h5" % BACKBONE_NAME

INFERENCE_THRESHOLD = 0.7


def get_snapshot_folder(run_nr, config_key, scales):
    scales_string = "_".join(map(str, scales))
    
    return os.path.join(LOG_FOLDER, "%s_run_%s_%s" % (run_nr, config_key, scales_string))
            


config = {}

def get(name):
    return config.get(name)

def set_config(config_):
    global config
    config = config_
    