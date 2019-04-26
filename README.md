# pictorial-maps-retinanet

## Installation (for Windows)

* Requires Python 3.6.x (https://www.python.org/downloads/)
* Requires CUDA Toolkit 9.0 (https://developer.nvidia.com/cuda-downloads) and corresponding cuDNN (https://developer.nvidia.com/rdp/cudnn-download)
* Download project
* Download and install Microsoft Build Tools (see “All Downloads” in https://visualstudio.microsoft.com/downloads/)
* pip install -r requirements.txt
* pip install \<path to this project>
* cd \<path to this project>
* python setup.py build_ext --inplace


## Inference

* Download trained model and set SHIP_DETECTION_WEIGHTS_PATH to the downloaded model in config.py
* Run detect_ships.py \<input folder with images of historic map> \<output folder for text files and images with detected bounding boxes>


## Training

* Download training data (= images + bounding box annotations) and set DATA_FOLDER to the downloaded folder in config.py
* Download [trained coco weights](https://github.com/fizyr/keras-retinanet/releases/download/0.5.0/resnet50_coco_best_v2.1.0.h5) for RetinaNet and set COCO_WEIGHTS_PATH to the downloaded model in config.py
* Adjust LOG_FOLDER in config.py. The trained models will be stored in this folder.
* Optionally adjust properties like scales (e.g. SCALE_ARRAYS = [[2&ast;&ast;0, 2&ast;&ast;(1/3), 2&ast;&ast;(2/3)]]), number of runs (e.g. RUN_NRS = ["1st"]), configuration in config.py (CONFIG_KEYS = ["small"])
* Run training.py to train the ship detector


## Evaluation

* Run model_converter.py to convert trained models into inference models
* Run evaluation.py to predict bounding boxes and scores for detecting ships (optionally enable the save_image flag to visualize detected and ground truth bounding boxes on the images)
* Run coco_metrics.py to calculate COCO metrics


## Source
* https://github.com/fizyr/keras-retinanet (Apache License, Copyright by Hans Gaiser)


#### Modifications
* Use of higher ResNet blocks in models\resnet.py and higher pyramid levels in utils\anchors.py to detect smaller objects on images
* Parametrization of scales, strides, and sizes so that it can be trained in multiples runs with different configurations