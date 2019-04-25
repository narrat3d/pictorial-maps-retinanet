import os
import shutil
from keras_retinanet.bin import train
from ship_detection import config

for run_nr in config.RUN_NRS:
    for config_key in config.CONFIG_KEYS:
        training_config = config.CONFIGS[config_key]
            
        for scales in config.SCALE_ARRAYS:
            training_config["scales"] = scales
            config.set_config(training_config)
            
            snapshot_folder = config.get_snapshot_folder(run_nr, config_key, scales) 
            
            if (os.path.exists(snapshot_folder)):
                shutil.rmtree(snapshot_folder)
            
            os.mkdir(snapshot_folder)
        
            train.main([
                '--steps=%s' % config.STEPS,
                '--backbone=%s' % config.BACKBONE_NAME,
                '--batch-size=%s' % config.BATCH_SIZE,
                '--epochs=%s' % config.EPOCHS,
                r'--weights=%s' % config.COCO_WEIGHTS_PATH,
                r'--snapshot-path=%s' % snapshot_folder,
                r'--tensorboard-dir=%s' % snapshot_folder,
                '--image-min-side=%s' % config.IMAGE_MIN_SIDE,
                '--image-max-side=%s' % config.IMAGE_MAX_SIDE,
                '--config=dummy',
                'csv',
                config.TRAINING_ANNOTATIONS_PATH,
                config.ANNOTATION_LABELS_PATH,
                '--val-annotations=%s' % config.VALIDATION_ANNOTATIONS_PATH
            ]) #    '--no-snapshots',