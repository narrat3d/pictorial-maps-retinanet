import os
from keras_retinanet.bin import convert_model
from ship_detection import config


for run_nr in config.RUN_NRS:
    for scales in config.SCALE_ARRAYS:
        for config_key in config.CONFIG_KEYS:
            training_config = config.CONFIGS[config_key]
            training_config["scales"] = scales
            config.set_config(training_config)
            
            snapshot_folder = config.get_snapshot_folder(run_nr, config_key, scales) 
            
            input_model_path = os.path.join(snapshot_folder, config.MODEL_FILE_NAME)
            
            if (not os.path.exists(input_model_path)):
                continue
            
            print (input_model_path)
            
            output_model_path = os.path.join(snapshot_folder, config.INFERENCE_MODEL_FILE_NAME)
            
            convert_model.main([
                input_model_path,
                output_model_path,
                "--backbone=%s" % config.BACKBONE_NAME,
                "--config=dummy"
            ])