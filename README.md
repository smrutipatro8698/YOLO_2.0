YOLOv3 code with custom training data module

Build model:
1. Do cd config/
2. Run !create_custom_model.sh in Collab

Run training module:
1. Run !python3 train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data

Run test module:
