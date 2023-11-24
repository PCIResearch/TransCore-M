from transcorem.mm_utils import get_model_name_from_path
from transcorem.eval.run_transcorem import eval_model

import argparse
import transcorem.config_param as config_param


## function: A script that implements the transcorem model inference example.


parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="./weights/transcorem_pci-v1.1_20231112")
parser.add_argument("--image-file", type=str, default="./picture/baby.jpeg")
parser.add_argument("--prompt", type=str, default="Please briefly describe the picture.")
parser.add_argument("--vision-path", type=str, default=None)
args = parser.parse_args()

# set MM model path
model_path = args.model_path

image_file = args.image_file
prompt = args.prompt

# set clip model path.
if args.vision_path:
    config_param.vision_model_path = args.vision_path

args_params = {"model_path": model_path, "model_base": None, 
    "query": prompt, "conv_mode": None, "image_file": image_file, "sep": ","}

output_result = eval_model(args_params)
print('output_result: ---------------------------')
print(output_result)
print('---------------------------')
