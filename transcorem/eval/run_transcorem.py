import argparse
import torch
from abc import abstractproperty

from transcorem.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from transcorem.conversation import conv_templates, SeparatorStyle
from transcorem.model.builder import load_pretrained_model
from transcorem.utils import disable_torch_init
from transcorem.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, highres_process_images

from PIL import Image

import requests
from PIL import Image
from io import BytesIO


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(args_params):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args_params["model_path"])
    tokenizer, model, image_processor, context_len = load_pretrained_model(args_params["model_path"], args_params["model_base"], model_name)

    qs = args_params["query"]
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    if "v1" in model_name.lower():  # wsy.add
        print("==============conv_mode: transcorem_v1")
        conv_mode = "transcorem_v1"
    else:
        print("==============conv_mode: transcorem_v0")
        conv_mode = "transcorem_v0"

    if args_params["conv_mode"] is not None and conv_mode != args_params["conv_mode"]:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args_params["conv_mode"], args_params["conv_mode"]))
    else:
        args_params["conv_mode"] = conv_mode

    conv = conv_templates[args_params["conv_mode"]].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image = load_image(args_params["image_file"])
    args = abstractproperty()
    args.image_aspect_ratio = 'pad'
    image_tensor = highres_process_images(image, image_processor, args, base_reso=336)
    image_tensor = [patch.unsqueeze(0).to("cuda", dtype=torch.float16) for patch in image_tensor]
    # image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=2048,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs
