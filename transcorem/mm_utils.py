from PIL import Image
from io import BytesIO
import base64

import torch
import numpy as np
from transformers import StoppingCriteria
from transcorem.constants import IMAGE_TOKEN_INDEX


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

def process_image(image, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    if image_aspect_ratio == 'pad':
        image = expand2square(image, tuple(int(x*255)
                              for x in image_processor.image_mean))
        image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        return image
    else:
        return image_processor(image, return_tensors='pt')['pixel_values'][0]

def sliding_windows(matrix, window_size, stride):
    height, width, c = matrix.shape
    window_rows = (height - window_size[0]) // stride + 1
    window_cols = (width - window_size[1]) // stride + 1
    windows = []
    for i in range(window_rows):
        for j in range(window_cols):
            window = matrix[i * stride:i * stride + window_size[0], j * stride:j * stride + window_size[1], :]
            windows.append(window)
    return windows

def get_patches(image, stride):
    image_array = np.array(image)
    windows = sliding_windows(image_array, window_size=(stride, stride), stride=stride)
    return windows

def highres_process_images(image, image_processor, model_cfg, base_reso):
    predefined_ratios = [(2, 2), (1, 4), (4, 1)]
    predefined_reso_list = [(w * base_reso, h * base_reso) for w, h in predefined_ratios]
    predefined_aspect_ratios = np.array([w / h for w, h in predefined_ratios])

    image_width, image_height = image.size
    aspect_ratio = image_width / image_height
    ar_errors = predefined_aspect_ratios - aspect_ratio
    predefined_bucket_id = np.abs(ar_errors).argmin()
    reso = predefined_reso_list[predefined_bucket_id]

    # need resize
    if image_width > reso[0] or image_height > reso[1]:
        image_resize = image.resize(reso, Image.Resampling.LANCZOS)
    else:
        image_resize = image.copy()

    w, h = image_resize.size
    result = Image.new(image_resize.mode, reso, tuple(int(x * 255) for x in image_processor.image_mean))
    paste_x = (reso[0] - w) // 2
    paste_y = (reso[1] - h) // 2
    result.paste(image_resize, (paste_x, paste_y))

    patches = get_patches(result, base_reso)
    patch_list = []
    for patch in patches:
        pt = image_processor.preprocess(patch, return_tensors="pt")["pixel_values"][0]
        patch_list.append(pt)

    global_feat = process_image(image, image_processor, model_cfg)
    patch_tensor = torch.stack(patch_list)
    local_feat = torch.mean(patch_tensor, dim=0)
    return local_feat, global_feat
