import os
from .clip_encoder import CLIPVisionTower
import transcorem.config_param as config_param

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    if config_param.vision_model_path:
        vision_tower = config_param.vision_model_path

    if not os.path.isabs(vision_tower) and config_param.model_path and not vision_tower.startswith("openai") and not vision_tower.startswith("laion") and not vision_tower.startswith("PCIResearch"):
        vision_tower = os.path.join(config_param.model_path, vision_tower)

    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or vision_tower.startswith("PCIResearch"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
