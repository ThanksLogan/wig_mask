import os
import cv2
import numpy as np
import torch

from .selfie_processing.pipeline import generate_wig_region_mask
from .selfie_processing.mask_ops import overlay_mask_preview
from .selfie_processing.types import WigMaskConfig

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(NODE_DIR, "models", "face_landmarker.task")
DEFAULT_TEMPLATE_MASK_PATH = os.path.join(NODE_DIR, "images", "template_mask.png")
DEFAULT_TEMPLATE_META_PATH = os.path.join(NODE_DIR, "images", "template_mask_meta.json")


def tensor_to_bgr_u8(image_tensor):
    # Comfy IMAGE is usually [B,H,W,C] float32 in [0,1]
    img = image_tensor.detach().cpu().numpy()
    rgb = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def bgr_u8_to_comfy_image(image_bgr):
    if not isinstance(image_bgr, np.ndarray):
        raise TypeError(f"Expected preview image as numpy.ndarray, got {type(image_bgr).__name__}")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb_f = rgb.astype(np.float32) / 255.0
    return torch.from_numpy(rgb_f)[None, ...]


def debug_to_preview_bgr(image_bgr, mask_u8, debug):
    if isinstance(debug, np.ndarray):
        return debug

    debug_mask = getattr(debug, "refined_mask", None)
    if isinstance(debug_mask, np.ndarray):
        return overlay_mask_preview(image_bgr, debug_mask)

    return overlay_mask_preview(image_bgr, mask_u8)


class WigMaskNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["wig", "face"],),
                "template_face_cutout_inset_px": ("INT", {"default": 4, "min": 0, "max": 64}),
            },
            "optional": {
                "model_path": ("STRING", {"default": DEFAULT_MODEL_PATH}),
                "template_mask_path": ("STRING", {"default": DEFAULT_TEMPLATE_MASK_PATH}),
                "template_meta_path": ("STRING", {"default": DEFAULT_TEMPLATE_META_PATH}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "preview")
    FUNCTION = "run"
    CATEGORY = "WigAI"

    def run(
        self,
        image,
        mode,
        template_face_cutout_inset_px,
        model_path=DEFAULT_MODEL_PATH,
        template_mask_path=DEFAULT_TEMPLATE_MASK_PATH,
        template_meta_path=DEFAULT_TEMPLATE_META_PATH,
    ):
        if image.ndim != 4:
            raise ValueError(f"Expected IMAGE tensor with 4 dims [B,H,W,C], got shape {tuple(image.shape)}")

        masks = []
        previews = []

        for i in range(image.shape[0]):
            bgr = tensor_to_bgr_u8(image[i])

            cfg = WigMaskConfig(
                mask_mode=mode,
                face_landmarker_model_path=model_path or None,
                template_mask_path=template_mask_path or None,
                template_meta_path=template_meta_path or None,
                template_face_cutout_inset_px=int(template_face_cutout_inset_px),
            )

            mask_u8, debug = generate_wig_region_mask(bgr, cfg)

            mask_f = torch.from_numpy(mask_u8.astype(np.float32) / 255.0)
            masks.append(mask_f)

            preview_bgr = debug_to_preview_bgr(bgr, mask_u8, debug)
            previews.append(bgr_u8_to_comfy_image(preview_bgr))

        mask_batch = torch.stack(masks, dim=0)
        preview_batch = torch.cat(previews, dim=0)

        return (mask_batch, preview_batch)


NODE_CLASS_MAPPINGS = {
    "WigMaskNode": WigMaskNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WigMaskNode": "Wig Mask (Template+MP)",
}
