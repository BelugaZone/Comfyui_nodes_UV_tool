"""
ComfyUI Custom Nodes — image_alpha_node.py
放置路径：ComfyUI/custom_nodes/image_alpha_node.py

包含两个节点：
  LoadImageRGBA  — 加载图像直接输出 4 通道 RGBA（等效 LoadImage + 合并图像Alpha）
  FlattenToBlack — 将 RGBA 透明背景合成为纯黑背景，输出 3 通道 RGB
"""

import os
import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence

import folder_paths


# ─────────────────────── Load Image RGBA ─────────────────────────

class LoadImageRGBANode:

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = sorted([
            f for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ])
        return {
            "required": {
                "image": (files, {"image_upload": True}),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("rgba",)
    FUNCTION      = "load"
    CATEGORY      = "UV Tools"

    @classmethod
    def IS_CHANGED(cls, image):
        return folder_paths.get_annotated_filepath(image)

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        if not folder_paths.exists_annotated_filepath(image):
            return f"文件不存在：{image}"
        return True

    def load(self, image):
        path     = folder_paths.get_annotated_filepath(image)
        img      = ImageSequence.Iterator(Image.open(path))[0]
        img      = ImageOps.exif_transpose(img).convert("RGBA")
        arr      = np.array(img, dtype=np.float32) / 255.0
        return (torch.from_numpy(arr).unsqueeze(0).clamp(0.0, 1.0),)


# ─────────────────────── Flatten To Black ────────────────────────

class FlattenToBlackNode:
    """
    将 RGBA 透明背景图像合成到纯黑背景，输出 3 通道 RGB。
    透明区域变为黑色，半透明区域按 alpha 混合。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rgba": ("IMAGE", {}),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)
    FUNCTION      = "flatten"
    CATEGORY      = "UV Tools"

    def flatten(self, rgba: torch.Tensor):
        if rgba.shape[-1] != 4:
            raise ValueError(
                f"输入需为 RGBA（4 通道），当前为 {rgba.shape[-1]} 通道。"
            )
        rgb   = rgba[:, :, :, :3]
        alpha = rgba[:, :, :, 3:4]          # (B, H, W, 1)
        # 黑底合成：result = rgb * alpha + 0 * (1 - alpha)
        result = (rgb * alpha).clamp(0.0, 1.0)
        return (result,)


# ─────────────────────── 注册 ─────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "LoadImageRGBA":  LoadImageRGBANode,
    "FlattenToBlack": FlattenToBlackNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageRGBA":  "Load Image RGBA",
    "FlattenToBlack": "Flatten To Black",
}

