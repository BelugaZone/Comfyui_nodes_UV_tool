"""
ComfyUI Custom Node — Transparent Seam Padding
放置路径：ComfyUI/custom_nodes/uv_seam_padding_node.py

对透明背景 RGBA 图像执行最近邻边缘洪泛填充（UV Padding / Island Bleeding）。
直接从图像的 alpha 通道识别内容区域，无需额外 mask 输入。

输入：
  image      : RGBA 图像，(B, H, W, 4) float32 [0,1]，alpha=0 为透明背景
  padding_px : 向外扩展的像素数（默认 15，0=填充全部透明区域）

输出：
  image : padding 后的 RGBA 图像，(B, H, W, 4) float32 [0,1]，填充区 alpha=1

依赖：scipy
"""

import torch
import numpy as np

try:
    from scipy.ndimage import distance_transform_edt as _edt
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


def _pad_one(rgba: np.ndarray, padding_px: int) -> np.ndarray:
    """
    对单张 RGBA float32 图执行透明背景最近邻洪泛填充。
    采样源限定为 alpha==1 的完全不透明像素，确保颜色无黑色混合偏差。
    填充区 alpha 置为 1，未填充的透明背景保持 alpha=0。
    """
    alpha = rgba[:, :, 3]                    # (H, W) float32 [0,1]
    rgb   = rgba[:, :, :3]                   # (H, W, 3)

    is_content  = alpha > 0.0
    source_mask = alpha >= (254.0 / 255.0)   # 完全不透明像素作采样源
    if not source_mask.any():
        source_mask = is_content

    if not is_content.any():
        return rgba.copy()
    if is_content.all():
        return rgba.copy()

    if padding_px > 0:
        fill_mask = (~is_content) & (_edt(~is_content) <= padding_px)
    else:
        fill_mask = ~is_content

    _, src_idx    = _edt(~source_mask, return_indices=True)
    nearest_rgb   = rgb[src_idx[0], src_idx[1]]

    result        = rgba.copy()
    result[fill_mask, :3] = nearest_rgb[fill_mask]
    result[fill_mask,  3] = 1.0              # 填充区变为完全不透明

    return result


class TransparentSeamPaddingNode:
    """
    透明背景 RGBA 图像的最近邻洪泛填充（UV Seam Padding）。
    直接读取图像 alpha 通道，无需额外 mask 输入。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":      ("IMAGE", {}),
                "padding_px": ("INT", {
                    "default": 15, "min": 0, "max": 2048, "step": 1,
                    "tooltip": "向外扩展像素数，0 = 填充全部透明区域"
                }),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)
    FUNCTION      = "process"
    CATEGORY      = "UV Tools"

    def process(self, image: torch.Tensor, padding_px: int):
        if not _SCIPY_OK:
            raise RuntimeError("需要 scipy：pip install scipy")

        # 确认输入为 4 通道 RGBA
        if image.shape[-1] != 4:
            raise ValueError(
                f"输入图像需为 RGBA（4 通道），当前为 {image.shape[-1]} 通道。\n"
                "请使用支持 alpha 输出的加载节点（如 Load Image w/ Alpha）。"
            )

        B = image.shape[0]
        out = []
        for i in range(B):
            rgba_np = image[i].cpu().numpy().astype(np.float32)   # (H, W, 4)
            padded  = _pad_one(rgba_np, padding_px)
            out.append(torch.from_numpy(padded))

        return (torch.stack(out, dim=0).clamp(0.0, 1.0),)


NODE_CLASS_MAPPINGS = {
    "TransparentSeamPadding": TransparentSeamPaddingNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TransparentSeamPadding": "Transparent Seam Padding",
}
