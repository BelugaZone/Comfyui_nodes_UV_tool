"""
ComfyUI Custom Node — Layer Merge
放置路径：ComfyUI/custom_nodes/layer_merge_node.py

将最多 15 张透明背景 RGBA 图像按 Photoshop "over" 混合从底到顶合并为单张 RGBA 图。
直接读取每张图像的 alpha 通道，配合 Join RGBA 节点使用。

输入：
  image_1    : 第 1 层（底层，必填），RGBA (1, H, W, 4) float32 [0,1]
  image_2~15 : 第 2~15 层（可选，不接则跳过）

输出：
  merged      : 合并后 RGBA 图像，(1, H, W, 4) float32 [0,1]
  merged_mask : 合并后掩码（有内容区域=1），(1, H, W) float32 [0,1]
"""

import torch


def _over(src_rgb, src_a, dst_rgb, dst_a):
    """Porter-Duff 'over'，直通 alpha，非预乘。"""
    out_a   = src_a + dst_a * (1.0 - src_a)
    sa      = src_a.unsqueeze(-1)
    da      = dst_a.unsqueeze(-1)
    oa      = out_a.unsqueeze(-1).clamp(min=1e-7)
    out_rgb = (src_rgb * sa + dst_rgb * da * (1.0 - sa)) / oa
    return out_rgb, out_a


class LayerMergeNode:

    @classmethod
    def INPUT_TYPES(cls):
        required = {"image_1": ("IMAGE", {})}
        optional = {f"image_{i}": ("IMAGE", {}) for i in range(2, 26)}
        return {"required": required, "optional": optional}

    RETURN_TYPES  = ("IMAGE", "MASK")
    RETURN_NAMES  = ("merged", "merged_mask")
    FUNCTION      = "merge"
    CATEGORY      = "UV Tools/Layer"

    def merge(self, image_1, **kwargs):
        layers = [image_1]
        for i in range(2, 26):
            img = kwargs.get(f"image_{i}")
            if img is not None:
                layers.append(img)

        if layers[0].shape[-1] != 4:
            raise ValueError(
                f"输入图像需为 RGBA（4 通道），当前为 {layers[0].shape[-1]} 通道。\n"
                "请先连接 Join RGBA 节点将 IMAGE + MASK 合并为 RGBA。"
            )

        H, W   = layers[0].shape[1], layers[0].shape[2]
        device  = layers[0].device

        result_rgb = torch.zeros(H, W, 3, dtype=torch.float32, device=device)
        result_a   = torch.zeros(H, W,    dtype=torch.float32, device=device)

        for layer in layers:
            src     = layer[0]           # (H, W, 4)
            src_rgb = src[:, :, :3]
            src_a   = src[:, :,  3]
            result_rgb, result_a = _over(src_rgb, src_a, result_rgb, result_a)

        result_rgb  = result_rgb.clamp(0.0, 1.0)
        result_a    = result_a.clamp(0.0, 1.0)
        merged_rgba = torch.cat([result_rgb, result_a.unsqueeze(-1)], dim=-1).unsqueeze(0)

        return (merged_rgba, result_a.unsqueeze(0))


NODE_CLASS_MAPPINGS        = {"LayerMerge": LayerMergeNode}
NODE_DISPLAY_NAME_MAPPINGS = {"LayerMerge": "Layer Merge"}
