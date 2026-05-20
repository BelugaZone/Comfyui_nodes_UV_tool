"""
ComfyUI Custom Node — UV Patch Extract
放置路径：ComfyUI/custom_nodes/uv_split_node.py

模拟手动拆分贴图流程：
  一张 UV 网格图对应一个部件，将整张网格洪水填充为实心掩码，
  应用到完整贴图上提取该部件的局部贴图。

输入：
  uv_mesh    : UV 网格图（透明背景+白色线框，ComfyUI 加载后白线在黑底上）
  uv_texture : 完整 UV 贴图（RGB）
  padding_px : 边缘洪泛填充像素数（默认 15）
  close_kernel: 形态学闭运算核大小，弥合线条断裂（默认 3）
  min_area   : 忽略面积小于此值的噪点轮廓（像素²，默认 100）

输出：
  patch : 纯黑背景 padding 后的局部贴图，(1, H, W, 3) float32 [0,1]
  mask  : 黑白掩码（白=内容区域），(1, H, W) float32 [0,1]

依赖：opencv-python, scipy
"""

import torch
import numpy as np
import cv2

try:
    from scipy.ndimage import distance_transform_edt as _edt
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


# ─────────────────────── 核心算法（自包含）────────────────────────

def _extract_line_mask(mesh_rgb: np.ndarray) -> np.ndarray:
    """
    从 RGB 网格图中提取二值线条掩码。
    ComfyUI LoadImage 将透明背景 PNG 合成到黑底，白色线框保持白色，
    用亮度阈值提取线条。
    """
    gray = cv2.cvtColor(mesh_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    return mask


def _flood_fill_solid(line_mask: np.ndarray) -> np.ndarray:
    """
    洪水填充法生成实心掩码。
    从四角向内漫延标记外部背景（128），未被标记区域（岛内部）取反得到实心掩码。
    内部网格线充当屏障，不影响轮廓填充结果。
    """
    h, w = line_mask.shape
    canvas = np.zeros((h + 2, w + 2), dtype=np.uint8)
    canvas[1:h+1, 1:w+1] = line_mask

    for seed in [(0, 0), (0, w + 1), (h + 1, 0), (h + 1, w + 1)]:
        if canvas[seed] == 0:
            cv2.floodFill(canvas, None, (seed[1], seed[0]), 128)

    inner = canvas[1:h+1, 1:w+1]
    return np.where(inner != 128, 255, 0).astype(np.uint8)


def _pad_transparent(rgba_arr: np.ndarray, padding_px: int) -> np.ndarray:
    """
    透明背景最近邻洪泛填充，输出 RGBA uint8。
    采样源限定为 alpha==255 像素（真实内容色，无黑色混合），填充区 alpha 置 255。
    """
    alpha = rgba_arr[:, :, 3]
    rgb   = rgba_arr[:, :, :3]

    is_content  = alpha > 0
    source_mask = alpha == 255
    if not source_mask.any():
        source_mask = is_content
    if not is_content.any():
        return rgba_arr.copy()

    fill_mask     = (~is_content) & (_edt(~is_content) <= padding_px)
    _, src_idx    = _edt(~source_mask, return_indices=True)
    nearest_color = rgb[src_idx[0], src_idx[1]]

    result_rgb   = rgb.copy();   result_rgb[fill_mask]   = nearest_color[fill_mask]
    result_alpha = alpha.copy(); result_alpha[fill_mask] = 255
    return np.dstack([result_rgb, result_alpha[:, :, None]]).astype(np.uint8)


def _rgba_to_black_bg(padded_rgba: np.ndarray) -> np.ndarray:
    """RGBA → 黑色背景 RGB（填充区保留颜色，透明区变黑）。"""
    black = np.zeros((*padded_rgba.shape[:2], 3), dtype=np.uint8)
    black[padded_rgba[:, :, 3] > 127] = padded_rgba[:, :, :3][padded_rgba[:, :, 3] > 127]
    return black


# ─────────────────────────── ComfyUI 节点 ────────────────────────

class UVPatchExtractNode:
    """
    根据一张 UV 网格图从完整贴图中提取对应部件的局部贴图。
    内部先做透明背景 padding（颜色准确），再转为纯黑背景输出。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "uv_mesh":      ("IMAGE", {}),
                "uv_texture":   ("IMAGE", {}),
                "padding_px":   ("INT", {"default": 15,  "min": 0, "max": 500,    "step": 1}),
                "close_kernel": ("INT", {"default": 3,   "min": 0, "max": 20,     "step": 1}),
                "min_area":     ("INT", {"default": 100, "min": 1, "max": 100000, "step": 1}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "MASK")
    RETURN_NAMES  = ("patch",  "mask")
    FUNCTION      = "process"
    CATEGORY      = "UV Tools"

    def process(self, uv_mesh, uv_texture, padding_px, close_kernel, min_area):
        if not _SCIPY_OK:
            raise RuntimeError("需要 scipy：pip install scipy")

        # ── ComfyUI tensor → numpy uint8 ─────────────────────────────────
        # (B, H, W, 3) float32 [0,1] → 取第一帧
        mesh_np = (uv_mesh[0].cpu().numpy()    * 255).astype(np.uint8)
        tex_np  = (uv_texture[0].cpu().numpy() * 255).astype(np.uint8)
        tex_h, tex_w = tex_np.shape[:2]

        # ── 提取线条掩码 ──────────────────────────────────────────────────
        line_mask = _extract_line_mask(mesh_np)

        if line_mask.shape != (tex_h, tex_w):
            line_mask = cv2.resize(line_mask, (tex_w, tex_h), interpolation=cv2.INTER_NEAREST)

        if close_kernel > 0:
            k = np.ones((close_kernel, close_kernel), np.uint8)
            line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, k)

        # ── 洪水填充 → 实心掩码（整张网格图 = 一个部件）────────────────────
        solid_mask = _flood_fill_solid(line_mask)

        # 过滤过小噪点（保留最大连通域或面积 >= min_area 的区域）
        n_labels, labels = cv2.connectedComponents(solid_mask, connectivity=8)
        clean_mask = np.zeros_like(solid_mask)
        for lbl in range(1, n_labels):
            if int(np.sum(labels == lbl)) >= min_area:
                clean_mask[labels == lbl] = 255
        solid_mask = clean_mask

        # ── 应用掩码到贴图 ────────────────────────────────────────────────
        rgba = np.zeros((tex_h, tex_w, 4), dtype=np.uint8)
        rgba[:, :, :3] = tex_np
        rgba[:, :, 3]  = solid_mask

        # 透明背景 padding → 黑色背景 RGB
        padded_rgba = _pad_transparent(rgba, padding_px)
        black_patch = _rgba_to_black_bg(padded_rgba)

        # ── → ComfyUI tensor ─────────────────────────────────────────────
        patch_out = torch.from_numpy(black_patch.astype(np.float32) / 255.0).unsqueeze(0)
        mask_out  = torch.from_numpy(solid_mask.astype(np.float32)  / 255.0).unsqueeze(0)

        return (patch_out, mask_out)


# ─────────────────────── ComfyUI 注册 ────────────────────────────

NODE_CLASS_MAPPINGS = {
    "UVPatchExtract": UVPatchExtractNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UVPatchExtract": "UV Patch Extract",
}
