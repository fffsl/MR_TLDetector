# -*- coding: utf-8 -*-

from pathlib import Path
import sys
from contextlib import contextmanager
from runpy import run_path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from MR_TLDetector.paths import application_path


ONERESTORE_ROOT = application_path() / "external" / "OneRestore"
ONERESTORE_CKPT_DIR = ONERESTORE_ROOT / "ckpts"
RESTORMER_ROOT = application_path() / "external" / "Restormer"
RESTORMER_DERAIN_CKPT = RESTORMER_ROOT / "Deraining" / "pretrained_models" / "deraining.pth"
SNOWFORMER_ROOT = application_path() / "external" / "SnowFormer"
SNOWFORMER_CKPT = SNOWFORMER_ROOT / "pretrained_models" / "SnowFormer_CSD.pth"


@contextmanager
def _temporary_sys_path(path):
    text_path = str(path)
    inserted = text_path not in sys.path
    if inserted:
        sys.path.insert(0, text_path)
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(text_path)
            except ValueError:
                pass


class OneRestoreRunner:
    _instance = None

    restore_weights = {
        "auto": "onerestore_real.tar",
        "rain": "onerestore_rain1200.tar",
        "snow": "onerestore_snow100k.tar",
        "low": "onerestore_lol.tar",
        "composite": "onerestore_cdd-11.tar",
    }

    prompt_by_task = {
        "rain": "rain",
        "snow": "snow",
        "low": "low",
    }

    def __init__(self):
        missing = [
            name for name in ["embedder_model.tar", *self.restore_weights.values()]
            if not (ONERESTORE_CKPT_DIR / name).exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing OneRestore checkpoint(s): " + ", ".join(missing)
            )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.embedder = None
        self.restorers = {}
        self.max_side = 960

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def restore(self, bgr_image, task):
        if bgr_image is None:
            return bgr_image
        if task not in self.restore_weights:
            return bgr_image.copy()

        self._ensure_models(task)
        source_shape = bgr_image.shape[:2]
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb = self._resize_for_model(rgb)

        lq_tensor = self._image_to_tensor(rgb)
        embed_tensor = F.interpolate(lq_tensor, size=(224, 224), mode="bilinear", align_corners=False)

        with torch.no_grad():
            if task == "auto" or task == "composite":
                text_embedding, _, _ = self.embedder(embed_tensor, "image_encoder")
            else:
                prompt = self.prompt_by_task[task]
                text_embedding, _, _ = self.embedder([prompt], "text_encoder")
            restored = self.restorers[task](lq_tensor, text_embedding)

        out_rgb = self._tensor_to_image(restored)
        if out_rgb.shape[:2] != source_shape:
            out_rgb = cv2.resize(out_rgb, (source_shape[1], source_shape[0]), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

    def restore_derain(self, bgr_image):
        restored = self.restore(bgr_image, "rain")
        restored = self._match_luminance(bgr_image, restored, min_gain=0.75, max_gain=1.15)
        return self._blend_preserving_edges(bgr_image, restored, strength=0.70)

    def restore_low_light(self, bgr_image):
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        mean_light = float(gray.mean())
        if mean_light >= 105.0:
            return bgr_image.copy()

        restored = self.restore(bgr_image, "low")
        enhanced = self._illumination_enhance(restored, target_light=118.0)
        enhanced = self._limit_highlights(enhanced, max_highlight_ratio=0.025)
        strength = float(np.clip((110.0 - mean_light) / 80.0, 0.35, 0.85))
        return cv2.addWeighted(bgr_image, 1.0 - strength, enhanced, strength, 0)

    def restore_auto_conservative(self, bgr_image):
        restored = self.restore(bgr_image, "auto")
        restored = self._match_luminance(bgr_image, restored, min_gain=0.80, max_gain=1.25)
        restored = self._limit_color_shift(bgr_image, restored, max_delta=28.0)
        restored = self._limit_highlights(restored, max_highlight_ratio=0.035)
        return self._blend_preserving_edges(bgr_image, restored, strength=0.58)

    def _ensure_models(self, task):
        with _temporary_sys_path(ONERESTORE_ROOT):
            from model.Embedder import Embedder
            from model.OneRestore import OneRestore

        if self.embedder is None:
            embedder_path = ONERESTORE_CKPT_DIR / "embedder_model.tar"
            model_info = self._load_state_dict(embedder_path)
            with _temporary_sys_path(ONERESTORE_ROOT):
                from model.Embedder import Embedder
            self.embedder = Embedder([
                "clear", "low", "haze", "rain", "snow",
                "low_haze", "low_rain", "low_snow", "haze_rain",
                "haze_snow", "low_haze_rain", "low_haze_snow",
            ])
            self.embedder.load_state_dict(model_info)
            self.embedder.to(self.device).eval()
            for param in self.embedder.parameters():
                param.requires_grad = False

        if task not in self.restorers:
            restore_path = ONERESTORE_CKPT_DIR / self.restore_weights[task]
            model_info = self._load_state_dict(restore_path)
            with _temporary_sys_path(ONERESTORE_ROOT):
                from model.OneRestore import OneRestore
            restorer = OneRestore()
            restorer.load_state_dict(model_info)
            restorer.to(self.device).eval()
            for param in restorer.parameters():
                param.requires_grad = False
            self.restorers[task] = restorer

    def _load_state_dict(self, path):
        if self.device.type == "cuda":
            return torch.load(str(path))
        return torch.load(str(path), map_location=torch.device("cpu"))

    def _resize_for_model(self, rgb_image):
        height, width = rgb_image.shape[:2]
        scale = min(1.0, self.max_side / max(height, width))
        if scale < 1.0:
            width = max(32, int(round(width * scale)))
            height = max(32, int(round(height * scale)))
            rgb_image = cv2.resize(rgb_image, (width, height), interpolation=cv2.INTER_AREA)

        pad_h = (8 - rgb_image.shape[0] % 8) % 8
        pad_w = (8 - rgb_image.shape[1] % 8) % 8
        if pad_h or pad_w:
            rgb_image = cv2.copyMakeBorder(
                rgb_image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
            )
        return rgb_image

    def _image_to_tensor(self, rgb_image):
        array = rgb_image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device)

    @staticmethod
    def _tensor_to_image(tensor):
        tensor = tensor.detach().clamp(0.0, 1.0).squeeze(0).cpu()
        array = tensor.numpy().transpose(1, 2, 0)
        return np.clip(array * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def _match_luminance(reference_bgr, output_bgr, min_gain=0.8, max_gain=1.25):
        ref_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        out_gray = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gain = float(ref_gray.mean() / max(out_gray.mean(), 1.0))
        gain = float(np.clip(gain, min_gain, max_gain))
        return np.clip(output_bgr.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    @staticmethod
    def _blend_preserving_edges(reference_bgr, output_bgr, strength=0.6):
        ref_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
        edge = cv2.Laplacian(ref_gray, cv2.CV_32F)
        edge = np.abs(edge)
        if float(edge.max()) > 0:
            edge = edge / float(edge.max())
        keep = cv2.GaussianBlur(edge, (0, 0), 1.6)
        alpha = np.clip(strength * (1.0 - 0.45 * keep), 0.25, strength)[:, :, None]
        blended = reference_bgr.astype(np.float32) * (1.0 - alpha) + output_bgr.astype(np.float32) * alpha
        return np.clip(blended, 0, 255).astype(np.uint8)

    @staticmethod
    def _illumination_enhance(bgr_image, target_light=118.0):
        source = bgr_image.astype(np.float32) / 255.0
        illumination = np.max(source, axis=2)
        illumination = cv2.bilateralFilter(illumination, 9, 0.12, 15)
        illumination = np.maximum(illumination, 0.10)
        gray_mean = float(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY).mean())
        gamma_strength = float(np.clip((target_light - gray_mean) / target_light, 0.15, 0.85))
        gain = 1.0 / np.power(illumination, 0.55 * gamma_strength)
        enhanced = source * gain[:, :, None]
        return np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def _limit_highlights(bgr_image, max_highlight_ratio=0.03):
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        highlight_ratio = float(np.mean(gray >= 245))
        if highlight_ratio <= max_highlight_ratio:
            return bgr_image
        scale = float(np.clip(1.0 - (highlight_ratio - max_highlight_ratio) * 2.2, 0.78, 1.0))
        return np.clip(bgr_image.astype(np.float32) * scale, 0, 255).astype(np.uint8)

    @staticmethod
    def _limit_color_shift(reference_bgr, output_bgr, max_delta=28.0):
        ref_lab = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        out_lab = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        delta_a = float(out_lab[:, :, 1].mean() - ref_lab[:, :, 1].mean())
        delta_b = float(out_lab[:, :, 2].mean() - ref_lab[:, :, 2].mean())
        shift = max(abs(delta_a), abs(delta_b))
        if shift <= max_delta:
            return output_bgr
        out_lab[:, :, 1] -= delta_a * 0.55
        out_lab[:, :, 2] -= delta_b * 0.55
        return cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


class RestormerDerainRunner:
    _instance = None

    def __init__(self):
        if not RESTORMER_DERAIN_CKPT.exists():
            raise FileNotFoundError(f"Missing Restormer deraining checkpoint: {RESTORMER_DERAIN_CKPT}")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.max_side = 960

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def is_model_available():
        return RESTORMER_DERAIN_CKPT.exists()

    def restore(self, bgr_image):
        if bgr_image is None:
            return bgr_image
        self._ensure_model()

        source_shape = bgr_image.shape[:2]
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb = self._resize_for_model(rgb)
        tensor = self._image_to_tensor(rgb)
        tensor, height, width = self._pad_to_multiple(tensor, multiple=8)

        with torch.no_grad():
            restored = self.model(tensor)

        restored = restored[:, :, :height, :width]
        out_rgb = self._tensor_to_image(restored)
        if out_rgb.shape[:2] != source_shape:
            out_rgb = cv2.resize(out_rgb, (source_shape[1], source_shape[0]), interpolation=cv2.INTER_LINEAR)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        out_bgr = OneRestoreRunner._match_luminance(bgr_image, out_bgr, min_gain=0.82, max_gain=1.12)
        out_bgr = WeatherArtifactCleaner.remove_rain_streaks(out_bgr)
        out_bgr = WeatherArtifactCleaner.mild_brighten(out_bgr, target_mean=128.0, max_gain=1.12)
        return OneRestoreRunner._blend_preserving_edges(bgr_image, out_bgr, strength=0.78)

    def _ensure_model(self):
        if self.model is not None:
            return

        arch_path = RESTORMER_ROOT / "basicsr" / "models" / "archs" / "restormer_arch.py"
        load_arch = run_path(str(arch_path))
        model = load_arch["Restormer"](
            inp_channels=3,
            out_channels=3,
            dim=48,
            num_blocks=[4, 6, 6, 8],
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type="WithBias",
            dual_pixel_task=False,
        )
        checkpoint = torch.load(
            str(RESTORMER_DERAIN_CKPT),
            map_location=self.device if self.device.type == "cuda" else torch.device("cpu"),
        )
        model.load_state_dict(checkpoint["params"])
        model.to(self.device).eval()
        for param in model.parameters():
            param.requires_grad = False
        self.model = model

    def _resize_for_model(self, rgb_image):
        height, width = rgb_image.shape[:2]
        scale = min(1.0, self.max_side / max(height, width))
        if scale < 1.0:
            width = max(32, int(round(width * scale)))
            height = max(32, int(round(height * scale)))
            rgb_image = cv2.resize(rgb_image, (width, height), interpolation=cv2.INTER_AREA)
        return rgb_image

    def _image_to_tensor(self, rgb_image):
        array = rgb_image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device)

    @staticmethod
    def _pad_to_multiple(tensor, multiple=8):
        height, width = tensor.shape[2], tensor.shape[3]
        target_h = ((height + multiple) // multiple) * multiple
        target_w = ((width + multiple) // multiple) * multiple
        pad_h = target_h - height if height % multiple != 0 else 0
        pad_w = target_w - width if width % multiple != 0 else 0
        if pad_h or pad_w:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), "reflect")
        return tensor, height, width

    @staticmethod
    def _tensor_to_image(tensor):
        tensor = tensor.detach().clamp(0.0, 1.0).squeeze(0).cpu()
        array = tensor.numpy().transpose(1, 2, 0)
        return np.clip(array * 255.0, 0, 255).astype(np.uint8)


class SnowFormerDesnowRunner:
    _instance = None

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.max_side = 960
        self.tile_size = 256
        self.tile_overlap = 32

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def is_model_available():
        return SNOWFORMER_CKPT.exists()

    def restore(self, bgr_image):
        if bgr_image is None:
            return bgr_image
        if not SNOWFORMER_CKPT.exists():
            return WeatherArtifactCleaner.fast_desnow(bgr_image)

        self._ensure_model()
        source_shape = bgr_image.shape[:2]
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb = self._resize_for_model(rgb)
        tensor = self._image_to_tensor(rgb)
        tensor, height, width = RestormerDerainRunner._pad_to_multiple(tensor, multiple=8)

        with torch.no_grad():
            restored = self._tile_restore(tensor)

        restored = restored[:, :, :height, :width]
        out_rgb = RestormerDerainRunner._tensor_to_image(restored)
        if out_rgb.shape[:2] != source_shape:
            out_rgb = cv2.resize(out_rgb, (source_shape[1], source_shape[0]), interpolation=cv2.INTER_LINEAR)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        out_bgr = OneRestoreRunner._match_luminance(bgr_image, out_bgr, min_gain=0.82, max_gain=1.18)
        out_bgr = WeatherArtifactCleaner.remove_snow_spots(out_bgr)
        out_bgr = WeatherArtifactCleaner.mild_brighten(out_bgr, target_mean=126.0, max_gain=1.12)
        return OneRestoreRunner._blend_preserving_edges(bgr_image, out_bgr, strength=0.72)

    def _ensure_model(self):
        if self.model is not None:
            return
        with _temporary_sys_path(SNOWFORMER_ROOT):
            from SnowFormer import Transformer

        model = Transformer()
        checkpoint = torch.load(
            str(SNOWFORMER_CKPT),
            map_location=self.device if self.device.type == "cuda" else torch.device("cpu"),
        )
        model.load_state_dict(checkpoint)
        model.to(self.device).eval()
        for param in model.parameters():
            param.requires_grad = False
        self.model = model

    def _tile_restore(self, tensor):
        batch, channels, height, width = tensor.shape
        tile = min(self.tile_size, height, width)
        if tile % 8 != 0:
            tile = max(8, tile - tile % 8)
        stride = max(8, tile - self.tile_overlap)
        h_idx_list = list(range(0, max(height - tile, 0), stride)) + [height - tile]
        w_idx_list = list(range(0, max(width - tile, 0), stride)) + [width - tile]
        output = torch.zeros(batch, channels, height, width, device=tensor.device, dtype=tensor.dtype)
        weight = torch.zeros_like(output)
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                patch = tensor[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                restored_patch = self.model(patch)
                output[..., h_idx:h_idx + tile, w_idx:w_idx + tile].add_(restored_patch)
                weight[..., h_idx:h_idx + tile, w_idx:w_idx + tile].add_(torch.ones_like(restored_patch))
        return torch.clamp(output / torch.clamp(weight, min=1.0), 0.0, 1.0)

    def _resize_for_model(self, rgb_image):
        height, width = rgb_image.shape[:2]
        scale = min(1.0, self.max_side / max(height, width))
        if scale < 1.0:
            width = max(32, int(round(width * scale)))
            height = max(32, int(round(height * scale)))
            rgb_image = cv2.resize(rgb_image, (width, height), interpolation=cv2.INTER_AREA)
        return rgb_image

    def _image_to_tensor(self, rgb_image):
        array = rgb_image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device)


class WeatherArtifactCleaner:
    @staticmethod
    def mild_brighten(bgr_image, target_mean=126.0, max_gain=1.12):
        gray_mean = float(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY).mean())
        if gray_mean >= target_mean:
            return bgr_image
        gain = float(np.clip(target_mean / max(gray_mean, 1.0), 1.0, max_gain))
        return np.clip(bgr_image.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    @staticmethod
    def remove_snow_spots(bgr_image):
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        low_sat = cv2.inRange(hsv, np.array([0, 0, 120], np.uint8), np.array([180, 90, 255], np.uint8))
        local_bright = cv2.subtract(gray, cv2.GaussianBlur(gray, (0, 0), 5))
        local_mask = cv2.bitwise_and(cv2.inRange(local_bright, 16, 255), low_sat)
        blob_contrast = cv2.subtract(gray, cv2.GaussianBlur(gray, (0, 0), 13))
        blob_mask = cv2.bitwise_and(cv2.inRange(blob_contrast, 10, 255), low_sat)
        candidates = cv2.bitwise_or(local_mask, blob_mask)
        candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates)
        mask = np.zeros_like(candidates)
        image_area = bgr_image.shape[0] * bgr_image.shape[1]
        max_area = max(160, int(image_area * 0.012))
        saturation = hsv[:, :, 1]
        edges = cv2.Canny(gray, 45, 110)
        edge_density = cv2.blur((edges > 0).astype(np.float32), (25, 25))
        for index in range(1, count):
            _, _, width, height, area = stats[index]
            aspect = max(width, height) / max(1, min(width, height))
            left, top = int(stats[index, cv2.CC_STAT_LEFT]), int(stats[index, cv2.CC_STAT_TOP])
            component = labels[top:top + height, left:left + width] == index
            mean_saturation = float(saturation[top:top + height, left:left + width][component].mean())
            mean_edge_density = float(edge_density[top:top + height, left:left + width][component].mean())
            is_small_snow = 3 <= area <= 260 and width <= 42 and height <= 42 and aspect <= 4.5
            is_blob_snow = 80 <= area <= max_area and width <= 95 and height <= 75 and aspect <= 3.0
            if mean_saturation <= 76.0 and mean_edge_density <= 0.085 and (is_small_snow or is_blob_snow):
                mask[top:top + height, left:left + width][component] = 255
        # Blurred flakes have a wide, low contrast halo that the sharp mask misses.
        halo = cv2.bitwise_and(cv2.inRange(blob_contrast, 5, 255), low_sat)
        halo = cv2.morphologyEx(halo, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        halo_count, halo_labels, halo_stats, _ = cv2.connectedComponentsWithStats(halo)
        for index in range(1, halo_count):
            left, top, width, height, area = halo_stats[index]
            if not (55 <= area <= max_area and width <= 95 and height <= 75):
                continue
            if max(width, height) / max(1, min(width, height)) > 2.8:
                continue
            component = halo_labels[top:top + height, left:left + width] == index
            density = float(edge_density[top:top + height, left:left + width][component].mean())
            if density <= 0.045:
                if area >= 130 and width >= 12 and height >= 12:
                    center = (int(left + width / 2), int(top + height / 2))
                    axes = (int(width / 2 + 3), int(height / 2 + 3))
                    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
                else:
                    mask[top:top + height, left:left + width][component] = 255
        if not np.any(mask):
            return bgr_image

        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        repaired = cv2.inpaint(bgr_image, mask, 4, cv2.INPAINT_TELEA)
        soft_mask = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 2.0)[:, :, None]
        blended = bgr_image.astype(np.float32) * (1.0 - 0.95 * soft_mask) + repaired.astype(np.float32) * (0.95 * soft_mask)
        return np.clip(blended, 0, 255).astype(np.uint8)

    @staticmethod
    def fast_desnow(bgr_image):
        cleaned = WeatherArtifactCleaner.remove_snow_spots(bgr_image)
        cleaned = WeatherArtifactCleaner.mild_brighten(cleaned, target_mean=128.0, max_gain=1.14)
        return OneRestoreRunner._blend_preserving_edges(bgr_image, cleaned, strength=0.86)

    @staticmethod
    def remove_rain_streaks(bgr_image):
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        vertical = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)

        smooth = cv2.GaussianBlur(gray, (0, 0), 3)
        bright_detail = cv2.subtract(vertical, smooth)
        _, bright_mask = cv2.threshold(bright_detail, 7, 255, cv2.THRESH_BINARY)

        horizontal_background = cv2.medianBlur(gray, 15)
        contrast_detail = cv2.absdiff(gray, horizontal_background)
        contrast_detail = cv2.morphologyEx(contrast_detail, cv2.MORPH_OPEN, np.ones((1, 5), np.uint8))
        _, contrast_mask = cv2.threshold(contrast_detail, 16, 255, cv2.THRESH_BINARY)

        mask = cv2.bitwise_or(bright_mask, contrast_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1, 13), np.uint8))

        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        selected = np.zeros_like(mask)
        image_area = bgr_image.shape[0] * bgr_image.shape[1]
        max_area = max(160, int(image_area * 0.025))
        for index in range(1, count):
            _, _, width, height, area = stats[index]
            if 10 <= area <= max_area and height >= max(18, width * 2) and width <= 14:
                selected[labels == index] = 255
        if not np.any(selected):
            return bgr_image

        selected = cv2.dilate(selected, np.ones((3, 3), np.uint8), iterations=1)
        repaired = cv2.inpaint(bgr_image, selected, 2, cv2.INPAINT_TELEA)
        return cv2.addWeighted(bgr_image, 0.22, repaired, 0.78, 0)


def _print_runtime_status():
    checkpoints = {
        "OneRestore embedder": ONERESTORE_CKPT_DIR / "embedder_model.tar",
        "OneRestore real": ONERESTORE_CKPT_DIR / "onerestore_real.tar",
        "OneRestore rain": ONERESTORE_CKPT_DIR / "onerestore_rain1200.tar",
        "OneRestore snow": ONERESTORE_CKPT_DIR / "onerestore_snow100k.tar",
        "OneRestore low-light": ONERESTORE_CKPT_DIR / "onerestore_lol.tar",
        "Restormer derain": RESTORMER_DERAIN_CKPT,
        "SnowFormer desnow": SNOWFORMER_CKPT,
    }
    print("MR_TLDetector restoration runtime")
    print(f"Project root: {application_path()}")
    print(f"PyTorch device: {'cuda:0' if torch.cuda.is_available() else 'cpu'}")
    for name, path in checkpoints.items():
        state = "OK" if path.exists() else "missing"
        print(f"{name}: {state} ({path})")


if __name__ == "__main__":
    _print_runtime_status()
