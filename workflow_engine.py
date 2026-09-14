import io
import json
import logging
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

class ImageWorkflowEngine:
    """
    Core image processing workflow engine for applying sequential transformations,
    filters, watermarks, and AI analysis steps to images.
    """

    PRESETS = {
        "Social Media Post": [
            {"type": "resize", "params": {"width": 1080, "height": 1080, "maintain_aspect_ratio": True}},
            {"type": "adjust_color", "params": {"brightness": 1.05, "contrast": 1.1, "saturation": 1.2, "sharpness": 1.1}},
            {"type": "watermark", "params": {"text": "Workflow Server", "position": "bottom-right", "color": "#FFFFFF", "opacity": 0.8}}
        ],
        "Aesthetic Vintage": [
            {"type": "adjust_color", "params": {"brightness": 0.95, "contrast": 1.15, "saturation": 0.7, "sharpness": 0.9}},
            {"type": "filter", "params": {"filter_type": "sepia"}},
            {"type": "filter", "params": {"filter_type": "vignette"}}
        ],
        "High Contrast Monochrome": [
            {"type": "filter", "params": {"filter_type": "grayscale"}},
            {"type": "adjust_color", "params": {"brightness": 1.0, "contrast": 1.4, "saturation": 1.0, "sharpness": 1.3}}
        ],
        "Soft Portrait Blur": [
            {"type": "adjust_color", "params": {"brightness": 1.05, "contrast": 1.0, "saturation": 1.05, "sharpness": 0.8}},
            {"type": "filter", "params": {"filter_type": "blur", "radius": 2}}
        ],
        "Edge Art Sketch": [
            {"type": "filter", "params": {"filter_type": "contour"}},
            {"type": "adjust_color", "params": {"brightness": 1.1, "contrast": 1.3, "saturation": 1.0, "sharpness": 1.5}}
        ]
    }

    @staticmethod
    def process_resize(img: Image.Image, width: int = 800, height: int = 800, maintain_aspect_ratio: bool = True) -> Image.Image:
        if width <= 0 or height <= 0:
            return img
        if maintain_aspect_ratio:
            img_copy = img.copy()
            img_copy.thumbnail((width, height), Image.Resampling.LANCZOS)
            return img_copy
        else:
            return img.resize((width, height), Image.Resampling.LANCZOS)

    @staticmethod
    def process_crop(img: Image.Image, left_pct: float = 0, top_pct: float = 0, right_pct: float = 100, bottom_pct: float = 100) -> Image.Image:
        w, h = img.size
        left = int((max(0.0, min(100.0, left_pct)) / 100.0) * w)
        top = int((max(0.0, min(100.0, top_pct)) / 100.0) * h)
        right = int((max(0.0, min(100.0, right_pct)) / 100.0) * w)
        bottom = int((max(0.0, min(100.0, bottom_pct)) / 100.0) * h)

        if right <= left or bottom <= top:
            return img

        return img.crop((left, top, right, bottom))

    @staticmethod
    def process_adjust_color(img: Image.Image, brightness: float = 1.0, contrast: float = 1.0, saturation: float = 1.0, sharpness: float = 1.0) -> Image.Image:
        res = img.convert("RGB")
        if brightness != 1.0:
            res = ImageEnhance.Brightness(res).enhance(brightness)
        if contrast != 1.0:
            res = ImageEnhance.Contrast(res).enhance(contrast)
        if saturation != 1.0:
            res = ImageEnhance.Color(res).enhance(saturation)
        if sharpness != 1.0:
            res = ImageEnhance.Sharpness(res).enhance(sharpness)
        return res

    @staticmethod
    def process_rotate_flip(img: Image.Image, angle: int = 0, flip_h: bool = False, flip_v: bool = False) -> Image.Image:
        res = img
        if angle in (90, 180, 270):
            res = res.rotate(-angle, expand=True)
        if flip_h:
            res = ImageOps.mirror(res)
        if flip_v:
            res = ImageOps.flip(res)
        return res

    @staticmethod
    def process_filter(img: Image.Image, filter_type: str = "grayscale", radius: float = 2.0) -> Image.Image:
        filter_type = filter_type.lower()
        res = img.convert("RGB")

        if filter_type == "grayscale":
            return ImageOps.grayscale(res).convert("RGB")

        elif filter_type == "sepia":
            gray = ImageOps.grayscale(res)
            # Apply sepia tone transformation
            sepia_img = Image.new("RGB", gray.size)

            # Use get_flattened_data if available in Pillow, or fallback to getdata
            if hasattr(gray, "get_flattened_data"):
                flat_data = gray.get_flattened_data()
            else:
                flat_data = gray.getdata()

            pixels = [
                (int(p * 240 / 255), int(p * 200 / 255), int(p * 145 / 255))
                for p in flat_data
            ]
            sepia_img.putdata(pixels)
            return sepia_img

        elif filter_type == "blur":
            return res.filter(ImageFilter.GaussianBlur(radius=radius))

        elif filter_type == "contour":
            return res.filter(ImageFilter.CONTOUR)

        elif filter_type == "edge_enhance":
            return res.filter(ImageFilter.EDGE_ENHANCE_MORE)

        elif filter_type == "invert":
            return ImageOps.invert(res)

        elif filter_type == "posterize":
            return ImageOps.posterize(res, 3)

        elif filter_type == "vignette":
            w, h = res.size
            vignette_mask = Image.new("L", (w, h), 255)
            draw = ImageDraw.Draw(vignette_mask)
            cx, cy = w / 2, h / 2
            max_dist = ((cx ** 2) + (cy ** 2)) ** 0.5
            for y in range(0, h, max(1, h // 100)):
                for x in range(0, w, max(1, w // 100)):
                    dist = (((x - cx) ** 2) + ((y - cy) ** 2)) ** 0.5
                    factor = 1.0 - min(1.0, (dist / max_dist) ** 1.8)
                    draw.rectangle([x, y, x + max(1, w // 100), y + max(1, h // 100)], fill=int(factor * 255))

            black = Image.new("RGB", (w, h), (0, 0, 0))
            return Image.composite(res, black, vignette_mask)

        return res

    @staticmethod
    def process_watermark(img: Image.Image, text: str = "Watermark", position: str = "bottom-right", color: str = "#FFFFFF", opacity: float = 0.8) -> Image.Image:
        if not text:
            return img

        res = img.convert("RGBA")
        overlay = Image.new("RGBA", res.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        font_size = max(16, int(min(res.size) * 0.05))
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        # Parse hex color
        hex_color = color.lstrip("#")
        if len(hex_color) == 6:
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        else:
            r, g, b = (255, 255, 255)
        alpha = int(opacity * 255)

        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            text_w = len(text) * font_size * 0.6
            text_h = font_size

        w, h = res.size
        margin = 20

        if position == "top-left":
            x, y = margin, margin
        elif position == "top-right":
            x, y = w - text_w - margin, margin
        elif position == "bottom-left":
            x, y = margin, h - text_h - margin
        elif position == "center":
            x, y = (w - text_w) / 2, (h - text_h) / 2
        else:  # bottom-right default
            x, y = w - text_w - margin, h - text_h - margin

        draw.text((x, y), text, fill=(r, g, b, alpha), font=font)

        out = Image.alpha_composite(res, overlay)
        return out.convert("RGB")

    @classmethod
    def execute_step(cls, img: Image.Image, step: dict) -> tuple[Image.Image, str]:
        step_type = step.get("type")
        params = step.get("params", {})
        log_msg = f"Step '{step_type}' applied with params {params}"

        if step_type == "resize":
            return cls.process_resize(img, **params), log_msg
        elif step_type == "crop":
            return cls.process_crop(img, **params), log_msg
        elif step_type == "adjust_color":
            return cls.process_adjust_color(img, **params), log_msg
        elif step_type == "rotate_flip":
            return cls.process_rotate_flip(img, **params), log_msg
        elif step_type == "filter":
            return cls.process_filter(img, **params), log_msg
        elif step_type == "watermark":
            return cls.process_watermark(img, **params), log_msg
        else:
            return img, f"Step '{step_type}' skipped (unknown type)"

    @classmethod
    def run_pipeline(cls, img: Image.Image, steps: list[dict]) -> tuple[Image.Image, list[str]]:
        current_img = img.copy().convert("RGB")
        logs = []
        for idx, step in enumerate(steps, start=1):
            try:
                current_img, step_log = cls.execute_step(current_img, step)
                logs.append(f"[{idx}] {step_log}")
            except Exception as e:
                logger.error(f"Error executing step {idx} ({step}): {e}")
                logs.append(f"[{idx}] Error in step '{step.get('type')}': {e}")

        return current_img, logs

    @staticmethod
    def export_workflow_json(steps: list[dict]) -> str:
        return json.dumps(steps, indent=2)

    @staticmethod
    def import_workflow_json(json_str: str) -> list[dict]:
        try:
            data = json.loads(json_str)
            if isinstance(data, list):
                return data
            return []
        except Exception as e:
            logger.error(f"Failed to parse workflow JSON: {e}")
            return []
