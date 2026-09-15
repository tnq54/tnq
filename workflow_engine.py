import io
import json
import time
import logging
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

class ImageWorkflowEngine:
    """
    n8n-style Node Graph image processing workflow engine.
    Supports modular Nodes (Trigger, Processor, Output), node bypass toggles,
    execution graph sequence, and per-node performance telemetry.
    """

    NODE_TYPES = {
        # Triggers
        "trigger_file": {"name": "File Upload Trigger", "category": "Trigger", "icon": "📁"},
        "trigger_telegram": {"name": "Telegram Photo Trigger", "category": "Trigger", "icon": "✈️"},
        "trigger_ai_gen": {"name": "AI Generation Trigger", "category": "Trigger", "icon": "✨"},
        # Processors
        "resize": {"name": "Resize Node", "category": "Processor", "icon": "📐"},
        "crop": {"name": "Crop Node", "category": "Processor", "icon": "✂️"},
        "adjust_color": {"name": "Color Adjustment Node", "category": "Processor", "icon": "🎨"},
        "rotate_flip": {"name": "Rotate & Flip Node", "category": "Processor", "icon": "🔄"},
        "filter": {"name": "Artistic Filter Node", "category": "Processor", "icon": "🎭"},
        "gemini_analysis": {"name": "Gemini AI Analyzer Node", "category": "Processor", "icon": "🧠"},
        # Outputs
        "watermark": {"name": "Watermark Node", "category": "Output", "icon": "🏷️"},
        "output_download": {"name": "Download Output Node", "category": "Output", "icon": "💾"}
    }

    PRESETS = {
        "n8n Social Media Flow": [
            {"id": "node_1", "type": "trigger_file", "name": "Image Input", "enabled": True, "params": {}},
            {"id": "node_2", "type": "resize", "name": "Format Square (1080x1080)", "enabled": True, "params": {"width": 1080, "height": 1080, "maintain_aspect_ratio": True}},
            {"id": "node_3", "type": "adjust_color", "name": "Vibrant Color Enhancement", "enabled": True, "params": {"brightness": 1.05, "contrast": 1.1, "saturation": 1.2, "sharpness": 1.1}},
            {"id": "node_4", "type": "watermark", "name": "Brand Watermark Overlay", "enabled": True, "params": {"text": "n8n Workflow Space", "position": "bottom-right", "color": "#FFFFFF", "opacity": 0.8}},
            {"id": "node_5", "type": "output_download", "name": "Export Final Asset", "enabled": True, "params": {}}
        ],
        "n8n Vintage Film Flow": [
            {"id": "node_1", "type": "trigger_file", "name": "Image Input", "enabled": True, "params": {}},
            {"id": "node_2", "type": "adjust_color", "name": "Warm Film Tone", "enabled": True, "params": {"brightness": 0.95, "contrast": 1.15, "saturation": 0.7, "sharpness": 0.9}},
            {"id": "node_3", "type": "filter", "name": "Sepia Filter Node", "enabled": True, "params": {"filter_type": "sepia"}},
            {"id": "node_4", "type": "filter", "name": "Vignette Shading Node", "enabled": True, "params": {"filter_type": "vignette"}},
            {"id": "node_5", "type": "output_download", "name": "Export Vintage Asset", "enabled": True, "params": {}}
        ],
        "n8n Monochrome Art Flow": [
            {"id": "node_1", "type": "trigger_file", "name": "Image Input", "enabled": True, "params": {}},
            {"id": "node_2", "type": "filter", "name": "Grayscale Filter Node", "enabled": True, "params": {"filter_type": "grayscale"}},
            {"id": "node_3", "type": "adjust_color", "name": "High Contrast Boost", "enabled": True, "params": {"brightness": 1.0, "contrast": 1.4, "saturation": 1.0, "sharpness": 1.3}},
            {"id": "node_4", "type": "watermark", "name": "Signature Stamp", "enabled": True, "params": {"text": "Monochrome Studio", "position": "bottom-left", "color": "#000000", "opacity": 0.9}}
        ],
        "n8n Edge Sketch Flow": [
            {"id": "node_1", "type": "trigger_file", "name": "Image Input", "enabled": True, "params": {}},
            {"id": "node_2", "type": "filter", "name": "Contour Edge Detect", "enabled": True, "params": {"filter_type": "contour"}},
            {"id": "node_3", "type": "adjust_color", "name": "Edge Sharpening", "enabled": True, "params": {"brightness": 1.1, "contrast": 1.3, "saturation": 1.0, "sharpness": 1.5}}
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
            sepia_img = Image.new("RGB", gray.size)

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
    def execute_node(cls, img: Image.Image, node: dict) -> tuple[Image.Image, dict]:
        """
        Executes a single n8n graph node with performance timing and telemetry recording.
        """
        node_id = node.get("id", "unknown_node")
        node_type = node.get("type", "unknown")
        node_name = node.get("name", node_type)
        is_enabled = node.get("enabled", True)
        params = node.get("params", {})

        meta = cls.NODE_TYPES.get(node_type, {"name": node_name, "category": "Custom", "icon": "⚙️"})

        if not is_enabled:
            return img, {
                "id": node_id,
                "name": node_name,
                "type": node_type,
                "status": "bypassed",
                "execution_time_ms": 0.0,
                "output_dimensions": f"{img.width}x{img.height}",
                "message": f"Node '{node_name}' disabled/bypassed."
            }

        start_time = time.perf_counter()
        processed_img = img

        try:
            if node_type == "resize":
                processed_img = cls.process_resize(img, **params)
            elif node_type == "crop":
                processed_img = cls.process_crop(img, **params)
            elif node_type == "adjust_color":
                processed_img = cls.process_adjust_color(img, **params)
            elif node_type == "rotate_flip":
                processed_img = cls.process_rotate_flip(img, **params)
            elif node_type == "filter":
                processed_img = cls.process_filter(img, **params)
            elif node_type == "watermark":
                processed_img = cls.process_watermark(img, **params)
            elif node_type in ("trigger_file", "trigger_telegram", "trigger_ai_gen", "output_download", "gemini_analysis"):
                # Pass-through / trigger / output nodes
                processed_img = img

            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            telemetry = {
                "id": node_id,
                "name": node_name,
                "type": node_type,
                "category": meta.get("category", "Processor"),
                "icon": meta.get("icon", "⚡"),
                "status": "success",
                "execution_time_ms": elapsed_ms,
                "output_dimensions": f"{processed_img.width}x{processed_img.height}",
                "message": f"Executed node '{node_name}' successfully ({elapsed_ms}ms)."
            }
            return processed_img, telemetry

        except Exception as e:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(f"Error executing node '{node_name}' ({node_type}): {e}")
            telemetry = {
                "id": node_id,
                "name": node_name,
                "type": node_type,
                "status": "error",
                "execution_time_ms": elapsed_ms,
                "output_dimensions": f"{img.width}x{img.height}",
                "message": f"Error in node '{node_name}': {e}"
            }
            return img, telemetry

    @classmethod
    def run_pipeline(cls, img: Image.Image, nodes: list[dict]) -> tuple[Image.Image, list[dict]]:
        """
        Executes an n8n node graph sequence across input image.
        Returns final output image and per-node execution telemetry metrics.
        """
        current_img = img.copy().convert("RGB")
        telemetry_list = []

        for idx, node in enumerate(nodes, start=1):
            current_img, node_telemetry = cls.execute_node(current_img, node)
            node_telemetry["step_number"] = idx
            telemetry_list.append(node_telemetry)

        return current_img, telemetry_list

    @staticmethod
    def export_workflow_json(nodes: list[dict]) -> str:
        return json.dumps(nodes, indent=2)

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
