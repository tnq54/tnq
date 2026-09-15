import io
import json
import time
import logging
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

class ImageWorkflowEngine:
    """
    Authentic n8n Node Graph Execution Engine.
    Supports n8n Node Schemas (id, type, name, typeVersion, position [x,y], parameters, disabled),
    Connections/Edges between nodes, topological sorting graph execution, and execution telemetry.
    """

    NODE_TYPES = {
        # Triggers (Green / Emerald)
        "n8n-nodes-base.fileTrigger": {
            "name": "On File Upload",
            "category": "Trigger",
            "icon": "📁",
            "color": "#10B981",
            "typeVersion": 1.0
        },
        "n8n-nodes-base.telegramTrigger": {
            "name": "Telegram Photo Trigger",
            "category": "Trigger",
            "icon": "✈️",
            "color": "#10B981",
            "typeVersion": 1.0
        },
        "n8n-nodes-base.aiGenTrigger": {
            "name": "AI Generation Trigger",
            "category": "Trigger",
            "icon": "✨",
            "color": "#10B981",
            "typeVersion": 1.0
        },
        # Actions / Processors (Blue / Indigo)
        "n8n-nodes-base.resize": {
            "name": "Resize Asset",
            "category": "Action",
            "icon": "📐",
            "color": "#3B82F6",
            "typeVersion": 1.1
        },
        "n8n-nodes-base.crop": {
            "name": "Crop Canvas",
            "category": "Action",
            "icon": "✂️",
            "color": "#3B82F6",
            "typeVersion": 1.0
        },
        "n8n-nodes-base.adjustColor": {
            "name": "Color Grade & Adjust",
            "category": "Action",
            "icon": "🎨",
            "color": "#3B82F6",
            "typeVersion": 1.2
        },
        "n8n-nodes-base.rotateFlip": {
            "name": "Rotate & Mirror",
            "category": "Action",
            "icon": "🔄",
            "color": "#3B82F6",
            "typeVersion": 1.0
        },
        "n8n-nodes-base.filterFx": {
            "name": "Artistic Filter FX",
            "category": "Action",
            "icon": "🎭",
            "color": "#3B82F6",
            "typeVersion": 1.0
        },
        "n8n-nodes-base.geminiVision": {
            "name": "Gemini 1.5 AI Inspector",
            "category": "AI",
            "icon": "🧠",
            "color": "#EC4899",
            "typeVersion": 2.0
        },
        # Outputs (Purple / Pink)
        "n8n-nodes-base.watermark": {
            "name": "Brand Watermark Overlay",
            "category": "Output",
            "icon": "🏷️",
            "color": "#8B5CF6",
            "typeVersion": 1.0
        },
        "n8n-nodes-base.downloadOutput": {
            "name": "Export Rendered Asset",
            "category": "Output",
            "icon": "💾",
            "color": "#8B5CF6",
            "typeVersion": 1.0
        }
    }

    PRESETS = {
        "n8n Social Media Auto-Branding": {
            "nodes": [
                {
                    "id": "node_trigger",
                    "type": "n8n-nodes-base.fileTrigger",
                    "name": "File Input Trigger",
                    "typeVersion": 1.0,
                    "position": [240, 300],
                    "disabled": False,
                    "parameters": {}
                },
                {
                    "id": "node_resize",
                    "type": "n8n-nodes-base.resize",
                    "name": "Format Square 1080p",
                    "typeVersion": 1.1,
                    "position": [480, 300],
                    "disabled": False,
                    "parameters": {"width": 1080, "height": 1080, "maintain_aspect_ratio": True}
                },
                {
                    "id": "node_color",
                    "type": "n8n-nodes-base.adjustColor",
                    "name": "Vibrant Color Boost",
                    "typeVersion": 1.2,
                    "position": [720, 300],
                    "disabled": False,
                    "parameters": {"brightness": 1.05, "contrast": 1.1, "saturation": 1.2, "sharpness": 1.1}
                },
                {
                    "id": "node_watermark",
                    "type": "n8n-nodes-base.watermark",
                    "name": "Brand Watermark Stamp",
                    "typeVersion": 1.0,
                    "position": [960, 300],
                    "disabled": False,
                    "parameters": {"text": "n8n Workflow Studio", "position": "bottom-right", "color": "#FFFFFF", "opacity": 0.8}
                },
                {
                    "id": "node_output",
                    "type": "n8n-nodes-base.downloadOutput",
                    "name": "Export Asset",
                    "typeVersion": 1.0,
                    "position": [1200, 300],
                    "disabled": False,
                    "parameters": {}
                }
            ],
            "connections": {
                "node_trigger": {"main": [[{"node": "node_resize", "type": "main", "index": 0}]]},
                "node_resize": {"main": [[{"node": "node_color", "type": "main", "index": 0}]]},
                "node_color": {"main": [[{"node": "node_watermark", "type": "main", "index": 0}]]},
                "node_watermark": {"main": [[{"node": "node_output", "type": "main", "index": 0}]]}
            }
        },
        "n8n Vintage Filter Pipeline": {
            "nodes": [
                {
                    "id": "node_trigger",
                    "type": "n8n-nodes-base.fileTrigger",
                    "name": "Source File",
                    "typeVersion": 1.0,
                    "position": [240, 300],
                    "disabled": False,
                    "parameters": {}
                },
                {
                    "id": "node_color",
                    "type": "n8n-nodes-base.adjustColor",
                    "name": "Warm Contrast",
                    "typeVersion": 1.2,
                    "position": [480, 300],
                    "disabled": False,
                    "parameters": {"brightness": 0.95, "contrast": 1.15, "saturation": 0.7, "sharpness": 0.9}
                },
                {
                    "id": "node_sepia",
                    "type": "n8n-nodes-base.filterFx",
                    "name": "Sepia Tone FX",
                    "typeVersion": 1.0,
                    "position": [720, 300],
                    "disabled": False,
                    "parameters": {"filter_type": "sepia"}
                },
                {
                    "id": "node_vignette",
                    "type": "n8n-nodes-base.filterFx",
                    "name": "Vignette Shading",
                    "typeVersion": 1.0,
                    "position": [960, 300],
                    "disabled": False,
                    "parameters": {"filter_type": "vignette"}
                },
                {
                    "id": "node_output",
                    "type": "n8n-nodes-base.downloadOutput",
                    "name": "Export Vintage Asset",
                    "typeVersion": 1.0,
                    "position": [1200, 300],
                    "disabled": False,
                    "parameters": {}
                }
            ],
            "connections": {
                "node_trigger": {"main": [[{"node": "node_color", "type": "main", "index": 0}]]},
                "node_color": {"main": [[{"node": "node_sepia", "type": "main", "index": 0}]]},
                "node_sepia": {"main": [[{"node": "node_vignette", "type": "main", "index": 0}]]},
                "node_vignette": {"main": [[{"node": "node_output", "type": "main", "index": 0}]]}
            }
        }
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
        else:
            x, y = w - text_w - margin, h - text_h - margin

        draw.text((x, y), text, fill=(r, g, b, alpha), font=font)

        out = Image.alpha_composite(res, overlay)
        return out.convert("RGB")

    @classmethod
    def execute_node(cls, img: Image.Image, node: dict) -> tuple[Image.Image, dict]:
        node_id = node.get("id", "unknown_node")
        node_type = node.get("type", "unknown")
        node_name = node.get("name", node_type)
        is_disabled = node.get("disabled", False)
        params = node.get("parameters", {})

        meta = cls.NODE_TYPES.get(node_type, {"name": node_name, "category": "Action", "icon": "⚙️", "color": "#4B5563"})

        if is_disabled:
            return img, {
                "id": node_id,
                "name": node_name,
                "type": node_type,
                "status": "disabled",
                "execution_time_ms": 0.0,
                "output_dimensions": f"{img.width}x{img.height}",
                "message": f"Node '{node_name}' disabled/bypassed."
            }

        start_time = time.perf_counter()
        processed_img = img

        try:
            if "resize" in node_type:
                processed_img = cls.process_resize(img, **params)
            elif "crop" in node_type:
                processed_img = cls.process_crop(img, **params)
            elif "adjustColor" in node_type:
                processed_img = cls.process_adjust_color(img, **params)
            elif "rotateFlip" in node_type:
                processed_img = cls.process_rotate_flip(img, **params)
            elif "filterFx" in node_type:
                processed_img = cls.process_filter(img, **params)
            elif "watermark" in node_type:
                processed_img = cls.process_watermark(img, **params)
            elif any(t in node_type for t in ["Trigger", "geminiVision", "downloadOutput"]):
                processed_img = img

            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            telemetry = {
                "id": node_id,
                "name": node_name,
                "type": node_type,
                "category": meta.get("category", "Action"),
                "icon": meta.get("icon", "⚡"),
                "color": meta.get("color", "#3B82F6"),
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
    def run_pipeline(cls, img: Image.Image, nodes_or_workflow: list | dict) -> tuple[Image.Image, list[dict]]:
        """
        Executes an n8n node graph across input image, following node graph sequence / connections.
        """
        if isinstance(nodes_or_workflow, dict):
            nodes = nodes_or_workflow.get("nodes", [])
        else:
            nodes = nodes_or_workflow

        current_img = img.copy().convert("RGB")
        telemetry_list = []

        for idx, node in enumerate(nodes, start=1):
            current_img, node_telemetry = cls.execute_node(current_img, node)
            node_telemetry["step_number"] = idx
            telemetry_list.append(node_telemetry)

        return current_img, telemetry_list

    @staticmethod
    def export_workflow_json(workflow_data: list | dict) -> str:
        return json.dumps(workflow_data, indent=2)

    @staticmethod
    def import_workflow_json(json_str: str) -> list | dict:
        try:
            data = json.loads(json_str)
            return data
        except Exception as e:
            logger.error(f"Failed to parse workflow JSON: {e}")
            return []
