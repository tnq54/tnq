import unittest
import json
import io
from PIL import Image
from workflow_engine import ImageWorkflowEngine
import app

class TestImageWorkflowEngine(unittest.TestCase):

    def setUp(self):
        self.test_img = Image.new("RGB", (200, 200), color=(255, 0, 0))

    def test_resize(self):
        resized = ImageWorkflowEngine.process_resize(self.test_img, width=100, height=100, maintain_aspect_ratio=False)
        self.assertEqual(resized.size, (100, 100))

    def test_crop(self):
        cropped = ImageWorkflowEngine.process_crop(self.test_img, left_pct=10, top_pct=10, right_pct=90, bottom_pct=90)
        self.assertEqual(cropped.size, (160, 160))

    def test_adjust_color(self):
        adjusted = ImageWorkflowEngine.process_adjust_color(self.test_img, brightness=1.2, contrast=1.1, saturation=1.3, sharpness=1.1)
        self.assertEqual(adjusted.size, self.test_img.size)

    def test_rotate_flip(self):
        rotated = ImageWorkflowEngine.process_rotate_flip(self.test_img, angle=90, flip_h=True, flip_v=False)
        self.assertEqual(rotated.size, (200, 200))

    def test_filters(self):
        filter_types = ["grayscale", "sepia", "blur", "contour", "edge_enhance", "invert", "posterize", "vignette"]
        for ft in filter_types:
            filtered = ImageWorkflowEngine.process_filter(self.test_img, filter_type=ft, radius=2)
            self.assertEqual(filtered.size, self.test_img.size)

    def test_watermark(self):
        watermarked = ImageWorkflowEngine.process_watermark(self.test_img, text="Test Watermark", position="bottom-right")
        self.assertEqual(watermarked.size, self.test_img.size)

    def test_n8n_node_graph_execution(self):
        workflow = {
            "nodes": [
                {"id": "n1", "type": "n8n-nodes-base.fileTrigger", "name": "Trigger", "position": [240, 300], "disabled": False, "parameters": {}},
                {"id": "n2", "type": "n8n-nodes-base.resize", "name": "Resize 100", "position": [480, 300], "disabled": False, "parameters": {"width": 100, "height": 100, "maintain_aspect_ratio": False}},
                {"id": "n3", "type": "n8n-nodes-base.filterFx", "name": "Sepia", "position": [720, 300], "disabled": False, "parameters": {"filter_type": "sepia"}},
                {"id": "n4", "type": "n8n-nodes-base.watermark", "name": "Watermark", "position": [960, 300], "disabled": False, "parameters": {"text": "n8n Test"}},
                {"id": "n5", "type": "n8n-nodes-base.downloadOutput", "name": "Output", "position": [1200, 300], "disabled": False, "parameters": {}}
            ]
        }
        out_img, telemetry = ImageWorkflowEngine.run_pipeline(self.test_img, workflow)
        self.assertEqual(out_img.size, (100, 100))
        self.assertEqual(len(telemetry), 5)
        for item in telemetry:
            self.assertEqual(item["status"], "success")

    def test_n8n_node_disabled_state(self):
        workflow = {
            "nodes": [
                {"id": "n1", "type": "n8n-nodes-base.resize", "name": "Disabled Resize", "position": [240, 300], "disabled": True, "parameters": {"width": 50, "height": 50, "maintain_aspect_ratio": False}},
                {"id": "n2", "type": "n8n-nodes-base.filterFx", "name": "Active Filter", "position": [480, 300], "disabled": False, "parameters": {"filter_type": "sepia"}}
            ]
        }
        out_img, telemetry = ImageWorkflowEngine.run_pipeline(self.test_img, workflow)
        self.assertEqual(out_img.size, (200, 200))
        self.assertEqual(telemetry[0]["status"], "disabled")
        self.assertEqual(telemetry[1]["status"], "success")

    def test_n8n_presets(self):
        for preset_name, workflow in ImageWorkflowEngine.PRESETS.items():
            out_img, telemetry = ImageWorkflowEngine.run_pipeline(self.test_img, workflow)
            self.assertIsNotNone(out_img)
            self.assertGreater(len(telemetry), 0)

    def test_json_export_import(self):
        workflow = {
            "nodes": [
                {"id": "n1", "type": "n8n-nodes-base.filterFx", "name": "Sepia", "disabled": False, "parameters": {"filter_type": "sepia"}}
            ]
        }
        json_str = ImageWorkflowEngine.export_workflow_json(workflow)
        imported = ImageWorkflowEngine.import_workflow_json(json_str)
        self.assertEqual(workflow, imported)

    def test_pdf_text_extraction(self):
        text = app.extract_pdf_text(b"invalid pdf data")
        self.assertIsNone(text)

    def test_gemini_summarization_fallback(self):
        res = app.summarize_with_gemini("Test text")
        self.assertIn("GOOGLE_API_KEY not found", res)

    def test_analyze_image_fallback(self):
        img_bytes = io.BytesIO()
        self.test_img.save(img_bytes, format="PNG")
        res = app.analyze_image_with_gemini(img_bytes.getvalue())
        self.assertIn("Gemini API unavailable", res)

if __name__ == "__main__":
    unittest.main()
