import os
import sys
import json
import tempfile
import unittest

import train_lora

try:
    from safetensors.torch import load_file as load_safetensors_file
except ImportError:
    load_safetensors_file = None

class TestImageLoRAStudio(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_dir = os.path.join(self.temp_dir.name, "dataset")
        self.output_dir = os.path.join(self.temp_dir.name, "output")
        self.config_dir = os.path.join(self.temp_dir.name, "config")
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parse_args_defaults(self):
        sys.argv = ["train_lora.py"]
        args = train_lora.parse_args()
        self.assertEqual(args.base_model, "black-forest-labs/FLUX.1-dev")
        self.assertEqual(args.resolution, 1024)
        self.assertEqual(args.lora_r, 16)
        self.assertEqual(args.mixed_precision, "fp16")

    def test_save_config(self):
        sys.argv = [
            "train_lora.py",
            "--base_model", "stabilityai/stable-diffusion-xl-base-1.0",
            "--config_save_dir", self.config_dir,
            "--resolution", "512",
            "--lora_r", "32"
        ]
        args = train_lora.parse_args()
        config_path = train_lora.save_config(args)

        self.assertTrue(os.path.exists(config_path))
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["base_model"], "stabilityai/stable-diffusion-xl-base-1.0")
        self.assertEqual(data["resolution"], 512)
        self.assertEqual(data["lora_r"], 32)

    def test_run_training_execution(self):
        sys.argv = [
            "train_lora.py",
            "--dataset_dir", self.dataset_dir,
            "--output_dir", self.output_dir,
            "--config_save_dir", self.config_dir,
            "--save_every_n_epochs", "1",
            "--use_safetensors"
        ]
        args = train_lora.parse_args()
        train_lora.run_training(args)

        files = os.listdir(self.output_dir)
        safetensors_files = [f for f in files if f.endswith(".safetensors")]
        self.assertGreater(len(safetensors_files), 0)

        checkpoint_path = os.path.join(self.output_dir, safetensors_files[0])
        if load_safetensors_file is not None:
            weights = load_safetensors_file(checkpoint_path)
            self.assertIn("lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight", weights)
        else:
            self.assertTrue(os.path.getsize(checkpoint_path) > 0)

    def test_app_import(self):
        try:
            import app
            self.assertIsNotNone(app.DATASET_DIR)
        except Exception as e:
            self.fail(f"Failed to import app.py: {e}")

if __name__ == "__main__":
    unittest.main()
