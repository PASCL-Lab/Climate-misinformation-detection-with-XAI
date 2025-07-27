import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import json

class SimpleONNXConverter:
    def __init__(self, model_path, output_dir="./onnx_model"):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.tokenizer = None
        self.model = None

    def load_model_and_tokenizer(self):
        print("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.eval()

    def convert_to_onnx(self, opset_version=14, max_length=256):
        print("Converting to ONNX...")
        dummy_input_ids = torch.randint(0, self.tokenizer.vocab_size, (1, max_length), dtype=torch.long)
        dummy_attention_mask = torch.ones(1, max_length, dtype=torch.long)

        onnx_path = self.output_dir / "model.onnx"

        torch.onnx.export(
            self.model,
            (dummy_input_ids, dummy_attention_mask),
            str(onnx_path),
            export_params=True,
            opset_version=opset_version,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "logits": {0: "batch_size"}
            }
        )
        print(f"✅ ONNX model saved to: {onnx_path}")

    def save_tokenizer_and_config(self):
        print("Saving tokenizer and config...")
        # Save tokenizer
        tokenizer_path = self.output_dir / "tokenizer"
        self.tokenizer.save_pretrained(tokenizer_path)

        # Save config.json
        config_path = self.output_dir / "config.json"
        self.model.config.to_json_file(config_path)

        # Save model_info.json (optional, for metadata)
        info = {
            "model_type": self.model.config.model_type,
            "num_labels": self.model.config.num_labels,
            "max_length": 256,
            "label_mapping": {
                "0": "REFUTE",
                "1": "SUPPORT"
            }
        }
        info_path = self.output_dir / "model_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)

        print(f"✅ Tokenizer, config, and metadata saved to: {self.output_dir}")

    def run(self):
        self.load_model_and_tokenizer()
        self.convert_to_onnx()
        self.save_tokenizer_and_config()

# --------------- Run this ---------------
if __name__ == "__main__":
    MODEL_PATH = "backend/models/climate_bert_2_label_wch"  # Change to your actual model folder
    OUTPUT_DIR = "./onnx_model"

    converter = SimpleONNXConverter(MODEL_PATH, OUTPUT_DIR)
    converter.run()
