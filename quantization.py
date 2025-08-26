import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dotenv import load_dotenv
load_dotenv()
PYTORCH_MODEL_PATH = os.getenv("PYTORCH_MODEL_PATH")  # For XAI
# Load normal model
model_name = PYTORCH_MODEL_PATH

# Load tokenizer (still same)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load fine-tuned model (with classification head)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Quantize only Linear layers
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save the whole quantized model
torch.save(quantized_model, "distilroberta_finetuned_quantized.pt")
