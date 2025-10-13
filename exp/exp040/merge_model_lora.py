from transformers import AutoModelForCausalLM
import torch
from peft import PeftModel
from pathlib import Path

MODEL_NAME = "Qwen/Qwen3-32B-AWQ"
ADAPTER_PATH = Path("outputs/exp039/202510131723/upload")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model = model.merge_and_unload()

model.save_pretrained(ADAPTER_PATH.parent / "merged_model")