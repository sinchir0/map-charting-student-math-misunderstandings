from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# epoch 3 4bit
# model_path = 'outputs/exp008/20250917182004/upload'
# quant_path = 'outputs/exp008/20250917182004/upload-awq'

# epoch 3 8bit
model_path = 'outputs/exp008_14b_multi-fulltrain/20250917182004/upload'
quant_path = 'outputs/exp008_14b_multi-fulltrain/20250917182004/upload-awq-ep3-8bit'

quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 8, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')