from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# epoch 3
# model_path = 'outputs/exp008/20250917182004/upload'
# quant_path = 'outputs/exp008/20250917182004/upload-awq'

# epoch 2
model_path = 'outputs/exp008_14b_multi-fulltrain/20250917182004/checkpoint/checkpoint-288'
quant_path = 'outputs/exp008_14b_multi-fulltrain/20250917182004/upload-awq-4bit-ep2'

quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

model_init_kwargs = {
    "low_cpu_mem_usage": True,
    "use_cache": False,
    "max_memory": {
        0: "32GIB",
        1: "32GIB",
        2: "32GIB",
        3: "32GIB",
        4: "32GIB",
        5: "32GIB",
        6: "32GIB",
        7: "32GIB",
        8: "32GIB",
        9: "32GIB",
        10: "32GIB",
        11: "32GIB",
        12: "32GIB",
        13: "32GIB",
        14: "32GIB",
        15: "32GIB",
    }
}

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, **model_init_kwargs
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')