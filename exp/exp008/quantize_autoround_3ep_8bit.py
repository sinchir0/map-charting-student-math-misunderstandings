from auto_round import AutoRound

# Load a model (supports FP8/BF16/FP16/FP32)
model_path = "outputs/exp008/20250917182004/upload"
quant_path = 'outputs/exp008/20250917182004/upload-autoround-8bit'

# Available schemes: "W2A16", "W3A16", "W4A16", "W8A16", "NVFP4", "MXFP4" (no real kernels), "GGUF:Q4_K_M", etc.
# ar = AutoRound(model_name_or_path, scheme="W4A16")
ar = AutoRound(model_path, scheme="W8A16")

# Highest accuracy (4–5× slower).
# `low_gpu_mem_usage=True` saves ~20GB VRAM but runs ~30% slower.
# ar = AutoRound(model_name_or_path, nsamples=512, iters=1000, low_gpu_mem_usage=True)

# Faster quantization (2–3× speedup) with slight accuracy drop at W4G128.
# ar = AutoRound(model_name_or_path, nsamples=128, iters=50, lr=5e-3)

# Supported formats: "auto_round" (default), "auto_gptq", "auto_awq", "llm_compressor", "gguf:q4_k_m", etc.
ar.quantize_and_save(output_dir=quant_path, format="auto_round")