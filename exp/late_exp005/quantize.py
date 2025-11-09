from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

MODEL_NAME = "outputs/late_exp004/20251108134354/upload"
SAVE_PATH = "outputs/late_exp005/quantized_model"

if __name__ == "__main__":
    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, # 4 ビットに量子化された形式で読み込むように指定
            bnb_4bit_use_double_quant=True, # 二重量子化の指定
            bnb_4bit_quant_type="nf4", # 4 ビット量子化のデータ型として NF4 を指定
            bnb_4bit_compute_dtype=torch.bfloat16 # 計算時のデータ型を指定
        )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    print(model)
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)