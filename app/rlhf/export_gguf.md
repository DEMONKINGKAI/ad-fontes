# Exporting the tuned generator to GGUF (Phase 4)

> Filled in with exact, reproduced commands during Phase 4. This is the skeleton
> so the steps are agreed up front.

## 0. Inputs
- QLoRA adapter checkpoint from `train_dpo.ipynb` (on Drive).
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`.

## 1. Merge adapter into the base model
```bash
python -m peft.utils.merge_and_unload ...   # exact invocation recorded in Phase 4
# -> ./ad-fontes-1.5b-dpo-merged  (HF format, fp16)
```

## 2. Convert to GGUF
```bash
git clone https://github.com/ggerganov/llama.cpp
python llama.cpp/convert_hf_to_gguf.py ./ad-fontes-1.5b-dpo-merged \
  --outfile ad-fontes-1.5b-dpo-f16.gguf --outtype f16
```

## 3. Quantize to Q4_K_M
```bash
./llama.cpp/llama-quantize ad-fontes-1.5b-dpo-f16.gguf \
  ad-fontes-1.5b-dpo-q4_k_m.gguf Q4_K_M
```

## 4. Sanity check locally
```bash
python -c "from llama_cpp import Llama; ..."   # schema-constrained smoke test
```

## 5. Upload to HF Hub
```bash
huggingface-cli upload DEMONKINGKAI/ad-fontes-generator-1.5b-dpo-gguf \
  ad-fontes-1.5b-dpo-q4_k_m.gguf
```

## 6. Wire into serving
Set `AD_FONTES_TUNED_GGUF_REPO` / `_FILE`; the Docker build bakes it into the image.

## Record here (Phase 4)
- exact package versions (llama.cpp commit, transformers, peft, trl)
- merged-model perplexity vs base, if measured
- GGUF file size, load time on 2 vCPU
