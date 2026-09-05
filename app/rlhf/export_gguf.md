# From DPO adapter to a served GGUF (Phase 4)

The order is: **train on Colab** (`train_dpo.ipynb`) → **merge** the QLoRA adapter
into the base model → **convert** to GGUF → **quantize** to Q4_K_M → **upload** to
the HF Hub → the API picks it up as `model: "tuned"`.

Every command below has been chosen to run inside a free Colab/Kaggle session or
locally; pin the versions shown.

---

## 0. Inputs

| | |
|---|---|
| base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| adapter | `qlora-dpo/` saved by `train_dpo.ipynb` to Drive (`adapter_model.safetensors` + `adapter_config.json`) |
| target repo | `DEMONKINGKAI/ad-fontes-generator-1.5b-dpo-gguf` |
| target file | `ad-fontes-1.5b-dpo-q4_k_m.gguf` (matches `AD_FONTES_TUNED_GGUF_FILE`) |

---

## 1. Merge the adapter into the base model  (Colab, ~3 min, CPU or GPU)

```python
# pip install "transformers>=4.46,<5" "peft>=0.13,<0.15" "torch>=2.2" --quiet
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "/content/drive/MyDrive/ad-fontes/qlora-dpo"   # from train_dpo.ipynb
OUT = "/content/ad-fontes-1.5b-dpo-merged"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, ADAPTER)
model = model.merge_and_unload()          # LoRA weights folded into the base
model.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
```

> Merge in **fp16**, not the 4-bit quantized model — merging into a bnb-4bit model
> silently degrades quality. `train_dpo.ipynb` trains 4-bit but this step reloads
> the base in fp16.

---

## 2. Convert the merged model to GGUF  (llama.cpp)

```bash
git clone --depth 1 https://github.com/ggerganov/llama.cpp
pip install -r llama.cpp/requirements.txt --quiet

python llama.cpp/convert_hf_to_gguf.py /content/ad-fontes-1.5b-dpo-merged \
  --outfile /content/ad-fontes-1.5b-dpo-f16.gguf --outtype f16
```

`convert_hf_to_gguf.py` reads the Qwen2 architecture directly — no extra flags
needed. Output is ~3 GB (f16).

---

## 3. Quantize to Q4_K_M

```bash
cmake -B llama.cpp/build llama.cpp -DLLAMA_CURL=OFF
cmake --build llama.cpp/build --target llama-quantize -j

llama.cpp/build/bin/llama-quantize \
  /content/ad-fontes-1.5b-dpo-f16.gguf \
  /content/ad-fontes-1.5b-dpo-q4_k_m.gguf Q4_K_M
```

Result is ~1.0 GB — the same footprint as the base GGUF, so the serving budget
(§2) is unchanged.

---

## 4. Sanity check locally  (llama-cpp-python, CPU)

```python
import json
from llama_cpp import Llama

llm = Llama("/content/ad-fontes-1.5b-dpo-q4_k_m.gguf", n_ctx=4096, verbose=False)
ctx = ('[1] id: threadfall#one-line-summary\nThreadfall > One-line summary\n'
       'A solo narrative RPG where story outcomes are decided by a deterministic '
       'causal engine, and the LLM only narrates what the engine has determined.')
out = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You answer about Kai's portfolio in the third person. "
         'Respond with JSON {"prose": ..., "claims": [{"text":..., "cite":[id]}]}.'},
        {"role": "user", "content": f"Passages:\n{ctx}\n\nQuestion: What is Threadfall?"},
    ],
    response_format={"type": "json_object"}, max_tokens=256, temperature=0.3,
)
print(json.loads(out["choices"][0]["message"]["content"]))
```

Expect valid `{prose, claims}` JSON, third person, one claim citing
`threadfall#one-line-summary`. If the format is broken the merge or convert step
went wrong — re-check step 1's dtype.

---

## 5. Upload to the HF Hub

```bash
huggingface-cli login   # or set HF_TOKEN
huggingface-cli upload DEMONKINGKAI/ad-fontes-generator-1.5b-dpo-gguf \
  /content/ad-fontes-1.5b-dpo-q4_k_m.gguf ad-fontes-1.5b-dpo-q4_k_m.gguf \
  --repo-type model
```

Add a short model card noting: base `Qwen2.5-1.5B-Instruct`, QLoRA DPO on
`ad-fontes` preference pairs, intended only for the ad-fontes portfolio assistant.

---

## 6. Serve it

Nothing to change in the API — `AD_FONTES_TUNED_GGUF_REPO` / `_FILE` already point
here, and the Docker build's `scripts.download_models --tuned-gguf` step becomes a
real download. To test before the upload, set `AD_FONTES_TUNED_GGUF_PATH` to the
local `.gguf`. `/api/health` then shows `generator:tuned` loaded and
`model: "tuned"` requests are served by it (`meta.model_requested` confirms).

---

## Record here after the real run (Phase 4)

- exact package versions: `trl`, `peft`, `transformers`, `bitsandbytes`, llama.cpp commit
- training: steps, final loss, `rewards/accuracies`, `rewards/margins`, wall-clock, Colab GPU
- reference-model log-prob sanity-check output
- merged-model vs. base perplexity on a held-out slice, if measured
- GGUF file size, load time on 2 vCPU
- link to the uploaded HF repo + commit
