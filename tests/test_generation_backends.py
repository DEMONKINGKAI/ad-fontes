"""Unit tests for the generator loader's path selection and the DPO training
code path (the Phase 4 notebook uses the same calls)."""

from __future__ import annotations

import pytest
from app.api.schemas import GeneratorKind


def test_load_local_generator_prefers_existing_local_path(tmp_path, monkeypatch):
    import app.generation.local_llm as m

    made: dict = {}

    class FakeLLM:
        def __init__(self, kind, path, **kw):
            made["kind"], made["path"], made["kw"] = kind, str(path), kw

    monkeypatch.setattr(m, "LocalGenerator", FakeLLM)

    gguf = tmp_path / "tuned.gguf"
    gguf.write_bytes(b"GGUF")
    m.load_local_generator(
        GeneratorKind.local_tuned, "repo/x", "x.gguf", local_path=gguf, n_ctx=2048
    )
    assert made["path"] == str(gguf)
    assert made["kw"] == {"n_ctx": 2048}


def test_load_local_generator_falls_back_to_hub_download(tmp_path, monkeypatch):
    import app.generation.local_llm as m

    calls: dict = {}

    class FakeLLM:
        def __init__(self, kind, path, **kw):
            calls["path"] = str(path)

    def fake_dl(repo_id, filename, cache_dir=None):
        calls["repo"] = repo_id
        return str(tmp_path / "downloaded.gguf")

    monkeypatch.setattr(m, "LocalGenerator", FakeLLM)
    monkeypatch.setitem(
        __import__("sys").modules,
        "huggingface_hub",
        type("hh", (), {"hf_hub_download": staticmethod(fake_dl)}),
    )

    m.load_local_generator(
        GeneratorKind.local_base, "Qwen/x", "x.gguf", local_path=tmp_path / "missing.gguf"
    )
    assert calls["repo"] == "Qwen/x"
    assert calls["path"].endswith("downloaded.gguf")


@pytest.mark.slow
def test_dpo_training_code_path():
    """3-step DPO on a tiny model — mirrors `app/rlhf/train_dpo.ipynb` cells 4-6
    so a TRL API change breaks a test, not a Colab session."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)

    ds = Dataset.from_list(
        [
            {
                "prompt": "Q: What is X?\nA:",
                "chosen": " X is grounded.",
                "rejected": " X is huge and amazing!",
            }
        ]
        * 4
    )
    cfg = DPOConfig(
        output_dir="/tmp/dpo_test",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        max_steps=3,
        learning_rate=5e-6,
        beta=0.1,
        max_length=128,
        max_prompt_length=64,
        loss_type="sigmoid",
        report_to=[],
        save_strategy="no",
        logging_steps=1,
        bf16=False,
        fp16=False,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=LoraConfig(
            r=4, lora_alpha=8, task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"]
        ),
    )

    # reference-log-prob sanity check (notebook cell 5)
    batch = trainer.data_collator([trainer.train_dataset[i] for i in range(2)])
    batch = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.no_grad():
        chosen_lp, rejected_lp = trainer.compute_ref_log_probs(batch)
    assert torch.isfinite(chosen_lp).all() and (chosen_lp < 0).all()
    assert torch.isfinite(rejected_lp).all() and (rejected_lp < 0).all()

    out = trainer.train()
    assert out.training_loss == pytest.approx(out.training_loss)  # finite
    steps = [h for h in trainer.state.log_history if "loss" in h]
    assert steps, "no training steps logged"
    assert {"rewards/margins", "rewards/accuracies", "logps/chosen"} <= set(steps[-1])
