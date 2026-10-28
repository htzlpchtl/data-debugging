"""Optional small synthetic-data SFT check; called only by remote/run.py.

Uses public-data selections and upstream hyperparameters, with standard PEFT /
Trainer instead of Unsloth. No general training framework or automatic sweep.
"""
import json
import random

from run import Runtime, evaluation_jobs, read_rows, remote_only, sha, write_json


def subsets(c):
    remote_only()
    n = c.cfg["training_subset_size"]
    if n <= 0:
        raise ValueError("Training is disabled in the scoring-only configuration.")
    rows = c.complete("scores", c.rows("candidates"))
    judged = {r["id"]: r for r in c.complete("judge_sample", rows)}
    if {r["question"] for r in rows} & set(c.evaluation["questions"] + c.extract["questions"]):
        raise ValueError("Training questions overlap extraction or evaluation.")

    def distinct_questions(ordered, excluded_ids=()):
        chosen, seen = [], set()
        for row in ordered:
            if row["question"] in seen or row["id"] in excluded_ids:
                continue
            chosen.append(row)
            seen.add(row["question"])
            if len(chosen) == n:
                return chosen
        raise ValueError("Too few distinct questions for requested subset.")

    ranked = sorted(rows, key=lambda r: (r["delta_projection"], r["id"]))
    high = distinct_questions(list(reversed(ranked)))
    low = distinct_questions(ranked, {r["id"] for r in high})
    shuffled = list(rows)
    random.Random(c.cfg["seed"]).shuffle(shuffled)
    arms = {"high": high, "low": low, "random": distinct_questions(shuffled)}
    destination = c.out / "subsets"
    destination.mkdir(exist_ok=True)
    summary = {}
    for name, chosen in arms.items():
        path = destination / f"{name}.jsonl"
        if path.exists() and read_rows(path) != chosen:
            raise ValueError("Existing subset differs; refusing replacement.")
        path.write_text("".join(json.dumps(r) + "\n" for r in chosen))
        text_scores = [judged[r["id"]]["trait"] for r in chosen if judged[r["id"]]["trait"] is not None]
        summary[name] = {"n": len(chosen), "sha256": sha(path),
            "mean_delta_projection": sum(r["delta_projection"] for r in chosen) / len(chosen),
            "mean_response_tokens": sum(r["response_tokens"] for r in chosen) / len(chosen),
            "fraction_hallucination_II": sum(r["label"] for r in chosen) / len(chosen),
            "text_judge_valid": len(text_scores),
            "mean_text_judge": sum(text_scores) / len(text_scores) if text_scores else None}
    summary["high_low_shared_questions"] = len({r["question"] for r in high} & {r["question"] for r in low})
    summary["scope"] = "One exploratory seed; subsets not matched on semantics or length. No incremental-prediction claim. Random arm may overlap either extreme arm."
    write_json(destination / "manifest.json", summary)


def train(c, arm):
    remote_only()
    c.require_preflight()
    if arm == "base" or c.cfg["training_subset_size"] <= 0:
        raise ValueError("Choose high, low or random in an SFT-enabled config.")
    source = c.out / "subsets" / f"{arm}.jsonl"
    manifest = json.loads((source.parent / "manifest.json").read_text())
    if sha(source) != manifest[arm]["sha256"]:
        raise ValueError("Subset was changed after selection.")
    destination = c.out / "adapters" / arm
    if (destination / "complete.json").exists():
        print(f"{arm}: already complete; no training repeated.")
        return
    # The baseline evaluation must be frozen before any training outcome is inspected.
    c.complete("evaluation_base", evaluation_jobs(c))
    from transformers import Trainer, TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    from peft import LoraConfig, get_peft_model
    set_seed(c.cfg["training_seed"])
    runtime = Runtime(c)
    model, tokenizer, torch = runtime.model, runtime.tokenizer, runtime.torch
    records, lengths, response_lengths = [], [], []
    for row in read_rows(source):
        messages = row["messages"]
        if len(messages) != 2 or [m["role"] for m in messages] != ["user", "assistant"]:
            raise ValueError("This small runner supports one user/assistant turn only.")
        prefix = tokenizer.encode(runtime.prompt(messages[:-1]), add_special_tokens=False)
        full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        if c.cfg.get("append_eos_token", False):
            # The official sft.py appends tokenizer.eos_token to the rendered chat.
            full += tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)
        if full[:len(prefix)] != prefix or len(full) <= len(prefix):
            raise ValueError("Response loss mask does not align with chat template.")
        if len(full) > c.cfg["max_sequence_tokens"]:
            raise ValueError("Training example exceeds token limit; no silent truncation or dropping.")
        records.append({"input_ids": full, "attention_mask": [1] * len(full),
                        "labels": [-100] * len(prefix) + full[len(prefix):]})
        lengths.append(len(full))
        response_lengths.append(len(full) - len(prefix))
    if len(records) != c.cfg["training_subset_size"]:
        raise ValueError("Training row count mismatch.")

    def collate(batch):
        width = max(len(r["input_ids"]) for r in batch)
        return {key: torch.tensor([r[key] + [padding] * (width - len(r[key])) for r in batch])
                for key, padding in (("input_ids", tokenizer.pad_token_id), ("attention_mask", 0), ("labels", -100))}

    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0,
        bias="none", task_type="CAUSAL_LM", use_rslora=True,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.enable_input_require_grads()
    model.config.use_cache = False
    args = TrainingArguments(output_dir=str(destination), num_train_epochs=c.cfg["training_epochs"],
        per_device_train_batch_size=c.cfg["training_batch_size"],
        gradient_accumulation_steps=c.cfg["gradient_accumulation_steps"],
        learning_rate=1e-5, warmup_steps=5, lr_scheduler_type="linear", weight_decay=0.01,
        optim="adamw_8bit", bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=c.cfg["training_seed"], data_seed=c.cfg["training_seed"],
        logging_steps=1, save_steps=10, save_total_limit=1, report_to="none",
        eval_strategy="no", dataloader_num_workers=0, remove_unused_columns=False)
    checkpoint = get_last_checkpoint(str(destination)) if destination.exists() else None
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "input_manifest.json", {"source_sha256": sha(source), "n": len(records),
        "epochs": c.cfg["training_epochs"], "input_tokens_per_epoch": sum(lengths),
        "supervised_tokens_per_epoch": sum(response_lengths), "max_input_tokens": max(lengths),
        "effective_batch_size": c.cfg["training_batch_size"] * c.cfg["gradient_accumulation_steps"],
        "validation": "No implicit split. Separate public behavioural evaluation prompts.",
        "framework_deviation": "Transformers Trainer + PEFT instead of Unsloth/TRL"})
    trainer = Trainer(model=model, args=args, train_dataset=records,
                      data_collator=collate, processing_class=tokenizer)
    result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(destination))
    tokenizer.save_pretrained(destination)
    write_json(destination / "complete.json", {"arm": arm, "metrics": result.metrics,
        "global_step": trainer.state.global_step, "epoch": trainer.state.epoch,
        "adapter_sha256": sha(destination / "adapter_model.safetensors")})
