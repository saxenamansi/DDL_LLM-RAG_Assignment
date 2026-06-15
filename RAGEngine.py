"""
RAGEngine.py
=============
RAG query engine: prompt construction, generation, and multi-model runner.

Functions
---------
build_messages          : construct system/user message list
apply_chat_template     : apply model-specific chat template via tokenizer
format_context          : format retrieved chunks into a numbered context block
rag_query_single_model  : full RAG pipeline for one already-loaded model
run_all_models          : load → run all questions → unload, for each model
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from Config import (
    MODEL_DTYPE, MODEL_MAP, MODEL_SIZE_B, OUTPUT_DIR,
    QUANTIZED_MODELS, SAMPLING_CONFIG, TOP_K_PER_COMPANY,
)
from DataClasses import Chunk
from VectorStore import VectorStore


def setup_logger(log_path: str = "rag_run.log") -> logging.Logger:
    """
    Set up a logger that writes to a .log file only — not to the notebook.
    The notebook cell stays clean; check the log file for full details.
    """
    logger = logging.getLogger("rag")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)
    logger.propagate = False
    return logger


# ── Prompt helpers ─────────────────────────────────────────────────────────────

def build_messages(system_prompt: str, user_prompt: str) -> list:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]


def apply_chat_template(tokenizer, messages: list) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def format_context(retrieved: List[Chunk]) -> str:
    parts = []
    for i, c in enumerate(retrieved, 1):
        parts.append(
            f"[{i}] Company: {c.company} | File: {c.source} | Page: {c.page}\n"
            f"{c.text}"
        )
    return "\n\n".join(parts)


def save_model_output(
    model_key:   str,
    model_id:    str,
    questions:   Dict[str, str],
    results:     List[Dict],
    out_dir:     Path,
) -> None:
    """
    Save all answers from one model to a plain-text file.
    Includes model name, size, quantization, and per-question answers with sources.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_key}_answers.txt"

    size        = MODEL_SIZE_B.get(model_key, "?")
    quantized   = model_key in QUANTIZED_MODELS
    precision   = "4-bit (bitsandbytes)" if quantized else MODEL_DTYPE.get(model_key, "fp16")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"Model     : {model_id}\n")
        f.write(f"Key       : {model_key}\n")
        f.write(f"Size      : {size}\n")
        f.write(f"Precision : {precision}\n")
        f.write("=" * 70 + "\n\n")

        for r in results:
            f.write(f"── {r['question_label']} " + "─" * 40 + "\n")
            f.write(f"Question : {r['question']}\n")
            f.write(f"Latency  : {r['latency_seconds']}s | Tokens: {r['n_tokens']} | {r['finish_reason']}\n\n")
            f.write("Answer:\n")
            f.write(r["answer"])
            f.write("\n\nSources:\n")
            for s in eval(r["sources"]):
                f.write(f"  - {s['company']} | {s['file']} | p.{s['page']}\n")
            f.write("\n" + "─" * 70 + "\n\n")

    print(f"  [{model_key}] answers saved → {out_path}")


# ── Single-model RAG query ─────────────────────────────────────────────────────

def rag_query_single_model(
    question:          str,
    store:             VectorStore,
    llm,
    tokenizer,
    top_k_per_company: int = TOP_K_PER_COMPANY,
    extra_system:      str = "",
) -> Dict:
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=SAMPLING_CONFIG["temperature"],
        top_p=SAMPLING_CONFIG["top_p"],
        max_tokens=SAMPLING_CONFIG["max_tokens"],
        repetition_penalty=SAMPLING_CONFIG["repetition_penalty"],
    )

    retrieved = store.retrieve(question, top_k_per_company=top_k_per_company)
    context   = format_context(retrieved)

    system_prompt = (
        "You are an expert ESG analyst. Answer questions using ONLY the provided "
        "context excerpts from sustainability reports. "
        "Cite every fact with its source company, file, and page number in square brackets. "
        "If data is missing or ambiguous, say so explicitly — do not hallucinate numbers. "
        + extra_system
    )
    user_prompt = (
        f"Context excerpts:\n\n{context}\n\n"
        f"---\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    messages = build_messages(system_prompt, user_prompt)
    prompt   = apply_chat_template(tokenizer, messages)

    start   = time.time()
    output  = llm.generate([prompt], sampling_params)
    elapsed = round(time.time() - start, 2)

    answer  = output[0].outputs[0].text.strip()
    sources = [
        {"company": c.company, "file": c.source, "page": c.page}
        for c in retrieved
    ]

    return {
        "answer":          answer,
        "sources":         str(sources),
        "latency_seconds": elapsed,
        "n_tokens":        len(output[0].outputs[0].token_ids),
        "finish_reason":   output[0].outputs[0].finish_reason,
    }


# ── Multi-model runner ─────────────────────────────────────────────────────────

def run_all_models(
    questions:    Dict[str, str],
    store:        VectorStore,
    model_map:    Dict[str, str] = MODEL_MAP,
    extra_system: str = "",
    log_path:     str = "rag_run.log",
    out_dir:      Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """
    Load each model sequentially, run all questions, unload, repeat.

    Models in QUANTIZED_MODELS are loaded with 4-bit bitsandbytes quantization
    to fit on a single H100 80GB. All other models run in fp16/bfloat16.
    Model name, size, and quantization are recorded in both the per-model
    txt file and the summary DataFrame.

    All verbose output goes to log_path; notebook cell prints a short summary.
    """
    import torch
    from vllm import LLM

    # Suppress vLLM's verbose stdout logging
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
    logging.getLogger("vllm").setLevel(logging.ERROR)

    log = setup_logger(log_path)
    print(f"Logging full output to: {log_path}")

    all_results = []

    for model_key, model_id in model_map.items():
        quantized = model_key in QUANTIZED_MODELS
        size      = MODEL_SIZE_B.get(model_key, "?")
        precision = "4-bit" if quantized else MODEL_DTYPE.get(model_key, "fp16")

        log.info("=" * 60)
        log.info(f"Loading: {model_key} | {model_id} | {size} | {precision}")
        log.info("=" * 60)
        print(f"  [{model_key}] {size} ({precision}) loading...")

        llm_kwargs = dict(
            model                  = model_id,
            dtype                  = MODEL_DTYPE[model_key],
            gpu_memory_utilization = 0.90,
            tensor_parallel_size   = 1,
            tokenizer              = model_id,
            hf_overrides           = {"local_files_only": True},
            disable_log_stats      = True
        )
        if quantized:
            llm_kwargs["quantization"] = "bitsandbytes"
            llm_kwargs["load_format"]  = "bitsandbytes"

        try:
            llm = LLM(**llm_kwargs)
        except Exception as e:
            log.error(f"Failed to load {model_key}: {e}")
            print(f"  [{model_key}] ERROR loading — see {log_path}")
            continue

        tokenizer = llm.get_tokenizer()
        log.info(f"Model loaded. Running {len(questions)} questions...")

        model_results = []
        for q_label, q_text in questions.items():
            log.info(f"  Question: {q_label}")
            try:
                result = rag_query_single_model(
                    question=q_text,
                    store=store,
                    llm=llm,
                    tokenizer=tokenizer,
                    extra_system=extra_system,
                )
            except Exception as e:
                log.error(f"  FAILED {q_label} with {model_key}: {e}")
                print(f"  [{model_key}] ERROR on {q_label} — see {log_path}")
                continue

            log.info(f"  Done: {result['latency_seconds']}s | {result['n_tokens']} tokens | {result['finish_reason']}")
            log.info(f"  Answer:\n{result['answer']}")
            log.info(f"  Sources: {result['sources']}\n")
            print(f"  [{model_key}] {q_label} done in {result['latency_seconds']}s")

            row = {
                "model":           model_key,
                "model_id":        model_id,
                "model_size":      size,
                "quantization":    precision,
                "question_label":  q_label,
                "question":        q_text,
                "answer":          result["answer"],
                "sources":         result["sources"],
                "latency_seconds": result["latency_seconds"],
                "n_tokens":        result["n_tokens"],
                "finish_reason":   result["finish_reason"],
            }
            model_results.append(row)
            all_results.append(row)

        # Save this model's answers to a txt file
        save_model_output(model_key, model_id, questions, model_results, out_dir)

        log.info(f"Unloading {model_key}...")
        del llm
        torch.cuda.empty_cache()
        log.info("GPU memory freed.\n")
        print(f"  [{model_key}] unloaded.\n")

    print(f"Done. {len(all_results)} results — full output in {log_path}")
    return pd.DataFrame(all_results)