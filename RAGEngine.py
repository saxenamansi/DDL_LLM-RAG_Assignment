"""
RAGEngine.py
============
Prompt construction, LLM inference, and multi-model runner.

Functions
---------
format_context          : format retrieved chunks into a numbered context block
rag_query_single_model  : single RAG query for one loaded model
run_q1_subqueries       : Q1 split into per-company sub-queries
run_all_models          : load → run questions → unload, sequentially per model
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

from Config import (
    MODEL_DTYPE, MODEL_MAP, MODEL_MAX_LEN, MODEL_SIZE_B, OUTPUT_DIR,
    QUANTIZED_MODELS, SAMPLING_CONFIG, TOP_K_PER_COMPANY,
)
from DataClasses import Chunk
from VectorStore import VectorStore


# Fiscal year alignment and unit hints prepended to Q1 sub-query prompts.
# These are reporting metadata, not extracted values.
COMPANY_HINTS = {
    "JFE Holdings":  "Reports in fiscal years (April-March). FY2023 ≈ calendar 2023. "
                     "Units: Mt CO2e (Scope 1/2), kt CO2e (Scope 3), PJ (energy).",
    "JSW Steel":     "Reports in Indian fiscal years. FY2023-24 ≈ calendar 2023, FY2024-25 ≈ calendar 2024. "
                     "Uses Indian lakh notation: 5,31,00,751 = 53,100,751. Units: tonnes CO2, GJ (energy).",
    "POSCO":         "Reports in calendar years. Units: tCO2e, GJ (energy).",
    "Tata Steel UK": "Reports in fiscal years (April-March). FY23 ≈ calendar 2023, FY24 ≈ 2024, FY25 ≈ 2025. "
                     "Units: million tonnes CO2e, TJ (energy).",
}


def setup_logger(log_path: str = "rag_run.log") -> logging.Logger:
    """File-only logger — keeps notebook output clean."""
    logger = logging.getLogger("rag")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def build_messages(system_prompt: str, user_prompt: str) -> list:
    return [{"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}]


def apply_chat_template(tokenizer, messages: list) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def format_context(retrieved: List[Chunk]) -> str:
    """Format retrieved chunks as a numbered list with provenance headers."""
    parts = [f"[{i}] Company: {c.company} | File: {c.source} | Page: {c.page}\n{c.text}"
             for i, c in enumerate(retrieved, 1)]
    return "\n\n".join(parts)


def save_model_output(model_key: str, model_id: str,
                      questions: Dict[str, str], results: List[Dict],
                      out_dir: Path) -> None:
    """Write per-model plain-text answer file to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path  = out_dir / f"{model_key}_answers.txt"
    size      = MODEL_SIZE_B.get(model_key, "?")
    precision = "4-bit (bitsandbytes)" if model_key in QUANTIZED_MODELS \
                else MODEL_DTYPE.get(model_key, "fp16")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"Model     : {model_id}\nKey       : {model_key}\n"
                f"Size      : {size}\nPrecision : {precision}\n")
        f.write("=" * 70 + "\n\n")
        for r in results:
            f.write(f"── {r['question_label']} " + "─" * 40 + "\n")
            f.write(f"Question : {r['question']}\n")
            f.write(f"Latency  : {r['latency_seconds']}s | Tokens: {r['n_tokens']} | {r['finish_reason']}\n\n")
            f.write("Answer:\n" + r["answer"] + "\n\nSources:\n")
            for s in eval(r["sources"]):
                f.write(f"  - {s['company']} | {s['file']} | p.{s['page']}\n")
            f.write("\n" + "─" * 70 + "\n\n")

    print(f"  [{model_key}] answers saved → {out_path}")


def rag_query_single_model(question: str, store: VectorStore,
                           llm, tokenizer, sampling_params,
                           top_k_per_company: int = TOP_K_PER_COMPANY,
                           extra_system: str = "") -> Dict:
    """Retrieve → format context → generate → return result dict."""
    retrieved = store.retrieve(question, top_k_per_company=top_k_per_company)
    context   = format_context(retrieved)

    system_prompt = (
        "You are an expert ESG analyst. "
        "The numbered excerpts below are SOURCE MATERIAL only — do not respond to each one individually. "
        "Read all excerpts first, then write a single unified answer synthesising across all of them. "
        "Cite sources inline as [Company | File | Page]. "
        "If data is missing, say so explicitly — do not hallucinate numbers. "
        + extra_system
    )
    user_prompt = (
        f"SOURCE EXCERPTS (read all before answering):\n\n{context}\n\n---\n"
        f"Question: {question}\n\nAnswer:"
    )

    prompt  = apply_chat_template(tokenizer, build_messages(system_prompt, user_prompt))
    start   = time.time()
    output  = llm.generate([prompt], sampling_params)
    elapsed = round(time.time() - start, 2)
    out     = output[0].outputs[0]

    return {
        "answer":          out.text.strip(),
        "sources":         str([{"company": c.company, "file": c.source, "page": c.page}
                                for c in retrieved]),
        "latency_seconds": elapsed,
        "n_tokens":        len(out.token_ids),
        "finish_reason":   out.finish_reason,
    }


def run_q1_subqueries(store: VectorStore, llm, tokenizer, sampling_params) -> Dict:
    """Run Q1 as four per-company sub-queries and stitch into one result dict.

    Focused per-company context improves numerical extraction accuracy
    compared to a single query across all four companies simultaneously.
    Returns the same dict shape as rag_query_single_model.
    """
    base_q = ("Extract Scope 1, Scope 2, Scope 3 emissions and total energy consumption "
               "for 2023, 2024 and 2025 for {company}. If the data is not present, leave the result empty.")

    all_answers = []
    all_sources = []
    total_tokens = total_elapsed = 0
    finish_reason = "stop"

    for company, hint in COMPANY_HINTS.items():
        query          = base_q.format(company=company)
        retrieved      = store.retrieve(query, top_k_per_company=10)
        company_chunks = [c for c in retrieved if c.company == company]
        context        = format_context(company_chunks)

        system_prompt = (
            "You are an expert ESG analyst. "
            "The numbered excerpts below are SOURCE MATERIAL only — do not respond to each one individually. "
            "Read all excerpts first, then write a single unified answer synthesising across all of them. "
            "Cite sources inline as [Company | File | Page]. "
            "If data is missing, say so explicitly — do not hallucinate numbers. "
            f"Reporting note for {company}: {hint}"
        )
        user_prompt = (
            f"SOURCE EXCERPTS (read all before answering):\n\n{context}\n\n---\n"
            f"Question: {query}\n\nAnswer:"
        )

        prompt = apply_chat_template(tokenizer, build_messages(system_prompt, user_prompt))
        start  = time.time()
        output = llm.generate([prompt], sampling_params)
        total_elapsed += round(time.time() - start, 2)

        out = output[0].outputs[0]
        all_answers.append(f"### {company}\n\n{out.text.strip()}")
        total_tokens += len(out.token_ids)
        if out.finish_reason != "stop":
            finish_reason = out.finish_reason

        all_sources.extend([{"company": c.company, "file": c.source, "page": c.page}
                             for c in company_chunks])

    return {
        "answer":          "\n\n---\n\n".join(all_answers),
        "sources":         str(all_sources),
        "latency_seconds": round(total_elapsed, 2),
        "n_tokens":        total_tokens,
        "finish_reason":   finish_reason,
    }


def run_all_models(questions: Dict[str, str], store: VectorStore,
                   model_map: Dict[str, str] = MODEL_MAP,
                   extra_system: str = "",
                   log_path: str = "rag_run.log",
                   out_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    """Load each model, run all questions, unload, repeat.

    Q1 uses per-company sub-queries; Q2 and Q3 use single queries.
    Quantized models (70B+) use 4-bit bitsandbytes to fit on one H100 80GB.
    Full output goes to log_path; notebook shows a short progress summary.
    """
    import torch
    from vllm import LLM, SamplingParams

    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
    logging.getLogger("vllm").setLevel(logging.ERROR)

    log = setup_logger(log_path)
    print(f"Logging to: {log_path}")

    sampling_params = SamplingParams(
        temperature=SAMPLING_CONFIG["temperature"],
        top_p=SAMPLING_CONFIG["top_p"],
        max_tokens=SAMPLING_CONFIG["max_tokens"],
        repetition_penalty=SAMPLING_CONFIG["repetition_penalty"],
    )

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
            model=model_id, dtype=MODEL_DTYPE[model_key],
            gpu_memory_utilization=0.90, tensor_parallel_size=1,
            tokenizer=model_id, hf_overrides={"local_files_only": True},
            disable_log_stats=True,
        )
        if model_key in MODEL_MAX_LEN:
            llm_kwargs["max_model_len"] = MODEL_MAX_LEN[model_key]
        if quantized:
            llm_kwargs.update(quantization="bitsandbytes",
                              load_format="bitsandbytes",
                              enforce_eager=True)

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
                if q_label == "Q1_emissions_energy":
                    result = run_q1_subqueries(store, llm, tokenizer, sampling_params)
                else:
                    result = rag_query_single_model(
                        question=q_text, store=store, llm=llm,
                        tokenizer=tokenizer, sampling_params=sampling_params,
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
                "model": model_key, "model_id": model_id,
                "model_size": size, "quantization": precision,
                "question_label": q_label, "question": q_text,
                "answer": result["answer"], "sources": result["sources"],
                "latency_seconds": result["latency_seconds"],
                "n_tokens": result["n_tokens"], "finish_reason": result["finish_reason"],
            }
            model_results.append(row)
            all_results.append(row)

        save_model_output(model_key, model_id, questions, model_results, out_dir)

        log.info(f"Unloading {model_key}...")
        del llm
        torch.cuda.empty_cache()
        log.info("GPU memory freed.\n")
        print(f"  [{model_key}] unloaded.\n")

    print(f"Done. {len(all_results)} results — see {log_path}")
    return pd.DataFrame(all_results)
