from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from core.complexity_estimation.entropy.logit_entropy import compute_entropy_from_logits
from core.entropy_dynamics.config import EntropyDynamicsConfig, InferenceMode, StudentModelConfig
from core.entropy_dynamics.prompt_builder import PrefixedPrompt, build_prefixed_prompts
from core.entropy_dynamics.reasoning_loader import TeacherReasoning, load_teacher_reasoning
from core.prompts.mmlu_cot_answer import answer_marker
from core.utils.device import DEVICE, DEVICE_MAP
from core.utils.seed import set_seed


@dataclass
class StepResult:
    """Single measurement: one model × one question × one k."""
    question_id: str
    model_label: str          # was "student_label" — now generic
    role: str                 # "student" or "proxy"
    k: int
    num_reasoning_tokens: int
    total_reasoning_tokens: int
    mode: str
    answer_entropy: float
    model_answer: str         # was "student_answer"
    model_correct: bool       # was "student_correct"
    gold_answer: str


@dataclass
class ExperimentResults:
    rows: list[dict] = field(default_factory=list)

    def append(self, result: StepResult):
        self.rows.append(result.__dict__)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)


class EntropyDynamicsRunner:
    """Runs the full entropy dynamics experiment for any role (student or proxy)."""

    def __init__(self, config: EntropyDynamicsConfig):
        self.config = config

    def run(self) -> pd.DataFrame:
        set_seed()

        first_model_id = self.config.students[0].model_id
        loader_tokenizer = AutoTokenizer.from_pretrained(first_model_id)

        print(f"Loading teacher reasoning from {self.config.teacher_reasoning_path}...")
        samples = load_teacher_reasoning(
            self.config.teacher_reasoning_path,
            tokenizer=loader_tokenizer,
            min_thinking_tokens=self.config.window_size,
        )
        print(f"Loaded {len(samples)} samples with valid reasoning chains.")

        all_results = ExperimentResults()

        for model_cfg in self.config.students:
            self._run_single_model(model_cfg, samples, all_results)

        # ── Use unique filename to prevent overwrites ──
        out_path = self.config.out_path / self.config.results_filename
        all_results.save(out_path)
        print(f"All results saved to {out_path}")

        return all_results.to_dataframe()

    def _run_single_model(
        self,
        model_cfg: StudentModelConfig,
        samples: list[TeacherReasoning],
        results: ExperimentResults,
    ):
        role = self.config.role.value
        print(f"\n{'='*60}")
        print(f"[{role}] {model_cfg.label} ({model_cfg.model_id})")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_cfg.model_id,
            device_map=DEVICE_MAP,
            torch_dtype=torch.bfloat16,
        )
        model.eval()

        # ── Checkpoint includes role to avoid collision ──
        checkpoint_path = (
            self.config.out_path
            / f"checkpoint_{role}_{model_cfg.label}_{self.config.mode.value}.parquet"
        )
        processed_ids = _load_processed_ids(checkpoint_path)
        print(f"Resuming: {len(processed_ids)} questions already processed.")

        t_start = time.perf_counter()

        for i, sample in enumerate(tqdm(samples, desc=f"[{role}/{model_cfg.label}]")):
            if sample.question_id in processed_ids:
                continue

            # Re-tokenize with this model's tokenizer
            sample.thinking_token_ids = tokenizer.encode(
                sample.thinking_text, add_special_tokens=False
            )

            prefixed_prompts = build_prefixed_prompts(
                sample=sample,
                tokenizer=tokenizer,
                window_size=self.config.window_size,
                mode=self.config.mode,
                dataset_type=self.config.dataset_type,
            )

            for prompt in prefixed_prompts:
                step_result = self._run_single_step(
                    model, tokenizer, prompt, model_cfg.label, role
                )
                results.append(step_result)

            if (i + 1) % self.config.batch_save_every == 0:
                results.save(checkpoint_path)
                elapsed = time.perf_counter() - t_start
                print(f"  [{role}/{model_cfg.label}] {i+1}/{len(samples)} "
                      f"({elapsed:.0f}s elapsed)")

        results.save(checkpoint_path)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _run_single_step(
        self,
        model,
        tokenizer: PreTrainedTokenizer,
        prompt: PrefixedPrompt,
        model_label: str,
        role: str,
    ) -> StepResult:
        """Run one forward pass and extract entropy."""
        input_ids = torch.tensor([prompt.input_ids], device=DEVICE)
        attention_mask = torch.ones_like(input_ids)

        if prompt.mode == InferenceMode.FORCED:
            return self._step_forced(
                model, tokenizer, input_ids, attention_mask, prompt, model_label, role
            )
        else:
            return self._step_continuation(
                model, tokenizer, input_ids, attention_mask, prompt, model_label, role
            )

    def _step_forced(
        self, model, tokenizer, input_ids, attention_mask,
        prompt: PrefixedPrompt, model_label: str, role: str,
    ) -> StepResult:
        """Mode A: generate 1 token, measure its entropy."""
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
            do_sample=self.config.do_sample,
            temperature=self.config.temperature if self.config.do_sample else None,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
        )

        first_token_logits = outputs.scores[0][0]
        entropy = compute_entropy_from_logits(first_token_logits).item()

        generated_id = outputs.sequences[0, input_ids.shape[1]].item()
        answer = tokenizer.decode([generated_id]).strip().lower()

        return StepResult(
            question_id=prompt.question_id,
            model_label=model_label,
            role=role,
            k=prompt.k,
            num_reasoning_tokens=prompt.num_reasoning_tokens,
            total_reasoning_tokens=prompt.total_reasoning_tokens,
            mode=prompt.mode.value,
            answer_entropy=entropy,
            model_answer=answer,
            model_correct=(answer == prompt.gold_answer),
            gold_answer=prompt.gold_answer,
        )

    def _step_continuation(
        self, model, tokenizer, input_ids, attention_mask,
        prompt: PrefixedPrompt, model_label: str, role: str,
    ) -> StepResult:
        """Mode B: model continues generating, measure avg entropy of tail."""
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.config.max_new_tokens_continuation,
            do_sample=self.config.do_sample,
            temperature=self.config.temperature if self.config.do_sample else None,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
        )

        gen_ids = outputs.sequences[0, input_ids.shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        answer = ""
        marker_pos = gen_text.find(answer_marker[1])
        marker_start = gen_text.rfind(answer_marker[0], 0, marker_pos if marker_pos != -1 else None)
        if marker_start != -1 and marker_pos != -1:
            answer = gen_text[marker_start + len(answer_marker[0]):marker_pos].strip().lower()

        scores = outputs.scores
        if not scores:
            entropy = float("nan")
        else:
            n_scores = len(scores)
            tail_size = min(self.config.tail_entropy_window, n_scores)
            tail_logits = torch.stack(
                [scores[i][0] for i in range(n_scores - tail_size, n_scores)],
                dim=0,
            )
            tail_entropies = compute_entropy_from_logits(tail_logits)
            entropy = tail_entropies.mean().item()

        return StepResult(
            question_id=prompt.question_id,
            model_label=model_label,
            role=role,
            k=prompt.k,
            num_reasoning_tokens=prompt.num_reasoning_tokens,
            total_reasoning_tokens=prompt.total_reasoning_tokens,
            mode=prompt.mode.value,
            answer_entropy=entropy,
            model_answer=answer,
            model_correct=(answer == prompt.gold_answer),
            gold_answer=prompt.gold_answer,
        )


def _load_processed_ids(checkpoint_path: Path) -> set[str]:
    if not checkpoint_path.exists():
        return set()
    try:
        df = pd.read_parquet(checkpoint_path, columns=["question_id"])
        return set(df["question_id"].unique())
    except Exception:
        return set()
