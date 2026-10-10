import sys
sys.path.insert(0, 'src')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.entropy_dynamics.reasoning_loader import load_teacher_reasoning
from core.entropy_dynamics.prompt_builder import build_prefixed_prompts
from core.entropy_dynamics.config import InferenceMode
from core.complexity_estimation.entropy.logit_entropy import compute_entropy_from_logits

TEACHER_PATH = "data/out/distillation/mmlu_synth_gptoss_b_t0_8.parquet"
STUDENT_ID   = "Qwen/Qwen2.5-3B"
MODE         = InferenceMode.FORCED
DATASET_TYPE = "mmlu"
WINDOW_SIZE  = 32
MAX_SAMPLES  = 5   

tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(STUDENT_ID, dtype=torch.bfloat16, device_map="auto")
model.eval()

samples = load_teacher_reasoning(TEACHER_PATH, tokenizer=tokenizer, min_thinking_tokens=WINDOW_SIZE)
samples = samples[:MAX_SAMPLES]
print(f"Loaded {len(samples)} samples\n")

SEP = "="*80

for sample in samples:
    sample.thinking_token_ids = tokenizer.encode(sample.thinking_text, add_special_tokens=False)

    prefixed_prompts = build_prefixed_prompts(
        sample=sample,
        tokenizer=tokenizer,
        window_size=WINDOW_SIZE,
        mode=MODE,
        dataset_type=DATASET_TYPE,
    )

    print(SEP)
    print(f"QUESTION ID : {sample.question_id}")
    print(f"QUESTION    : {sample.question}")
    print(f"OPTIONS     : {sample.options}")
    print(f"GOLD ANSWER : {sample.gold_answer}")
    print(f"TOTAL REASONING TOKENS (teacher): {len(sample.thinking_token_ids)}")
    print()
    print("FULL TEACHER REASONING TEXT")
    print(sample.thinking_text)
    print()
    print(f"Total k steps: {len(prefixed_prompts)}  (window={WINDOW_SIZE} tokens each)")
    print(SEP)

    for prompt in prefixed_prompts:
        print(f"\n{'─'*60}")
        print(f"  k={prompt.k} | reasoning tokens fed to student: {prompt.num_reasoning_tokens} / {prompt.total_reasoning_tokens}")

        decoded_prompt = tokenizer.decode(prompt.input_ids, skip_special_tokens=False)
        print(f"\n FULL PROMPT TO STUDENT")
        print(decoded_prompt)

        input_ids = torch.tensor([prompt.input_ids], device=model.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        logits = outputs.scores[0][0]
        entropy = compute_entropy_from_logits(logits).item()
        generated_id = outputs.sequences[0, input_ids.shape[1]].item()
        student_answer = tokenizer.decode([generated_id]).strip().lower()
        correct = (student_answer == sample.gold_answer)

        probs = torch.softmax(logits, dim=-1)
        top5 = torch.topk(probs, 5)
        top5_tokens = [(tokenizer.decode([idx.item()]).strip(), prob.item())
                       for idx, prob in zip(top5.indices, top5.values)]

        print(f"\n  ── STUDENT OUTPUT ──")
        print(f"  Generated token : '{tokenizer.decode([generated_id])}'  (id={generated_id})")
        print(f"  Parsed answer   : '{student_answer}'")
        print(f"  Gold answer     : '{sample.gold_answer}'")
        print(f"  Correct         : {correct}")
        print(f"  Entropy         : {entropy:.4f}")
        print(f"  Top-5 candidates: {top5_tokens}")

    print()