JUDGE_SYSTEM_PROMPT = """### Project Context
We are creating a high-quality synthetic dataset for training small models.

1. "Original Reasoning" (branch A) is our "gold standard." It represents the raw, correct chain of reasoning that led to the correct answer.

2. "Metadata-stripped Reasoning with Correct Answer Explanation" (branch B) is a version that explains the correct answer instead of trying to "guess" the correct answer by generating a reasoning trace.

### Problem

When we generated branch B, we saw that the reasoning chain includes statements that reveal that this reasoning chain is generated when the answer is given beforehand. It is going ot ruin th etraining objective. We asked an LLM to clean the generated chain for branch B.

### Your goal

Rate the 'Cleaned reasoning' on 2 criteria (1-10 scale):

1. logical_preservation:
   - 10: All logical steps, calculations, and crucial evidence are preserved in the exact original sequence. The deductive flow (Step A -> Step B -> Conclusion) remains unchanged.
   - 1: Critical steps are missing, or the reasoning order is distorted (e.g., stating the final answer before the logic/derivations, whereas the original built up to it).

2. noise_removal:
   - 10: Perfectly stripped of conversational fillers ("Let's think", "We need to", "Step 1"), meta-language, and self-corrections. Contains ONLY declarative facts and deductions.
   - 1: Still contains significant conversational filler, headers like "Why this is wrong", or meta-commentary.


The output for branch A is provided as the golden reference training signal

### Examples

Input:
Original: "We need to select correct option letter. The second law of thermodynamics says that heat does not spontaneously flow from a colder body to a hotter body without external energy. Option j: 'cold to hot without external energy'. That matches. So answer is j."
Cleaned: "The second law states heat flows spontaneously from hot to cold; it cannot flow from cold to hot without external work. Option J says 'cold to hot without external energy' – this is contrary to the law. Therefore, J is correct."
Output:
{"logical_preservation": 10, "noise_removal": 10}

Input:
Original: "We need to answer multiple choice. The question: In personnel selection, the selection ratio is defined as the ... Options: a) ratio of successful..., b) ratio of number... Let's think. Usually selection ratio is applicants hired vs available."
Cleaned: "D is correct. A, B, C are wrong."
Output:
{"logical_preservation": 1, "noise_removal": 10}

Input:
Original: "We need to compute equivalent capacitance of series: 1/C_eq = sum(1/C_i). Here n=5, C=5. So 1/C_eq = 5/5 = 1. So answer is h."
Cleaned: "Why other options are wrong and why H (1 µF) is correct: series formula C_eq = 1/sum. For five identical 5 µF in series: C_eq = 1 µF. Rule out other options."
Output:
{"logical_preservation": 3, "noise_removal": 6}

### Task

Return ONLY a JSON object.
Format: {"logical_preservation": X, "noise_removal": X}
"""


JUDGE_USER_TEMPLATE = """Original reasoning:
{original}

Cleaned reasoning:
{cleaned}

Evaluate the cleaning quality."""


FEWSHOT_EXAMPLE = {
    "original": """Let me analyze this question step by step. We need to find the area of a circle with radius 5. 
Note that the formula for circle area is A = πr². Let's calculate: r = 5, so r² = 25. 
Therefore A = π × 25 = 25π ≈ 78.54. Now let me check the options. Option A says 78.5, which matches our calculation.""",
    "cleaned": """We need to find the area of a circle with radius 5.
The formula for circle area is A = πr². r = 5, so r² = 25.
Therefore A = π × 25 = 25π ≈ 78.54. Option A says 78.5, which matches our calculation.""",
    "rating": '{"logical_preservation": 5, "noise_removal": 5}'
}


def create_judge_prompt(original: str, cleaned: str, examples: list = None) -> list:
    messages = [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}]
    
    if examples:
        for ex in examples:
            messages.append({"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                original=ex["original"], cleaned=ex["cleaned"]
            )})
            messages.append({"role": "assistant", "content": ex["rating"]})
    
    messages.append({"role": "user", "content": JUDGE_USER_TEMPLATE.format(
        original=original, cleaned=cleaned
    )})
    
    return messages
