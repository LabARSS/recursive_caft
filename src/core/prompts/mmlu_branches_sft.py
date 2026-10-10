from typing import Optional
import ast

from core.prompts.mmlu_option_ids import option_ids


def branch_a_sys_prompt(subject: Optional[str] = None) -> str:
    """Branch A: Q + options -> answer + CoT"""
    if subject:
        base = f"The following are multiple choice questions about {subject}."
    else:
        base = "The following are multiple choice questions."
    return base + " Provide reasoning and the correct answer."


def branch_a_user_prompt(question: str, options: list[str]) -> str:
    """Format question and options for Branch A"""
    opts = "\n".join([f"{oid}. {text}".strip() for oid, text in zip(option_ids, options)])
    return f"Question: {question.strip()}\n\nOptions:\n{opts}\n\nProvide reasoning and answer."


def branch_b_sys_prompt(subject: Optional[str] = None) -> str:
    """Branch B: Q + options + gold -> explanation why correct"""
    if subject:
        base = f"The following are multiple choice questions about {subject}."
    else:
        base = "The following are multiple choice questions."
    return base + " Given the correct option letter, explain why it is correct."


def branch_b_user_prompt(question: str, options: list[str], gold_letter: str) -> str:
    """Format question, options, and gold answer for Branch B"""
    opts = "\n".join([f"{oid}. {text}".strip() for oid, text in zip(option_ids, options)])
    return (
        f"Question: {question.strip()}\n\n"
        f"Options:\n{opts}\n\n"
        f"Correct answer: {gold_letter}\n\n"
        f"Explain why this is correct."
    )


def branch_c_sys_prompt(subject: Optional[str] = None) -> str:
    """Branch C: Q + options + wrong answer -> error analysis"""
    if subject:
        base = f"The following are multiple choice questions about {subject}."
    else:
        base = "The following are multiple choice questions."
    return base + " Analyze the error in the previous answer and explain the correct reasoning."


def branch_c_user_prompt(question: str, options: list[str], prev_answer: str, gold_answer: str) -> str:
    """Format question, options, previous wrong answer, and gold for Branch C"""
    opts = "\n".join([f"{oid}. {text}".strip() for oid, text in zip(option_ids, options)])
    return (
        f"Question: {question.strip()}\n\n"
        f"Options:\n{opts}\n\n"
        f"Previous answer: {prev_answer}\n"
        f"Correct answer: {gold_answer}\n\n"
        f"Analyze the error and explain the correct reasoning."
    )
