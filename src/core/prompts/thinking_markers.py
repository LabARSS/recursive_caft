# Thinking/reasoning markers used to wrap the reasoning chain
# during training with distilled reasoning data.
#
# Format in generated text:
#   <think>reasoning content here</think>[[answer]]
#
# These markers are model-agnostic. If you need model-specific
# thinking tokens (e.g. Phi-4's native <|thinking|>), override
# these values or add a mapping here.

THINKING_START = "<think>"
THINKING_END = "</think>"