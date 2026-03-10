import re

from loguru import logger as _logger


class ArgLogger:
    def __getattr__(self, level):
        def log(msg, *args, **kwargs):
            # Count how many {} placeholders (excluding escaped {{}})
            placeholders = len(re.findall(r"(?<!\{)\{[^{]*?\}(?!\})", msg))
            used = args[:placeholders]
            unused = args[placeholders:]

            formatted = msg.format(*used) if used else msg
            if unused:
                formatted += " | " + " | ".join(str(a) for a in unused)

            getattr(_logger.opt(depth=1), level)(formatted, **kwargs)

        return log


logger = ArgLogger()
