import os

import httpx
from openai import OpenAI

openrouter = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_TOKEN"),
    timeout=httpx.Timeout(connect=15.0, read=180.0, write=15.0, pool=15.0),
)
