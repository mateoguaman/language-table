"""Unified model API.

Usage:
    from opro.models import call_model

    resp = await call_model(
        prompt="Hello",
        model_id="gemini-3-flash-preview",   # or "gpt-5.5", "claude-sonnet-4-5"
        thinking_level="MEDIUM",
    )
    print(resp.text)

Supported providers (detected by model_id prefix):
    gemini-*    → Google Gemini via google-genai (GOOGLE_API_KEY)
    gpt-*       → OpenAI GPT via openai SDK (OPENAI_API_KEY)
    claude-*    → Anthropic Claude via anthropic SDK (ANTHROPIC_API_KEY)
    deepseek-*  → DeepSeek via OpenAI-compatible SDK (DEEPSEEK_API_KEY)
                  Models: deepseek-v4-flash, deepseek-v4-pro
"""

from opro.models.base import call_model, ModelResponse

__all__ = ["call_model", "ModelResponse"]
