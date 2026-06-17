import os
from langchain.chat_models import init_chat_model

def get_llm(temperature=0.7, max_retries=2, **kwargs):
    """
    Generic LLM initialization that reads from environment variables.
    This avoids needing to change code when switching providers.
    
    Expected environment variables:
    - LLM_MODEL: The model name (e.g. 'llama-3.3-70b-versatile', 'gpt-4o', 'claude-3-5-sonnet-20240620')
    - LLM_PROVIDER: The provider name (e.g. 'groq', 'openai', 'anthropic')
    """
    model = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
    provider = os.environ.get("LLM_PROVIDER", "groq")
    
    return init_chat_model(
        model=model,
        model_provider=provider,
        temperature=temperature,
        max_retries=max_retries,
        **kwargs
    )
