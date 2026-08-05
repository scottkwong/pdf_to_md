"""
Modular LLM provider abstraction for OpenRouter and direct API providers.

This module provides a unified interface for interacting with multiple LLM
providers including OpenRouter, OpenAI, Anthropic, and Google. It can be
easily copied to other projects for reuse.

All providers support vision processing with base64-encoded images.
"""
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI
import google.genai as genai

# Load environment variables
load_dotenv()


@dataclass
class TokenUsage:
    """Token usage from an LLM API call."""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None  # Set by provider when API returns actual cost


@dataclass
class VisionResult:
    """Result from a vision processing call."""
    text: str
    usage: TokenUsage = field(default_factory=TokenUsage)


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def process_vision(
        self,
        image_base64: str,
        prompt: str,
        prior_text: Optional[str] = None,
        model: str = "",
        max_tokens: int = 4096,
    ) -> VisionResult:
        """
        Process an image with vision model.

        Args:
            image_base64: Base64-encoded image string.
            prompt: Text prompt for the model.
            prior_text: Optional prior text for context.
            model: Model identifier (may be ignored if provider has default).
            max_tokens: Maximum tokens in response.

        Returns:
            VisionResult with text and token usage.
        """
        pass


class OpenRouterProvider(BaseProvider):
    """Provider for OpenRouter API (supports all models)."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenRouter provider.

        Args:
            api_key: OpenRouter API key. If None, reads from OPENROUTER_API_KEY.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    def process_vision(
        self,
        image_base64: str,
        prompt: str,
        prior_text: Optional[str] = None,
        model: str = "",
        max_tokens: int = 4096,
    ) -> VisionResult:
        """
        Process image using OpenRouter API.

        Args:
            image_base64: Base64-encoded image string.
            prompt: Text prompt for the model.
            prior_text: Optional prior text for context.
            model: Model identifier (e.g., "openai/gpt-5.2").
            max_tokens: Maximum tokens in response.

        Returns:
            VisionResult with text and token usage.
        """
        if not model:
            raise ValueError("Model must be specified for OpenRouter")

        full_prompt = prompt
        if prior_text:
            full_prompt = f"{prompt}\n\n<prior_text>\n{prior_text}\n</prior_text>"

        # Newer models use max_completion_tokens, older use max_tokens
        create_kwargs = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
        }
        # Try max_completion_tokens first (for newer models), fallback to max_tokens
        try:
            create_kwargs["max_completion_tokens"] = max_tokens
            response = self.client.chat.completions.create(**create_kwargs)
        except Exception:
            # Fallback to max_tokens for older models
            create_kwargs.pop("max_completion_tokens", None)
            create_kwargs["max_tokens"] = max_tokens
            response = self.client.chat.completions.create(**create_kwargs)

        usage = TokenUsage()
        if response.usage:
            usage.input_tokens = response.usage.prompt_tokens or 0
            usage.output_tokens = response.usage.completion_tokens or 0
            # OpenRouter may provide actual cost
            cost = getattr(response.usage, "cost", None)
            if cost is not None:
                usage.cost_usd = float(cost)

        return VisionResult(
            text=response.choices[0].message.content,
            usage=usage,
        )


class FireworksProvider(BaseProvider):
    """Provider for Fireworks AI (OpenAI-compatible, open-weight vision models).

    Fireworks serves open-weight and licensed models (Qwen-VL, etc.) behind an
    OpenAI-compatible API, so this mirrors OpenRouterProvider: same chat schema
    and base64 data-URL images, just a different base URL and key. It does not
    provide access to proprietary OpenAI/Anthropic/Google models.
    """

    BASE_URL = "https://api.fireworks.ai/inference/v1"

    # Reasoning models (e.g. qwen3p7-plus) spend completion budget on thinking
    # before the answer, so dense pages truncate mid-thought at 4096. Request
    # at least this much; only generated tokens are billed.
    REASONING_TOKEN_HEADROOM = 16384

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Fireworks provider.

        Args:
            api_key: Fireworks API key. If None, reads from FIREWORKS_API_KEY.
        """
        self.api_key = api_key or os.getenv("FIREWORKS_API_KEY")
        if not self.api_key:
            raise ValueError("FIREWORKS_API_KEY not found in environment")
        self.client = OpenAI(api_key=self.api_key, base_url=self.BASE_URL)

    def process_vision(
        self,
        image_base64: str,
        prompt: str,
        prior_text: Optional[str] = None,
        model: str = "",
        max_tokens: int = 4096,
    ) -> VisionResult:
        """
        Process image using the Fireworks OpenAI-compatible API.

        Args:
            image_base64: Base64-encoded image string.
            prompt: Text prompt for the model.
            prior_text: Optional prior text for context.
            model: Fireworks model path (e.g.
                "accounts/fireworks/models/qwen3p7-plus").
            max_tokens: Maximum tokens in response.

        Returns:
            VisionResult with text and token usage.
        """
        if not model:
            raise ValueError("Model must be specified for Fireworks")

        full_prompt = prompt
        if prior_text:
            full_prompt = f"{prompt}\n\n<prior_text>\n{prior_text}\n</prior_text>"

        effective_max_tokens = max(max_tokens, self.REASONING_TOKEN_HEADROOM)
        response = self.client.chat.completions.create(
            model=model,
            max_tokens=effective_max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
        )

        choice = response.choices[0]
        # When a reasoning model exhausts the budget mid-thought, Fireworks
        # returns the partial thinking in `content` (reasoning_content stays
        # None). Raise so the caller's retry gets another sample instead of
        # writing thinking text into the markdown output.
        if (
            choice.finish_reason == "length"
            and getattr(choice.message, "reasoning_content", None) is None
        ):
            raise ValueError(
                "Fireworks response truncated during reasoning "
                f"(finish_reason=length at {effective_max_tokens} tokens); "
                "no answer was produced."
            )

        usage = TokenUsage()
        if response.usage:
            usage.input_tokens = response.usage.prompt_tokens or 0
            usage.output_tokens = response.usage.completion_tokens or 0

        return VisionResult(
            text=choice.message.content,
            usage=usage,
        )


class OpenAIProvider(BaseProvider):
    """Provider for direct OpenAI API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        self.client = OpenAI(api_key=self.api_key)

    def process_vision(
        self,
        image_base64: str,
        prompt: str,
        prior_text: Optional[str] = None,
        model: str = "gpt-4o",
        max_tokens: int = 4096,
    ) -> VisionResult:
        """
        Process image using OpenAI API.

        Args:
            image_base64: Base64-encoded image string.
            prompt: Text prompt for the model.
            prior_text: Optional prior text for context.
            model: Model identifier (default: "gpt-4o").
            max_tokens: Maximum tokens in response.

        Returns:
            VisionResult with text and token usage.
        """
        full_prompt = prompt
        if prior_text:
            full_prompt = f"{prompt}\n\n<prior_text>\n{prior_text}\n</prior_text>"

        # Newer models use max_completion_tokens, older use max_tokens
        create_kwargs = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
        }
        # Try max_completion_tokens first (for newer models), fallback to max_tokens
        try:
            create_kwargs["max_completion_tokens"] = max_tokens
            response = self.client.chat.completions.create(**create_kwargs)
        except Exception:
            # Fallback to max_tokens for older models
            create_kwargs.pop("max_completion_tokens", None)
            create_kwargs["max_tokens"] = max_tokens
            response = self.client.chat.completions.create(**create_kwargs)

        usage = TokenUsage()
        if response.usage:
            usage.input_tokens = response.usage.prompt_tokens or 0
            usage.output_tokens = response.usage.completion_tokens or 0

        return VisionResult(
            text=response.choices[0].message.content,
            usage=usage,
        )


class AnthropicProvider(BaseProvider):
    """Provider for direct Anthropic API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        self.client = Anthropic(api_key=self.api_key)

    def process_vision(
        self,
        image_base64: str,
        prompt: str,
        prior_text: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 4096,
    ) -> VisionResult:
        """
        Process image using Anthropic API.

        Args:
            image_base64: Base64-encoded image string (no data URL prefix).
            prompt: Text prompt for the model.
            prior_text: Optional prior text for context.
            model: Model identifier (default: "claude-3-5-sonnet-20241022").
            max_tokens: Maximum tokens in response.

        Returns:
            VisionResult with text and token usage.
        """
        full_prompt = prompt
        if prior_text:
            full_prompt = f"{prompt}\n\n<prior_text>\n{prior_text}\n</prior_text>"

        # Anthropic uses base64 string directly (no data URL prefix)
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64,
                            },
                        },
                    ],
                }
            ],
        )

        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        return VisionResult(
            text=response.content[0].text,
            usage=usage,
        )


class GoogleProvider(BaseProvider):
    """Provider for direct Google/Gemini API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Google provider.

        Args:
            api_key: Google API key. If None, reads from GOOGLE_API_KEY or
                GEMINI_API_KEY.

        Raises:
            ValueError: If API key is not found.
        """
        self.api_key = (
            api_key
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY or GEMINI_API_KEY not found in environment"
            )
        # Initialize client with API key for new google.genai API
        # The new SDK uses Client class for API key management
        self.client = genai.Client(api_key=self.api_key)

    def process_vision(
        self,
        image_base64: str,
        prompt: str,
        prior_text: Optional[str] = None,
        model: str = "gemini-3-pro",
        max_tokens: int = 4096,
    ) -> VisionResult:
        """
        Process image using Google Gemini API.

        Args:
            image_base64: Base64-encoded image string.
            prompt: Text prompt for the model.
            prior_text: Optional prior text for context.
            model: Model identifier (default: "gemini-3-pro").
            max_tokens: Maximum tokens in response (may be ignored by API).

        Returns:
            VisionResult with text and token usage.
        """
        full_prompt = prompt
        if prior_text:
            full_prompt = f"{prompt}\n\n<prior_text>\n{prior_text}\n</prior_text>"

        # Try to use the specified model, fallback to available models.
        # Fallbacks use current stable models (gemini-1.5-* deprecated).
        # google.genai SDK: Client.models.generate_content with contents dict.
        model_to_try = model
        last_error: Optional[Exception] = None
        fallback_models = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-2.5-flash-lite",
        ]
        for attempt_model in [model_to_try] + fallback_models:
            try:
                response = self.client.models.generate_content(
                    model=attempt_model,
                    contents=[
                        {"role": "user", "parts": [
                            {"text": full_prompt},
                            {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}}
                        ]}
                    ],
                )
                last_error = None
                break
            except Exception as e:
                last_error = e
                continue
        if last_error is not None:
            raise ValueError(
                f"Could not use any available Google model. Last error: {last_error}"
            ) from last_error

        # Extract usage metadata
        usage = TokenUsage()
        usage_meta = getattr(response, "usage_metadata", None)
        if usage_meta:
            usage.input_tokens = getattr(usage_meta, "prompt_token_count", 0) or 0
            usage.output_tokens = getattr(usage_meta, "candidates_token_count", 0) or 0

        # Extract text from response
        if hasattr(response, "text"):
            text = response.text
        elif hasattr(response, "candidates") and response.candidates:
            text = response.candidates[0].content.parts[0].text
        else:
            text = str(response)

        return VisionResult(text=text, usage=usage)


def load_models_config() -> Dict:
    """
    Load and parse models.json configuration file.

    Returns:
        Dictionary of model configurations.

    Raises:
        FileNotFoundError: If models.json is not found.
        json.JSONDecodeError: If models.json is invalid JSON.
    """
    from models_config import load_models_config as _load
    return _load()


def validate_openrouter_key(api_key: str) -> bool:
    """
    Validate OpenRouter API key by pinging the keys API.

    Args:
        api_key: OpenRouter API key to validate.

    Returns:
        True if key is valid (200 response), False otherwise.
    """
    import requests

    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5,
        )
        return response.status_code == 200
    except Exception:
        return False


def get_available_providers(validate_keys: bool = False) -> Dict[str, bool]:
    """
    Check which API keys are available in the environment.

    Args:
        validate_keys: If True, validate keys with actual API requests.
            This is slower but more accurate.

    Returns:
        Dictionary mapping provider names to availability (True/False).
    """
    providers = {
        "openrouter": bool(os.getenv("OPENROUTER_API_KEY")),
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "google": bool(
            os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        ),
        "fireworks": bool(os.getenv("FIREWORKS_API_KEY")),
    }

    # If validate_keys is True, actually test the OpenRouter key
    if validate_keys and providers["openrouter"]:
        api_key = os.getenv("OPENROUTER_API_KEY")
        providers["openrouter"] = validate_openrouter_key(api_key)

    return providers


def get_available_models_for_keys() -> List[Dict]:
    """
    Get list of models available based on detected API keys.

    Returns:
        List of model dictionaries with keys: name, display_name, provider.
    """
    available = get_available_providers()
    models_config = load_models_config()
    available_models = []

    def add_direct_model(
        name: str, config: Dict, provider_name: str
    ) -> None:
        if available.get(provider_name):
            available_models.append(
                {
                    "name": name,
                    "display_name": config.get("direct_id", ""),
                    "provider": provider_name,
                }
            )

    def add_openrouter_model(name: str, config: Dict) -> bool:
        openrouter_id = config.get("openrouter_id")
        if not openrouter_id:
            return False
        available_models.append(
            {
                "name": name,
                "display_name": openrouter_id,
                "provider": "openrouter",
            }
        )
        return True

    # If OpenRouter is available, use models with OpenRouter IDs first
    if available["openrouter"]:
        for model_name, model_config in models_config.items():
            if add_openrouter_model(model_name, model_config):
                continue
            add_direct_model(
                model_name, model_config, model_config["provider"]
            )
    else:
        # Otherwise, check direct provider APIs
        for model_name, model_config in models_config.items():
            add_direct_model(
                model_name, model_config, model_config["provider"]
            )

    return available_models


def prompt_for_fallback(
    available_models: List[Dict], requested_model: str
) -> Optional[str]:
    """
    Display interactive menu for model selection (numbered choices).

    Args:
        available_models: List of available model dictionaries.
        requested_model: The originally requested model name.

    Returns:
        Selected model name, or None if user chooses to exit.
    """
    print(f"\nRequested model '{requested_model}' is not available.")
    print("Available models:")
    for i, model in enumerate(available_models, 1):
        provider_name = model["provider"].title()
        print(f"  {i}) {model['display_name']} (via {provider_name} API)")

    print(f"  {len(available_models) + 1}) Exit")

    while True:
        try:
            choice = input(f"\nSelect option [1-{len(available_models) + 1}]: ")
            choice_num = int(choice)
            if 1 <= choice_num <= len(available_models):
                return available_models[choice_num - 1]["name"]
            elif choice_num == len(available_models) + 1:
                return None
            else:
                print(
                    f"Invalid choice. Please enter a number between 1 and "
                    f"{len(available_models) + 1}."
                )
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nExiting...")
            return None


def resolve_model(
    model_name: str, prefer_openrouter: bool = True
) -> Tuple[str, BaseProvider]:
    """
    Resolve model name to appropriate provider instance.

    Args:
        model_name: Model identifier (e.g., "openai", "openai-gpt4o").
        prefer_openrouter: If True, prefer OpenRouter when available.

    Returns:
        Tuple of (model_id, provider_instance).

    Raises:
        ValueError: If model not found or no API keys available.
    """
    models_config = load_models_config()

    if model_name not in models_config:
        raise ValueError(f"Model '{model_name}' not found in models.json")

    model_config = models_config[model_name]
    provider_name = model_config["provider"]
    available = get_available_providers()

    # Determine which provider to use
    use_openrouter = False
    openrouter_id = model_config.get("openrouter_id")
    if prefer_openrouter and available["openrouter"] and openrouter_id:
        use_openrouter = True
        model_id = openrouter_id
        provider = OpenRouterProvider()
    elif provider_name == "openai" and available["openai"]:
        model_id = model_config["direct_id"]
        provider = OpenAIProvider()
    elif provider_name == "anthropic" and available["anthropic"]:
        model_id = model_config["direct_id"]
        provider = AnthropicProvider()
    elif provider_name == "google" and available["google"]:
        model_id = model_config["direct_id"]
        provider = GoogleProvider()
    elif provider_name == "fireworks" and available["fireworks"]:
        model_id = model_config["direct_id"]
        provider = FireworksProvider()
    else:
        # No API key available - try fallback
        available_models = get_available_models_for_keys()
        if not available_models:
            raise ValueError(
                f"No API keys available for model '{model_name}'. "
                "Please set at least one API key in your .env file."
            )

        # Show interactive menu
        selected_model = prompt_for_fallback(available_models, model_name)
        if selected_model is None:
            raise ValueError("User chose to exit. No model selected.")

        # Recursively resolve the selected model
        return resolve_model(selected_model, prefer_openrouter)

    return model_id, provider


def create_provider(
    model: str, prefer_openrouter: bool = True
) -> Tuple[BaseProvider, str]:
    """
    Create provider instance for the specified model.

    Args:
        model: Model identifier from models.json.
        prefer_openrouter: If True, prefer OpenRouter when available.

    Returns:
        Tuple of (provider_instance, model_id).

    Raises:
        ValueError: If model not found or no API keys available.
    """
    model_id, provider = resolve_model(model, prefer_openrouter)
    return provider, model_id

