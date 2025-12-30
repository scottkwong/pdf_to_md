#!/bin/bash
# Create .env file from shell environment variables

ENV_FILE=".env"

echo "Creating $ENV_FILE file..."

cat > "$ENV_FILE" << ENVFILE
# API Keys for LLM Providers
# Auto-generated from shell environment

# OpenRouter API Key (supports all models)
OPENROUTER_API_KEY=${OPENROUTER_API_KEY}

# Direct Provider API Keys
OPENAI_API_KEY=${OPENAI_API_KEY}
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}

# Google/Gemini API Key
GEMINI_API_KEY=${GEMINI_API_KEY}
ENVFILE

echo "✓ Created $ENV_FILE with your API keys"
echo ""
echo "Keys found:"
[ -n "$OPENROUTER_API_KEY" ] && echo "  ✓ OPENROUTER_API_KEY" || echo "  ✗ OPENROUTER_API_KEY (missing)"
[ -n "$OPENAI_API_KEY" ] && echo "  ✓ OPENAI_API_KEY" || echo "  ✗ OPENAI_API_KEY (missing)"
[ -n "$ANTHROPIC_API_KEY" ] && echo "  ✓ ANTHROPIC_API_KEY" || echo "  ✗ ANTHROPIC_API_KEY (missing)"
[ -n "$GEMINI_API_KEY" ] && echo "  ✓ GEMINI_API_KEY" || echo "  ✗ GEMINI_API_KEY (missing)"
