"""AI Copilot Agent using the Anthropic API directly (no langchain)"""

from ..utils.config import settings
import logging
import json

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "claude-sonnet-5"
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are a helpful AI assistant for a data analytics platform called Decisera.
You help users understand their datasets, models, and analytics results.
Provide clear, concise, and accurate responses.

Only use the workspace data given to you to answer questions about datasets or
models — never invent dataset or model names, metrics, or dates that aren't
present in it. If something the user asks about isn't in this data, say so
plainly instead of guessing."""


def _normalize_messages(messages: list) -> list:
    """Coerce a message list into Anthropic's required strict alternation.

    History assembled from a chat UI can end up with two consecutive
    "user" turns — e.g. a prior turn errored out and was dropped, or the
    client sends slightly malformed history — which the API rejects
    outright. Merge consecutive same-role turns instead of erroring.
    """
    normalized: list = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role not in ("user", "assistant") or not content:
            continue
        if normalized and normalized[-1]["role"] == role:
            normalized[-1]["content"] += f"\n\n{content}"
        else:
            normalized.append({"role": role, "content": content})
    # Anthropic requires the conversation to start with a "user" turn.
    while normalized and normalized[0]["role"] != "user":
        normalized.pop(0)
    return normalized


def _build_workspace_context() -> str:
    """Summarize the user's actual datasets and models for grounding.

    Without this, the LLM has no knowledge of what's really in the
    workspace and will confidently invent plausible-sounding models and
    datasets instead of reporting real ones.
    """
    try:
        from .tools import copilot_tools

        datasets = copilot_tools.list_available_datasets()
        models = copilot_tools.list_available_models()

        if not datasets and not models:
            return "The workspace currently has no datasets or trained models."

        parts = []
        if datasets:
            parts.append("Datasets:\n" + json.dumps(datasets, default=str, indent=2))
        else:
            parts.append("Datasets: none uploaded yet.")

        if models:
            parts.append("Trained models:\n" + json.dumps(models, default=str, indent=2))
        else:
            parts.append("Trained models: none trained yet.")

        return "\n\n".join(parts)
    except Exception as e:
        logger.warning(f"Failed to build workspace context for copilot: {e}")
        return "Workspace data is currently unavailable."


class AICopilotAgent:
    def __init__(self):
        """Initialize the AI Copilot with Anthropic Claude."""
        self.client = None

        try:
            # Check if API key is available
            if not settings.anthropic_api_key:
                logger.warning("Anthropic API key is not set in environment")
                return

            # Lazy import to avoid blocking app startup with heavy module load
            import anthropic

            self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

            logger.info("✓ AI Copilot client created (will be tested on first use)")

        except Exception as e:
            logger.error(f"Failed to create copilot client: {type(e).__name__}: {e}")
            self.client = None

    def query(self, user_input: str, history: list = None) -> str:
        """
        Process user query and return response.

        Args:
            user_input: The user's question
            history: Prior turns of this conversation, oldest first, as
                {"role": "user"|"assistant", "content": str} dicts — lets
                follow-up questions ("what's its accuracy?") resolve against
                what was actually said earlier in the chat.

        Returns:
            AI-generated response string
        """
        try:
            # Check if API key is available
            if not settings.anthropic_api_key:
                return "AI Copilot requires an Anthropic API key to be configured. Please set ANTHROPIC_API_KEY in your environment."

            # Lazy import (deferred from module level to avoid blocking startup)
            import anthropic

            # Create client fresh each time to avoid caching issues
            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

            # Ground the model in the user's actual workspace data so it
            # reports real datasets/models instead of inventing plausible
            # ones. Only the latest turn carries this — it can go stale
            # across a long conversation, but repeating it on every prior
            # turn would waste tokens for no benefit.
            workspace_context = _build_workspace_context()

            user_message = f"""Workspace data:
{workspace_context}

User question: {user_input}"""

            messages = _normalize_messages(
                list(history or []) + [{"role": "user", "content": user_message}]
            )

            # Get response from Claude
            response = client.messages.create(
                model=DEFAULT_MODEL_NAME,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
            return "".join(
                block.text for block in response.content if block.type == "text"
            )

        except Exception as e:
            # Log the full error for debugging
            logger.error(f"Copilot error: {type(e).__name__}: {str(e)}")

            error_msg = str(e).lower()
            error_type = type(e).__name__

            # Check for specific Anthropic API errors
            if "notfounderror" in error_type.lower() or "404" in error_msg:
                logger.error(
                    f"Claude model not found or configuration error: {error_type}: {str(e)}"
                )
                return (
                    f"AI model configuration error: The requested Claude model ({DEFAULT_MODEL_NAME}) "
                    "was not found or is deprecated. Please verify the configured model name."
                )
            elif "ratelimiterror" in error_type.lower() or "rate limit" in error_msg:
                logger.warning("Anthropic API rate limit / quota exceeded")
                return "The AI service quota has been exceeded. Please try again later or check your usage at https://console.anthropic.com/settings/usage"
            elif (
                "authenticationerror" in error_type.lower()
                or "api_key" in error_msg
                or "authentication" in error_msg
                or "invalid x-api-key" in error_msg
            ):
                return "There was an authentication issue with the AI service. Please verify your Anthropic API key is valid."
            elif "permissiondeniederror" in error_type.lower() or "permission" in error_msg:
                return "Permission denied. Please check that your API key has access to the requested Claude model."
            elif "timeout" in error_msg or "apitimeouterror" in error_type.lower():
                return (
                    "The request timed out. Please try again with a simpler question."
                )
            elif "safety" in error_msg or "blocked" in error_msg:
                return "I cannot provide a response to that question due to content safety policies. Please try rephrasing your question."
            else:
                # Return a generic error with the actual error type for debugging
                logger.error(f"Unhandled error type: {error_type}")
                return f"I encountered an error: {error_type}. Please try again or contact support if this persists."


# Lazy singleton initialization - ensures settings are loaded first
_copilot_agent_instance = None


def get_copilot_agent() -> AICopilotAgent:
    """Get or create the copilot agent singleton (lazy initialization)"""
    global _copilot_agent_instance

    if _copilot_agent_instance is None:
        try:
            logger.info("Initializing AI Copilot agent (lazy load)...")
            _copilot_agent_instance = AICopilotAgent()
            logger.info("✓ Copilot agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to create copilot agent: {e}")
            # Create a blank instance as fallback
            _copilot_agent_instance = AICopilotAgent()

    return _copilot_agent_instance


# Create a proxy class that uses lazy loading
class CopilotAgentProxy:
    """Proxy class that lazily initializes the copilot agent"""

    def query(self, user_input: str, history: list = None) -> str:
        """Forward query to the lazily-initialized agent"""
        return get_copilot_agent().query(user_input, history)


# Export the proxy as copilot_agent for backward compatibility
copilot_agent = CopilotAgentProxy()
