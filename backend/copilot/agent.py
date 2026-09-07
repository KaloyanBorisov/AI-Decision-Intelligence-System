"""AI Copilot Agent using Google Generative AI directly (no langchain)"""

from ..utils.config import settings
import logging
import json

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "gemini-3.6-flash"


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
        """Initialize the AI Copilot with Google Gemini."""
        self.model = None

        try:
            # Check if API key is available
            if not settings.google_api_key:
                logger.warning("Google API key is not set in environment")
                return

            # Lazy import to avoid blocking app startup with heavy module load
            import google.generativeai as genai

            # Configure Google Generative AI
            genai.configure(api_key=settings.google_api_key)

            # Create the model - gemini-3.6-flash
            # Don't test it during init - just create it
            self.model = genai.GenerativeModel(DEFAULT_MODEL_NAME)

            logger.info("✓ AI Copilot model created (will be tested on first use)")

        except Exception as e:
            logger.error(f"Failed to create copilot model: {type(e).__name__}: {e}")
            self.model = None

    def query(self, user_input: str) -> str:
        """
        Process user query and return response.

        Args:
            user_input: The user's question

        Returns:
            AI-generated response string
        """
        try:
            # Check if API key is available
            if not settings.google_api_key:
                return "AI Copilot requires a Google API key to be configured. Please set GOOGLE_API_KEY in your environment."

            # Lazy import (deferred from module level to avoid blocking startup)
            import google.generativeai as genai

            # Reconfigure API on each request to avoid caching issues
            genai.configure(api_key=settings.google_api_key)

            # Create model fresh each time
            model = genai.GenerativeModel(DEFAULT_MODEL_NAME)

            # Ground the model in the user's actual workspace data so it
            # reports real datasets/models instead of inventing plausible
            # ones.
            workspace_context = _build_workspace_context()

            prompt = f"""You are a helpful AI assistant for a data analytics platform called Decisera.
You help users understand their datasets, models, and analytics results.
Provide clear, concise, and accurate responses.

Only use the workspace data below to answer questions about datasets or
models — never invent dataset or model names, metrics, or dates that
aren't present in it. If something the user asks about isn't in this
data, say so plainly instead of guessing.

Workspace data:
{workspace_context}

User question: {user_input}"""

            # Get response from Gemini
            response = model.generate_content(prompt)
            return response.text

        except Exception as e:
            # Log the full error for debugging
            logger.error(f"Copilot error: {type(e).__name__}: {str(e)}")

            error_msg = str(e).lower()
            error_type = type(e).__name__

            # Check for specific Google API errors
            if "notfound" in error_type.lower() or "404" in error_msg:
                logger.error(
                    f"Gemini model not found or configuration error: {error_type}: {str(e)}"
                )
                return (
                    f"AI model configuration error: The requested Gemini model ({DEFAULT_MODEL_NAME}) "
                    "was not found or is deprecated. Please verify the configured model name."
                )
            elif "resourceexhausted" in error_type.lower():
                logger.warning("Google API quota exceeded")
                return "The AI service quota has been exceeded. Please try again later or check your API quota at https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas"
            elif (
                "api_key" in error_msg
                or "authentication" in error_msg
                or "api key" in error_msg
                or "unauthenticated" in error_type.lower()
            ):
                return "There was an authentication issue with the AI service. Please verify your Google API key is valid."
            elif "permissiondenied" in error_type.lower() or "permission" in error_msg:
                if (
                    "disabled" in error_msg
                    or "has not been used" in error_msg
                    or "enable" in error_msg
                ):
                    logger.error(
                        f"Gemini API disabled on Google Cloud project: {str(e)}"
                    )
                    return """The Gemini API is not enabled for your Google Cloud project. Please:

1. Visit https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com
2. Click "Enable" to activate the Generative Language API
3. Verify your API key has access to the Generative Language API."""
                return "Permission denied. Please check that your API key has access to the Gemini API."
            elif "timeout" in error_msg:
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

    def query(self, user_input: str) -> str:
        """Forward query to the lazily-initialized agent"""
        return get_copilot_agent().query(user_input)


# Export the proxy as copilot_agent for backward compatibility
copilot_agent = CopilotAgentProxy()
