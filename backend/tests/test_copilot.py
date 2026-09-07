"""
Unit tests for AI Copilot Agent with complete mocking (zero external network/API calls).
"""

import pytest
from unittest.mock import patch, MagicMock
from backend.copilot.agent import AICopilotAgent, CopilotAgentProxy, get_copilot_agent


def _mock_text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


class TestCopilotAgent:

    def test_query_without_api_key_returns_configuration_message(self):
        """When API key is not configured, query returns guidance without attempting network calls."""
        with patch("backend.copilot.agent.settings") as mock_settings:
            mock_settings.anthropic_api_key = ""
            agent = AICopilotAgent()
            response = agent.query("How do I train a model?")
            assert "AI Copilot requires an Anthropic API key to be configured" in response
            assert "ANTHROPIC_API_KEY" in response

    def test_query_success_with_mocked_llm(self):
        """When API key is set, query correctly prompts the LLM and extracts the response text."""
        with patch("backend.copilot.agent.settings") as mock_settings, patch(
            "anthropic.Anthropic"
        ) as mock_client_cls:

            mock_settings.anthropic_api_key = "mock-valid-key"

            # Mock the Anthropic client instance and its messages.create response
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.content = [_mock_text_block("This is a simulated AI Copilot answer.")]
            mock_client.messages.create.return_value = mock_response
            mock_client_cls.return_value = mock_client

            agent = AICopilotAgent()
            result = agent.query("Explain random forest")

            assert result == "This is a simulated AI Copilot answer."
            mock_client_cls.assert_called_with(api_key="mock-valid-key")
            mock_client.messages.create.assert_called_once()

            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert call_kwargs["model"] == "claude-sonnet-5"
            assert "User question: Explain random forest" in call_kwargs["messages"][0]["content"]

    def test_query_handles_auth_error_gracefully(self):
        """When the LLM raises an authentication / invalid key error, agent catches it and returns guidance."""
        with patch("backend.copilot.agent.settings") as mock_settings, patch(
            "anthropic.Anthropic"
        ) as mock_client_cls:

            mock_settings.anthropic_api_key = "invalid-key"
            mock_client = MagicMock()

            class AuthenticationError(Exception):
                pass

            mock_client.messages.create.side_effect = AuthenticationError(
                "invalid x-api-key"
            )
            mock_client_cls.return_value = mock_client

            agent = AICopilotAgent()
            result = agent.query("Hello")

            assert "There was an authentication issue with the AI service" in result

    def test_query_handles_quota_error_gracefully(self):
        """When the LLM raises a rate limit / quota error, agent catches it and informs the user."""
        with patch("backend.copilot.agent.settings") as mock_settings, patch(
            "anthropic.Anthropic"
        ) as mock_client_cls:

            mock_settings.anthropic_api_key = "test-key"
            mock_client = MagicMock()

            class RateLimitError(Exception):
                pass

            mock_client.messages.create.side_effect = RateLimitError(
                "rate limit exceeded"
            )
            mock_client_cls.return_value = mock_client

            agent = AICopilotAgent()
            result = agent.query("Hello")

            assert "The AI service quota has been exceeded" in result

    def test_query_handles_model_not_found_gracefully(self):
        """When the model is not found, agent returns model configuration error."""
        with patch("backend.copilot.agent.settings") as mock_settings, patch(
            "anthropic.Anthropic"
        ) as mock_client_cls:

            mock_settings.anthropic_api_key = "test-key"
            mock_client = MagicMock()

            class NotFoundError(Exception):
                pass

            mock_client.messages.create.side_effect = NotFoundError(
                "404 model: claude-sonnet-5 not found"
            )
            mock_client_cls.return_value = mock_client

            agent = AICopilotAgent()
            result = agent.query("Hello")

            assert "AI model configuration error" in result
            assert "claude-sonnet-5" in result

    def test_query_handles_permission_denied_gracefully(self):
        """When the API key lacks access to the model, agent returns permission guidance."""
        with patch("backend.copilot.agent.settings") as mock_settings, patch(
            "anthropic.Anthropic"
        ) as mock_client_cls:

            mock_settings.anthropic_api_key = "test-key"
            mock_client = MagicMock()

            class PermissionDeniedError(Exception):
                pass

            mock_client.messages.create.side_effect = PermissionDeniedError(
                "permission denied for this model"
            )
            mock_client_cls.return_value = mock_client

            agent = AICopilotAgent()
            result = agent.query("Hello")

            assert "Permission denied" in result

    def test_query_includes_history_for_multiturn_context(self):
        """Prior turns are forwarded to the LLM so follow-up questions resolve correctly."""
        with patch("backend.copilot.agent.settings") as mock_settings, patch(
            "anthropic.Anthropic"
        ) as mock_client_cls:

            mock_settings.anthropic_api_key = "mock-valid-key"

            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.content = [_mock_text_block("It's 0.98 accuracy.")]
            mock_client.messages.create.return_value = mock_response
            mock_client_cls.return_value = mock_client

            agent = AICopilotAgent()
            history = [
                {"role": "user", "content": "Tell me about the housing model"},
                {"role": "assistant", "content": "It's an XGBoost classifier."},
            ]
            result = agent.query("What's its accuracy?", history=history)

            assert result == "It's 0.98 accuracy."
            call_kwargs = mock_client.messages.create.call_args.kwargs
            sent_messages = call_kwargs["messages"]
            assert sent_messages[0] == history[0]
            assert sent_messages[1] == history[1]
            assert sent_messages[2]["role"] == "user"
            assert "What's its accuracy?" in sent_messages[2]["content"]

    def test_query_normalizes_consecutive_same_role_turns(self):
        """Two consecutive user turns in history (e.g. after a dropped error turn)
        are merged rather than sent as-is, since Anthropic requires strict
        user/assistant alternation."""
        with patch("backend.copilot.agent.settings") as mock_settings, patch(
            "anthropic.Anthropic"
        ) as mock_client_cls:

            mock_settings.anthropic_api_key = "mock-valid-key"

            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.content = [_mock_text_block("Answer.")]
            mock_client.messages.create.return_value = mock_response
            mock_client_cls.return_value = mock_client

            agent = AICopilotAgent()
            # Two consecutive "user" turns, as could happen if an assistant
            # turn errored out and was dropped from history.
            history = [{"role": "user", "content": "First question"}]
            agent.query("Second question", history=history)

            call_kwargs = mock_client.messages.create.call_args.kwargs
            sent_messages = call_kwargs["messages"]
            roles = [m["role"] for m in sent_messages]
            # No two consecutive entries share a role
            assert all(roles[i] != roles[i + 1] for i in range(len(roles) - 1))
            assert roles[0] == "user"

    def test_copilot_proxy_and_singleton(self):
        """Verify proxy forwards queries correctly."""
        proxy = CopilotAgentProxy()
        with patch("backend.copilot.agent.settings") as mock_settings:
            mock_settings.anthropic_api_key = ""
            res = proxy.query("Test question")
            assert "AI Copilot requires an Anthropic API key to be configured" in res
