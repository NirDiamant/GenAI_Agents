"""Integration tests for MiniMax provider via utils.llm_provider.

These tests make real API calls and require a valid MINIMAX_API_KEY
environment variable.  Skip automatically when the key is absent.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
SKIP_REASON = "MINIMAX_API_KEY not set"


@unittest.skipUnless(MINIMAX_API_KEY, SKIP_REASON)
class TestMiniMaxIntegration(unittest.TestCase):
    """Live integration tests against the MiniMax API."""

    def _get_llm(self, **kwargs):
        from utils.llm_provider import get_llm
        return get_llm(provider="minimax", **kwargs)

    # ------------------------------------------------------------------
    # Basic connectivity
    # ------------------------------------------------------------------

    def test_simple_prompt(self):
        llm = self._get_llm(temperature=0.1)
        resp = llm.invoke("Say hello in one sentence.")
        self.assertTrue(len(resp.content) > 0)

    def test_response_is_string(self):
        llm = self._get_llm(temperature=0.1)
        resp = llm.invoke("What is 1+1?")
        self.assertIsInstance(resp.content, str)

    # ------------------------------------------------------------------
    # Model variants
    # ------------------------------------------------------------------

    def test_default_model(self):
        llm = self._get_llm(temperature=0.1)
        resp = llm.invoke("Reply with the word 'ok'.")
        self.assertIn("ok", resp.content.lower())

    def test_m27_model(self):
        llm = self._get_llm(model="MiniMax-M2.7", temperature=0.1)
        resp = llm.invoke("Reply with the word 'ok'.")
        self.assertIn("ok", resp.content.lower())

    def test_highspeed_model(self):
        llm = self._get_llm(model="MiniMax-M2.5-highspeed", temperature=0.1)
        resp = llm.invoke("Reply with the word 'ok'.")
        self.assertIn("ok", resp.content.lower())

    # ------------------------------------------------------------------
    # Temperature handling
    # ------------------------------------------------------------------

    def test_temperature_clamped(self):
        """Temperature=0 should be auto-clamped; the API should not reject it."""
        llm = self._get_llm(temperature=0)
        resp = llm.invoke("What is 2+2?")
        self.assertIn("4", resp.content)


@unittest.skipUnless(MINIMAX_API_KEY, SKIP_REASON)
class TestMiniMaxConversation(unittest.TestCase):
    """Test multi-turn conversation with MiniMax."""

    def test_context_retention(self):
        from utils.llm_provider import get_llm
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.runnables.history import RunnableWithMessageHistory
        from langchain_community.chat_message_histories import ChatMessageHistory

        llm = get_llm(provider="minimax", temperature=0.1)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Keep answers concise."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        store = {}

        def get_history(session_id):
            if session_id not in store:
                store[session_id] = ChatMessageHistory()
            return store[session_id]

        chain = RunnableWithMessageHistory(
            prompt | llm,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        config = {"configurable": {"session_id": "integration_test"}}

        r1 = chain.invoke({"input": "My favorite color is blue."}, config=config)
        self.assertTrue(len(r1.content) > 0)

        r2 = chain.invoke({"input": "What is my favorite color?"}, config=config)
        self.assertIn("blue", r2.content.lower())


if __name__ == "__main__":
    unittest.main()
