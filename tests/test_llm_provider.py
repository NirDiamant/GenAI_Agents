"""Unit tests for utils.llm_provider."""

import os
import unittest
from unittest.mock import patch, MagicMock

import sys

# Ensure the repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.llm_provider import (
    PROVIDERS,
    get_llm,
    get_provider_info,
    list_providers,
)


class TestListProviders(unittest.TestCase):
    """Tests for list_providers()."""

    def test_returns_list(self):
        result = list_providers()
        self.assertIsInstance(result, list)

    def test_contains_openai(self):
        self.assertIn("openai", list_providers())

    def test_contains_minimax(self):
        self.assertIn("minimax", list_providers())

    def test_at_least_two_providers(self):
        self.assertGreaterEqual(len(list_providers()), 2)


class TestGetProviderInfo(unittest.TestCase):
    """Tests for get_provider_info()."""

    def test_openai_config(self):
        info = get_provider_info("openai")
        self.assertIsNone(info["base_url"])
        self.assertEqual(info["api_key_env"], "OPENAI_API_KEY")
        self.assertEqual(info["default_model"], "gpt-4o-mini")

    def test_minimax_config(self):
        info = get_provider_info("minimax")
        self.assertEqual(info["base_url"], "https://api.minimax.io/v1")
        self.assertEqual(info["api_key_env"], "MINIMAX_API_KEY")
        self.assertEqual(info["default_model"], "MiniMax-M2.7")

    def test_unknown_provider_raises(self):
        with self.assertRaises(ValueError) as ctx:
            get_provider_info("nonexistent")
        self.assertIn("nonexistent", str(ctx.exception))

    def test_returns_copy(self):
        info = get_provider_info("minimax")
        info["base_url"] = "modified"
        self.assertEqual(
            PROVIDERS["minimax"]["base_url"],
            "https://api.minimax.io/v1",
        )


class TestGetLlm(unittest.TestCase):
    """Tests for get_llm()."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"})
    @patch("utils.llm_provider.ChatOpenAI")
    def test_openai_default(self, mock_cls):
        mock_cls.return_value = MagicMock()
        llm = get_llm(provider="openai")
        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args[1]
        self.assertEqual(call_kwargs["model"], "gpt-4o-mini")
        self.assertEqual(call_kwargs["api_key"], "test-openai-key")
        self.assertNotIn("base_url", call_kwargs)

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-minimax-key"})
    @patch("utils.llm_provider.ChatOpenAI")
    def test_minimax_default(self, mock_cls):
        mock_cls.return_value = MagicMock()
        llm = get_llm(provider="minimax")
        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args[1]
        self.assertEqual(call_kwargs["model"], "MiniMax-M2.7")
        self.assertEqual(call_kwargs["api_key"], "test-minimax-key")
        self.assertEqual(call_kwargs["base_url"], "https://api.minimax.io/v1")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-minimax-key"})
    @patch("utils.llm_provider.ChatOpenAI")
    def test_minimax_custom_model(self, mock_cls):
        mock_cls.return_value = MagicMock()
        get_llm(provider="minimax", model="MiniMax-M2.5-highspeed")
        call_kwargs = mock_cls.call_args[1]
        self.assertEqual(call_kwargs["model"], "MiniMax-M2.5-highspeed")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-minimax-key"})
    @patch("utils.llm_provider.ChatOpenAI")
    def test_minimax_temperature_clamping_zero(self, mock_cls):
        mock_cls.return_value = MagicMock()
        get_llm(provider="minimax", temperature=0)
        call_kwargs = mock_cls.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 0.01)

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-minimax-key"})
    @patch("utils.llm_provider.ChatOpenAI")
    def test_minimax_temperature_negative_clamped(self, mock_cls):
        mock_cls.return_value = MagicMock()
        get_llm(provider="minimax", temperature=-0.5)
        call_kwargs = mock_cls.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 0.01)

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-minimax-key"})
    @patch("utils.llm_provider.ChatOpenAI")
    def test_minimax_temperature_positive_preserved(self, mock_cls):
        mock_cls.return_value = MagicMock()
        get_llm(provider="minimax", temperature=0.5)
        call_kwargs = mock_cls.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 0.5)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "key"})
    @patch("utils.llm_provider.ChatOpenAI")
    def test_openai_temperature_zero_not_clamped(self, mock_cls):
        mock_cls.return_value = MagicMock()
        get_llm(provider="openai", temperature=0)
        call_kwargs = mock_cls.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 0)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "key"})
    @patch("utils.llm_provider.ChatOpenAI")
    def test_max_tokens_forwarded(self, mock_cls):
        mock_cls.return_value = MagicMock()
        get_llm(provider="openai", max_tokens=2048)
        call_kwargs = mock_cls.call_args[1]
        self.assertEqual(call_kwargs["max_tokens"], 2048)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "key"})
    @patch("utils.llm_provider.ChatOpenAI")
    def test_extra_kwargs_forwarded(self, mock_cls):
        mock_cls.return_value = MagicMock()
        get_llm(provider="openai", top_p=0.9)
        call_kwargs = mock_cls.call_args[1]
        self.assertEqual(call_kwargs["top_p"], 0.9)

    def test_unknown_provider_raises(self):
        with self.assertRaises(ValueError):
            get_llm(provider="nonexistent")

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_raises(self):
        # Remove all env vars to trigger the error
        for key in ("OPENAI_API_KEY", "MINIMAX_API_KEY"):
            os.environ.pop(key, None)
        with self.assertRaises(EnvironmentError) as ctx:
            get_llm(provider="openai")
        self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_minimax_key_raises(self):
        for key in ("OPENAI_API_KEY", "MINIMAX_API_KEY"):
            os.environ.pop(key, None)
        with self.assertRaises(EnvironmentError) as ctx:
            get_llm(provider="minimax")
        self.assertIn("MINIMAX_API_KEY", str(ctx.exception))


class TestProviderRegistry(unittest.TestCase):
    """Tests for the PROVIDERS registry itself."""

    def test_all_providers_have_required_keys(self):
        required = {"base_url", "api_key_env", "default_model"}
        for name, config in PROVIDERS.items():
            self.assertTrue(
                required.issubset(config.keys()),
                f"Provider {name!r} is missing keys: {required - config.keys()}",
            )

    def test_minimax_base_url_is_https(self):
        self.assertTrue(
            PROVIDERS["minimax"]["base_url"].startswith("https://")
        )

    def test_openai_has_no_base_url(self):
        self.assertIsNone(PROVIDERS["openai"]["base_url"])


if __name__ == "__main__":
    unittest.main()
