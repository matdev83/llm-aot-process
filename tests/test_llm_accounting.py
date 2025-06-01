import unittest
import os
import sqlite3
import json
from unittest.mock import patch, Mock, call
import time
import requests

from src.llm_client import LLMClient
from src.aot.dataclasses import LLMCallStats

DB_NAME = "llm_accounting.db"

# @unittest.skip("Skipping all LLMAccounting tests due to stubbed llm_accounting module and internal changes to LLMClient")
class TestLLMAccountingIntegration(unittest.TestCase):

    def setUp(self):
        os.environ["OPENROUTER_API_KEY"] = "dummy_key_for_testing"
        self.llm_client = LLMClient(api_key="dummy_key_for_testing", enable_audit_logging=True)
        if os.path.exists(DB_NAME):
            os.remove(DB_NAME)
        self.mock_monotonic_patch = patch('time.monotonic')
        self.mock_monotonic = self.mock_monotonic_patch.start()
        self.mock_monotonic.side_effect = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    def tearDown(self):
        if os.path.exists(DB_NAME):
            os.remove(DB_NAME)
        del os.environ["OPENROUTER_API_KEY"]
        self.mock_monotonic_patch.stop()

    @unittest.skip("Skipping due to stubbed llm_accounting module in LLMClient")
    @patch('requests.post')
    def test_successful_llm_call_logging(self, mock_post):
        # Configure mock for requests.post
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response_payload = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            "model": "test_model_success"
        }
        mock_response.json.return_value = mock_response_payload
        mock_post.return_value = mock_response

        track_usage_patch = patch.object(self.llm_client.accounting, 'track_usage', wraps=self.llm_client.accounting.track_usage)
        mock_track_usage = track_usage_patch.start()
        self.addCleanup(track_usage_patch.stop)

        prompt = "Test prompt for success"
        models = ["test_model_success"]
        temperature = 0.7
        
        content, stats = self.llm_client.call(prompt, models, temperature)

        self.assertEqual(content, "Test response")
        self.assertEqual(stats.prompt_tokens, 10)
        self.assertEqual(stats.completion_tokens, 20)
        self.assertEqual(stats.model_name, "test_model_success")
        self.assertEqual(stats.call_duration_seconds, 1.0)
        self.assertEqual(mock_track_usage.call_count, 2)
        first_call_args = mock_track_usage.call_args_list[0][1]
        self.assertEqual(first_call_args['model'], "test_model_success")
        self.assertEqual(first_call_args['prompt_tokens'], len(prompt.split()))
        second_call_args = mock_track_usage.call_args_list[1][1]
        self.assertEqual(second_call_args['model'], "test_model_success")
        self.assertEqual(second_call_args['prompt_tokens'], 10)
        self.assertEqual(second_call_args['completion_tokens'], 20)
        self.assertEqual(second_call_args['execution_time'], 1.0)

    @unittest.skip("Skipping due to stubbed llm_accounting module in LLMClient")
    @patch('requests.post')
    def test_api_error_with_usage_logging(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 500
        mock_error_payload = {
            "error": {"message": "Server error"},
            "usage": {"prompt_tokens": 5, "completion_tokens": 0},
            "model": "test_model_api_error"
        }
        mock_response.json.return_value = mock_error_payload
        mock_response.text = json.dumps(mock_error_payload)
        mock_post.return_value = mock_response
        mock_response.raise_for_status = Mock(side_effect=requests.exceptions.HTTPError("Server Error", response=mock_response))
        mock_track_usage = patch.object(self.llm_client.accounting, 'track_usage', wraps=self.llm_client.accounting.track_usage).start()
        self.addCleanup(mock_track_usage.stop)

        prompt = "Test prompt for API error"
        models = ["test_model_api_error"]
        content, stats = self.llm_client.call(prompt, models, 0.7)
        self.assertTrue(content.startswith("Error: API call to test_model_api_error (HTTP 500)"))
        self.assertEqual(stats.prompt_tokens, 5)
        self.assertEqual(mock_track_usage.call_count, 2)
            
    @unittest.skip("Skipping due to stubbed llm_accounting module in LLMClient")
    @patch('requests.post')
    def test_network_error_logging(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")
        mock_track_usage = patch.object(self.llm_client.accounting, 'track_usage', wraps=self.llm_client.accounting.track_usage).start()
        self.addCleanup(mock_track_usage.stop)
        prompt = "Test prompt for network error"
        models = ["test_model_network_error"]
        content, stats = self.llm_client.call(prompt, models, 0.7)
        self.assertTrue(content.startswith("Error: API call to test_model_network_error timed out"))
        self.assertEqual(mock_track_usage.call_count, 2)

    @unittest.skip("Skipping due to stubbed llm_accounting module in LLMClient")
    @patch('requests.post')
    def test_model_failover_logging(self, mock_post):
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            "choices": [{"message": {"content": "Failover successful"}}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 25},
            "model": "model_two_success"
        }
        mock_post.side_effect = [
            requests.exceptions.Timeout("Timeout on model_one_fail"),
            mock_response_success
        ]
        mock_track_usage = patch.object(self.llm_client.accounting, 'track_usage', wraps=self.llm_client.accounting.track_usage).start()
        self.addCleanup(mock_track_usage.stop)
        prompt = "Test prompt for failover"
        models = ["model_one_fail", "model_two_success"]
        self.mock_monotonic.side_effect = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] # Reset for this test
        content, stats = self.llm_client.call(prompt, models, 0.7)
        self.assertEqual(content, "Failover successful")
        self.assertEqual(stats.model_name, "model_two_success")
        self.assertEqual(mock_track_usage.call_count, 4)


if __name__ == '__main__':
    unittest.main()
