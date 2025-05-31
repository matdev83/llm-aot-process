import time
import requests
import logging
from typing import List, Optional, Tuple, cast
# from llm_accounting import LLMAccounting # Temporarily commented out
# from llm_accounting.audit_log import AuditLogger # Temporarily commented out
# from llm_accounting.models.limits import LimitScope, LimitType, TimeInterval # Temporarily commented out
# from llm_accounting.backends.sqlite import SQLiteBackend # Temporarily commented out

from src.aot.dataclasses import LLMCallStats
from src.aot.constants import OPENROUTER_API_URL, HTTP_REFERER, APP_TITLE

class LLMClient:
    def __init__(self, api_key: str, api_url: str = OPENROUTER_API_URL, http_referer: str = HTTP_REFERER, app_title: str = APP_TITLE,
                 enable_rate_limiting: bool = True, enable_audit_logging: bool = True):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY must be provided.")
        self.api_key = api_key
        self.api_url = api_url
        self.http_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": http_referer,
            "X-Title": app_title,
            "Content-Type": "application/json"
        }
        # sqlite_backend_instance = SQLiteBackend(db_path="data/audit_log.sqlite") # Temporarily commented out
        # self.accounting = LLMAccounting(backend=sqlite_backend_instance) # Temporarily commented out
        self.accounting = None # Placeholder
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_audit_logging = enable_audit_logging
        # self.audit_logger = AuditLogger(backend=sqlite_backend_instance) if enable_audit_logging else None # Temporarily commented out
        self.audit_logger = None # Placeholder

    def call(self, prompt: str, models: List[str], temperature: float, max_tokens: Optional[int] = None, top_p: Optional[float] = None) -> Tuple[str, LLMCallStats]:
        # --- TEMPORARY MODIFICATION FOR TESTING ORCHESTRATOR ---
        model_name_to_report = models[0] if models else "dummy_model"
        
        # Extract string content from prompt object/list for stub logic
        prompt_str_for_stub = ""
        if isinstance(prompt, list) and prompt:
            if isinstance(prompt[0], dict) and "content" in prompt[0]:
                prompt_str_for_stub = prompt[0]["content"] # Use content of the first message
        elif isinstance(prompt, str): # Should not happen if processor sends list
            prompt_str_for_stub = prompt

        logging.info(f"LLMClient.call (TEMPORARILY STUBBED): Called with model {model_name_to_report}, temp={temperature}, max_tokens={max_tokens}, top_p={top_p}, prompt: '{prompt_str_for_stub[:100]}...'")
        dummy_response_content = f"Conceptual response for model {model_name_to_report}: "

        if "classify" in prompt_str_for_stub.lower():
            dummy_response_content += "CONTINUE" # Default classification
        elif "initial" in prompt_str_for_stub.lower() or "thought" in prompt_str_for_stub.lower() or "generate" in prompt_str_for_stub.lower() or "first thought" in prompt_str_for_stub.lower():
            # Added "first thought" as initial prompt templates often include this.
            dummy_response_content += "Your thought: This is a new thought." # Default thought, include marker
        else: # Fallback for other types of prompts, e.g. one-shot from orchestrator
            dummy_response_content += "Some generic content from stubbed LLM."

        dummy_stats = LLMCallStats(
            model_name=model_name_to_report,
            prompt_tokens=len(prompt_str_for_stub.split()), # Approximate
            completion_tokens=len(dummy_response_content.split()), # Approximate
            call_duration_seconds=0.1
        )
        return dummy_response_content, dummy_stats
        # --- END OF TEMPORARY MODIFICATION ---

        # Original code below, commented out for now:
        # if not models:
            # logging.error("No models provided for LLM call.")
            # return "Error: No models configured for call.", LLMCallStats(model_name="N/A")

        # last_error_content_for_failover = f"Error: All models ({', '.join(models)}) in the list failed."
        # last_error_stats_for_failover = LLMCallStats(model_name=models[0] if models else "N/A")
        # request_id = None # Initialize request_id

        # for model_name in models:
            # payload = {
                # "model": model_name,
                # "messages": [{"role": "user", "content": prompt}],
                # "temperature": temperature,
                # "reasoning": {"effort": "high"}
            # }
            # if max_tokens is not None:
                # payload["max_tokens"] = max_tokens
            # if top_p is not None:
                # payload["top_p"] = top_p

            # current_call_stats = LLMCallStats(model_name=model_name)
            # call_start_time = time.monotonic()
            
            # try:
                # # if self.enable_rate_limiting: # Temporarily commented out
                # #     allowed, reason = self.accounting.check_quota(
                # #         model=model_name,
                # #         username="api_user",
                # #         caller_name="LLMClient",
                # #         input_tokens=len(prompt.split()) # Approximate token count for quota check
                # #     )
                # #     if not allowed:
                # #         logging.warning(f"Rate limit exceeded for model {model_name}: {reason}. Trying next model if available.")
                # #         last_error_content_for_failover = f"Error: Rate limit exceeded - {reason}"
                # #         last_error_stats_for_failover = current_call_stats
                # #         continue # Skip to the next model

                # # Log the request before making the API call
                # # self.accounting.track_usage( # Temporarily commented out
                # #     model=model_name,
                # #     prompt_tokens=len(prompt.split()),  # Approximate token count
                # #     caller_name="LLMClient",
                # #     username="api_user"
                # # )
                # logging.info(f"Attempting LLM call with model: {model_name}")

                # response = requests.post(self.api_url, headers=self.http_headers, json=payload, timeout=90)
                # current_call_stats.call_duration_seconds = time.monotonic() - call_start_time
                
                # response_payload = None
                # try:
                    # response_payload = response.json()
                # except requests.exceptions.JSONDecodeError:
                    # logging.warning(f"Could not parse JSON response from {model_name}. Status: {response.status_code}, Body: {response.text}")
                    # # Still attempt to log response with what we have if it's an error scenario handled below

                # if 400 <= response.status_code < 600:
                    # error_body_text = response.text
                    # if response_payload is not None:
                        # response_dict = response_payload  # Type checker now knows this is not None
                        # if 'error' in response_dict:
                            # error_body_text = str(response_dict.get('error', {}).get('message', error_body_text))
                    
                    # logging.warning(f"API call to {model_name} failed with HTTP status {response.status_code}: {error_body_text}. Trying next model if available.")
                    # last_error_content_for_failover = f"Error: API call to {model_name} (HTTP {response.status_code}) - {error_body_text}"
                    
                    # # Try to get usage even from error responses
                    # if response_payload is not None:
                        # usage = response_payload.get("usage", {})  # Now safe due to None check
                        # current_call_stats.completion_tokens = usage.get("completion_tokens", 0)
                        # current_call_stats.prompt_tokens = usage.get("prompt_tokens", 0)
                    
                    # # self.accounting.track_usage( # Temporarily commented out
                    # #     model=model_name,
                    # #     prompt_tokens=current_call_stats.prompt_tokens,
                    # #     completion_tokens=current_call_stats.completion_tokens,
                    # #     execution_time=current_call_stats.call_duration_seconds,
                    # #     caller_name="LLMClient",
                    # #     username="api_user"
                    # # )
                    # last_error_stats_for_failover = current_call_stats
                    # continue

                # response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx) if not handled above
                
                # data = response_payload # Already parsed

                # if data is not None:
                    # if "error" in data:
                        # error_msg = data['error'].get('message', str(data['error']))
                        # logging.error(f"LLM API Error ({model_name}) in 2xx response: {error_msg}")
                        # content = f"Error: API returned an error - {error_msg}"
                    # elif not data.get("choices") or not data["choices"][0].get("message"):
                        # logging.error(f"Unexpected API response structure from {model_name}: {data}")
                        # content = "Error: Unexpected API response structure"
                    # else:
                        # content = data["choices"][0]["message"]["content"]
                    
                    # usage = data.get("usage", {})
                # else:
                    # content = "Error: No valid response data received"
                    # usage = {}
                # current_call_stats.completion_tokens = usage.get("completion_tokens", 0)
                # current_call_stats.prompt_tokens = usage.get("prompt_tokens", 0)

                # # self.accounting.track_usage( # Temporarily commented out
                # #     model=model_name,
                # #     prompt_tokens=current_call_stats.prompt_tokens,
                # #     completion_tokens=current_call_stats.completion_tokens,
                # #     execution_time=current_call_stats.call_duration_seconds,
                # #     caller_name="LLMClient",
                # #     username="api_user"
                # # )

                # # if self.enable_audit_logging and self.audit_logger: # Temporarily commented out
                # #     self.audit_logger.log_prompt(
                # #         app_name="LLMClient",
                # #         user_name="api_user",
                # #         model=model_name,
                # #         prompt_text=prompt
                # #     )
                # #     # Explicitly cast data to dict to satisfy Pylance
                # #     response_data = cast(dict, data)
                # #     self.audit_logger.log_response(
                # #         app_name="LLMClient",
                # #         user_name="api_user",
                # #         model=model_name,
                # #         response_text=content,
                # #         remote_completion_id=response_data.get("id")
                # #     )

                # if content.startswith("Error:"):
                    # if model_name != models[-1]:
                         # logging.warning(f"LLM call to {model_name} resulted in error: {content}. Trying next model.")
                         # last_error_content_for_failover = content
                         # last_error_stats_for_failover = current_call_stats
                         # continue
                # else:
                    # if current_call_stats.completion_tokens == 0 and content:
                         # logging.warning(f"completion_tokens reported as 0 from {model_name}, but content was received.")
                
                # return content, current_call_stats

            # except requests.exceptions.HTTPError as e:
                # current_call_stats.call_duration_seconds = time.monotonic() - call_start_time
                # response_data_for_log = None
                # if e.response is not None:
                    # try:
                        # response_data_for_log = e.response.json()
                        # usage = response_data_for_log.get("usage", {})
                        # current_call_stats.completion_tokens = usage.get("completion_tokens", 0)
                        # current_call_stats.prompt_tokens = usage.get("prompt_tokens", 0)
                    # except:
                        # response_data_for_log = {"error": str(e), "status_code": e.response.status_code, "text": e.response.text}
                # else:
                    # response_data_for_log = {"error": str(e)}

                    # # self.accounting.track_usage( # Temporarily commented out
                    # #     model=model_name,
                    # #     prompt_tokens=current_call_stats.prompt_tokens,
                    # #     completion_tokens=current_call_stats.completion_tokens,
                    # #     execution_time=current_call_stats.call_duration_seconds,
                    # #     caller_name="LLMClient",
                    # #     username="api_user"
                    # # )
                
                # if e.response is not None and (400 <= e.response.status_code < 500 and e.response.status_code not in [401, 403, 429]):
                    # logging.warning(f"API request to {model_name} failed with HTTPError (status {e.response.status_code}): {e}. Trying next model.")
                    # last_error_content_for_failover = f"Error: API call to {model_name} (HTTP {e.response.status_code}) - {e}"
                    # last_error_stats_for_failover = current_call_stats
                    # continue
                # else:
                    # logging.error(f"API request to {model_name} failed with non-failover HTTPError: {e}")
                    # return f"Error: API call failed - {e}", current_call_stats

            # except requests.exceptions.Timeout as e:
                # current_call_stats.call_duration_seconds = time.monotonic() - call_start_time
                # # self.accounting.track_usage( # Temporarily commented out
                # #     model=model_name,
                # #     execution_time=current_call_stats.call_duration_seconds,
                # #     caller_name="LLMClient",
                # #     username="api_user"
                # # )
                # logging.warning(f"API request to {model_name} timed out after {current_call_stats.call_duration_seconds:.2f}s. Trying next model if available.")
                # last_error_content_for_failover = f"Error: API call to {model_name} timed out"
                # last_error_stats_for_failover = current_call_stats
                # continue

            # except requests.exceptions.RequestException as e:
                # current_call_stats.call_duration_seconds = time.monotonic() - call_start_time
                # # self.accounting.track_usage( # Temporarily commented out
                # #     model=model_name,
                # #     execution_time=current_call_stats.call_duration_seconds,
                # #     caller_name="LLMClient",
                # #     username="api_user"
                # # )
                # logging.warning(f"API request to {model_name} failed with RequestException: {e}. Trying next model if available.")
                # last_error_content_for_failover = f"Error: API call to {model_name} failed - {e}"
                # last_error_stats_for_failover = current_call_stats
                # continue
            
            # finally: # Ensure request_id is reset if the loop continues for another model
                # if model_name != models[-1]: # If there are more models to try
                    # request_id = None # Reset for the next model attempt in the loop
        
        # logging.error(f"All models ({', '.join(models)}) failed. Last error from model '{last_error_stats_for_failover.model_name}': {last_error_content_for_failover}")
        # # If all models failed, the last error's request_id should have already been logged with its respective error.
        # # If the loop was never entered (e.g. models list was initially empty, though caught earlier), request_id would be None.
        # return last_error_content_for_failover, last_error_stats_for_failover
