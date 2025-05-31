import logging
from typing import Optional

from .dataclasses import L2TNodeCategory

logger = logging.getLogger(__name__)


class L2TResponseParser:
    @staticmethod
    def parse_l2t_initial_response(response_text: str) -> Optional[str]:
        """
        Parses the raw text response from the LLM after an initial prompt.
        Looks for "Your thought:" and extracts the following text.
        """
        try:
            # Case-insensitive search for the marker
            marker = "Your thought:"
            marker_idx = response_text.lower().rfind(marker.lower())
            if marker_idx != -1:
                # Extract text after the marker
                # Add len(marker) to marker_idx to get the starting position of the thought
                thought = response_text[marker_idx + len(marker) :].strip()
                if thought:
                    return thought
                else:
                    logger.warning(
                        "Found 'Your thought:' but no subsequent text was found in response: '%s'",
                        response_text
                    )
                    return None
            else:
                logger.warning(
                    "'Your thought:' marker not found in initial response: '%s'",
                    response_text
                )
                return None
        except Exception as e:
            logger.warning(
                "Error parsing initial LLM response: '%s'. Error: %s",
                response_text,
                e,
                exc_info=True,
            )
            return None

    @staticmethod
    def parse_l2t_node_classification_response(
        response_text: str,
    ) -> Optional[L2TNodeCategory]:
        """
        Parses the raw text response from the LLM after a node classification prompt.
        Looks for "Your classification:" and extracts the category string.
        Matches the string to an L2TNodeCategory enum member.
        """
        try:
            marker = "Your classification:"
            marker_idx = response_text.lower().rfind(marker.lower()) # Use rfind to get the last occurrence
            if marker_idx != -1:
                category_str = response_text[marker_idx + len(marker) :].strip()
                if not category_str:
                    logger.warning(
                        "Found 'Your classification:' but no subsequent text was found in response: '%s'",
                        response_text
                    )
                    return None
                try:
                    # Attempt to match the first word to a category to handle cases where LLM might add extra text.
                    # e.g. "FINAL_ANSWER because..."
                    potential_category = category_str.split()[0].upper()
                    return L2TNodeCategory[potential_category]
                except (KeyError, ValueError):
                    logger.warning(
                        "Invalid L2TNodeCategory extracted: '%s' from response: '%s'",
                        category_str,
                        response_text,
                    )
                    return None
            else:
                logger.warning(
                    "'Your classification:' marker not found in classification response: '%s'",
                    response_text,
                )
                return None
        except Exception as e:
            logger.warning(
                "Error parsing node classification LLM response: '%s'. Error: %s",
                response_text,
                e,
                exc_info=True,
            )
            return None

    @staticmethod
    def parse_l2t_thought_generation_response(
        response_text: str,
    ) -> Optional[str]:
        """
        Parses the raw text response from the LLM after a thought generation prompt.
        Looks for "Your new thought:" and extracts the following text.
        """
        try:
            marker = "Your new thought:"
            marker_idx = response_text.lower().rfind(marker.lower())
            if marker_idx != -1:
                thought = response_text[marker_idx + len(marker) :].strip()
                if thought:
                    return thought
                else:
                    logger.warning(
                        "Found 'Your new thought:' but no subsequent text was found in response: '%s'",
                        response_text
                    )
                    return None
            else:
                logger.warning(
                    "'Your new thought:' marker not found in thought generation response: '%s'",
                    response_text,
                )
                return None
        except Exception as e:
            logger.warning(
                "Error parsing thought generation LLM response: '%s'. Error: %s",
                response_text,
                e,
                exc_info=True,
            )
            return None
