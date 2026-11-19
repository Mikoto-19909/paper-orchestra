from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from autoraters.agent_review import perform_review_agentreview
from utils.openai_utils import get_openai_image_part_from_path


class ResearchRoutingTests(unittest.TestCase):
    def test_review_and_meta_review_route_to_openai(self):
        with patch("utils.llm_backend_utils.call_openai_models_with_text_prompt",
                   return_value={"parsed_response": {"Overall": 6}}) as send:
            result = perform_review_agentreview("Synthetic abstract", "gpt-6.2-astra", 2)
        self.assertEqual(result["Overall"], 6)
        self.assertEqual(send.call_count, 3)
        self.assertTrue(all(call.kwargs["model_name"] == "gpt-6.2-astra" for call in send.call_args_list))

    def test_mime_from_actual_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "misnamed.jpg"
            Image.new("RGB", (2, 2)).save(path, format="PNG")
            part, = get_openai_image_part_from_path(path)
            self.assertTrue(part["image_url"]["url"].startswith("data:image/png;base64,"))
