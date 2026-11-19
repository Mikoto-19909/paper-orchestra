import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from research_cli import DEFAULT, parse_response, run, validate, verify_sources
from utils.research_manuscript import paragraph


class SyntheticGateway:
    def __init__(self):
        self.calls = []

    def request(self, stage, task, model, prompt, max_output_tokens, **kwargs):
        self.calls.append(stage)
        fixture = {"title":"Synthetic unit test","abstract":"Two inputs were checked.",
                   "sections":[{"heading":"Evidence","paragraphs":["Two inputs were checked [[fixture]]."]}]}
        outputs = {
            "outline":{"sections":["Evidence"],"questions":[],"limits":[]},"discovery":{"papers":[{"title":"Synthetic study",
            "key":"fixture","authors":"Example, A.","year":"2001","url":"https://example.test/paper",
            "question":"Synthetic check","locator":"Section 2","support":"Synthetic comparison."}]},
            "literature":{"paragraphs":["Synthetic comparison [[fixture]]."]},
            "review":{"issues":[],"assessment":"Sufficient"},"writing":fixture,"revision":fixture,
            "review_summary":{"changes":[],"limitations":[]},"content_check":{"passed":True,"checks":["Checked input count"],"issues":[]},
            "visual":{"passed":True,"issues":[]},
        }
        result = {"status":"completed","output":[{"content":[{"type":"output_text","text":json.dumps(outputs[stage])}]}]}
        if stage == "discovery":
            assert kwargs["search_calls"] == 2
            result["output"].append({"type":"web_search_call","status":"completed","action":{
                "sources":[{"url":"https://example.test/paper"}]}})
        if stage == "visual":
            assert kwargs["images"][0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        return result


class ResearchPipelineTest(unittest.TestCase):
    def test_complete_json_recovered_from_preamble(self):
        result = {"status":"completed","output":[{"content":[{"type":"output_text",
                  "text":'Here is the result:\n{"changes":[],"limitations":[]}'}]}]}
        self.assertEqual(parse_response(result),{"changes":[],"limitations":[]})
        result["status"] = "incomplete"
        with self.assertRaisesRegex(ValueError,"incomplete"):
            parse_response(result)

    def test_invalid_stage_not_accepted(self):
        with self.assertRaisesRegex(ValueError,"outline"):
            validate("outline",{"sections":[]})

    def test_source_normalization_and_deduplication(self):
        paper = {"title":"Synthetic primary text","key":"fixture","year":"2001","authors":"Example, A.",
                 "url":"https://example.test/paper","question":"Unit test","locator":"Section 2","support":"Synthetic finding"}
        raw = {"output":[{"type":"web_search_call","status":"completed","action":{"sources":[
                          {"url":paper["url"]+"?utm_source=test"}]}}]}
        self.assertEqual(verify_sources({"papers":[paper,dict(paper,key="duplicate")]},raw),{"papers":[paper]})

    def test_ungrounded_search_rejected(self):
        with self.assertRaisesRegex(ValueError,"No completed web search"):
            verify_sources({"papers":[]},{"output":[]})

    def test_invalid_citation_rejected(self):
        with self.assertRaisesRegex(ValueError,"Unknown citation"):
            paragraph("See [[wrong]]",{"fixture"},set())

    @unittest.skipUnless(os.environ.get("PAPER_TEX_TEST"), "Run with PAPER_TEX_TEST=1 for full TeX integration")
    def test_real_typesetting_mock_responses_and_resume(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("PAPER_TEST_TMP")) as td:
            root = Path(td)
            m = root/"inputs"
            m.mkdir()
            (m/"idea_sparse.md").write_text("Count synthetic inputs.")
            (m/"experimental_log.md").write_text("There are exactly two inputs.")
            g = SyntheticGateway()
            cfg = DEFAULT | {"public_search":"Synthetic test question"}
            args = (m,root/"run",root/"ledger","unit","formal",cfg,g)
            original = g.request
            def interrupted(stage,*a,**kwargs):
                if stage == "review_summary":
                    raise RuntimeError("simulated interruption")
                return original(stage,*a,**kwargs)
            with patch.object(g,"request",side_effect=interrupted), self.assertRaisesRegex(RuntimeError,"interruption"):
                run(*args)
            result = run(*args)
            self.assertTrue(Path(result["pdf"]).is_file())
            self.assertGreaterEqual(result["pages"],1)
            self.assertEqual(g.calls.count("writing"),1)
            self.assertEqual(g.calls.count("review"),1)
            n = len(g.calls)
            self.assertEqual(result,run(*args))
            self.assertEqual(len(g.calls),n)

