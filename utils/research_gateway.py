"""Durable, budgeted Responses API transport for the OpenAI research CLI.

All project runs share one SQLite file. Money is integer billionths of USD.
A request in an unknown state keeps its reservation until billing reconciliation.
"""
import base64
from contextlib import closing, contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from uuid import uuid4
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener


USD = 1_000_000_000
SEARCH_FEE = USD // 100
SEARCH_CONTEXT = 131_072  # conservatively round the documented 128k window up
PRICE_DATE = "2026-11-19"
# Ordinary input, cache reads, 30-minute writes, 12-hour writes, total output.
RATES = {
    "gpt-6.2-sol": (2000, 100, 2500, 5000, 10000),
    "gpt-6.2-astra": (8000, 400, 10000, 20000, 40000),
}
PRICING = "https://developers.openai.com/api/docs/pricing"
COUNTING = "https://help.openai.com/en/articles/4936856-understanding-and-counting-tokens"


class BudgetExceeded(RuntimeError):
    pass


class ResearchRequestError(RuntimeError):
    pass


def _json(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _integer(value):
    return type(value) is int and value >= 0


def _image(path):
    data = Path(path).read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif data[:6] in (b"GIF87a", b"GIF89a"):
        mime = "image/gif"
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        raise ValueError(f"Unsupported image bytes in {Path(path).name}")
    return {"type": "input_image", "detail": "high",
            "image_url": f"data:{mime};base64," + base64.b64encode(data).decode("ascii")}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


def _post(endpoint, payload):
    key = os.environ.get("OPENAI_API_KEY")
    project = os.environ.get("OPENAI_PROJECT_ID")
    if not key or not project:
        raise ResearchRequestError("Research key and verified OPENAI_PROJECT_ID required")
    headers = {"Authorization": "Bearer " + key, "OpenAI-Project": project,
               "Content-Type": "application/json", "Accept": "application/json"}
    if os.environ.get("OPENAI_ORG_ID"):
        headers["OpenAI-Organization"] = os.environ["OPENAI_ORG_ID"]
    # Use the system HTTPS proxy on Windows while restricting the destination.
    req = Request("https://api.openai.com/v1/responses" + endpoint,
                  data=_json(payload).encode(), headers=headers, method="POST")
    try:
        with build_opener(_NoRedirect()).open(req, timeout=300) as response:
            return response.status, response.headers.get("x-request-id"), response.read()
    except HTTPError as exc:
        with exc:
            body = re.sub(rb"sk-[A-Za-z0-9_*.-]+", b"[REDACTED]", exc.read())
            return exc.code, exc.headers.get("x-request-id"), body


def _reserve(model, initial, output, calls):
    # Each hosted tool step can process another full search context, including
    # prior text, generated reasoning and returned search content. Only default
    # 30-minute cache writes are requested. Output includes reasoning.
    upper = initial + calls * SEARCH_CONTEXT
    _, _, write, _, out = RATES[model]
    long = upper > 272000
    return upper * write * (2 if long else 1) + output * out * (3 if long else 2) // 2 + calls * SEARCH_FEE


def _settle(model, data, calls, reserve):
    """Return (reported metrics, exact estimate or None, budget exposure)."""
    usage = data.get("usage") or {}
    details = usage.get("input_tokens_details") or {}
    writes = details.get("cache_write_tokens_details") or {}
    tools = data.get("tool_usage")
    items = data.get("output")
    observed = (sum(i.get("type") == "web_search_call" for i in items)
                if isinstance(items, list) else None)
    reported = None
    if isinstance(tools, dict):
        reported = (tools.get("web_search") or {}).get("num_requests") if "web_search" in tools else 0
    metrics = dict(input_tokens=usage.get("input_tokens"), cached_tokens=details.get("cached_tokens"),
                   cache_write_tokens=details.get("cache_write_tokens"),
                   ttl_30m_tokens=writes.get("ttl_30m_tokens"), ttl_12h_tokens=writes.get("ttl_12h_tokens"),
                   output_tokens=usage.get("output_tokens"),
                   reasoning_tokens=(usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
                   search_actions_observed=observed, billable_searches=reported,
                   search_content_tokens=details.get("search_content_tokens"))
    total, output = metrics["input_tokens"], metrics["output_tokens"]
    if not (_integer(total) and _integer(output)):
        return metrics, None, reserve
    read, written = metrics["cached_tokens"], metrics["cache_write_tokens"]
    ordinary, cached, short, extended, out = RATES[model]
    factor = 2 if total > 272000 else 1
    output_cost = output * out * (3 if factor == 2 else 2) // 2
    if not (_integer(read) and _integer(written) and read + written <= total):
        return metrics, None, max(reserve, total * short * factor + output_cost + calls * SEARCH_FEE)
    if writes:
        longer, shorter = writes.get("ttl_12h_tokens"), writes.get("ttl_30m_tokens")
        if not (_integer(longer) and _integer(shorter) and longer + shorter == written):
            return metrics, None, reserve
    else:
        # The payload selects the documented 30-minute default; retain null
        # for the absent response subfields in the exported metrics.
        longer, shorter = 0, written
    tokens = factor * ((total - read - written) * ordinary + read * cached
                       + shorter * short + longer * extended) + output_cost
    if calls and not _integer(reported):
        return metrics, None, max(tokens + calls * SEARCH_FEE, reserve)
    fee = SEARCH_FEE * (reported or 0)
    metrics["estimated_token_fees_usd"] = tokens / USD
    metrics["estimated_search_tool_fees_usd"] = fee / USD
    return metrics, tokens + fee, tokens + fee


_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
 id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, phase TEXT NOT NULL,
 stage TEXT NOT NULL, task TEXT NOT NULL, digest TEXT NOT NULL, endpoint TEXT NOT NULL,
 attempt INTEGER NOT NULL, requested TEXT NOT NULL, model TEXT NOT NULL, config TEXT NOT NULL,
 status TEXT NOT NULL, exposure INTEGER NOT NULL, reserved INTEGER NOT NULL,
 estimated INTEGER, http_status INTEGER, request_id TEXT, raw BLOB,
 response_id TEXT, actual_model TEXT, response_status TEXT, reason TEXT, metrics TEXT,
 UNIQUE(run_id,phase,stage,task,digest,endpoint,attempt)
);
"""


class Gateway:
    def __init__(self, root, run_id, phase):
        if not run_id or phase not in ("validation", "formal"):
            raise ValueError("Expected run_id and validation or formal phase")
        self.root, self.run_id, self.phase = Path(root).resolve(), run_id, phase
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "raw").mkdir(exist_ok=True)
        self.db = self.root / "ledger.sqlite3"
        with self._db() as con:
            con.execute(_SCHEMA)

    @contextmanager
    def _db(self):
        with closing(sqlite3.connect(self.db, timeout=30)) as con:
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA synchronous=FULL")
            with con:
                con.execute("BEGIN IMMEDIATE")
                yield con

    def _budget(self, con, additional):
        sums = con.execute("SELECT COALESCE(SUM(exposure),0), "
                           "COALESCE(SUM(CASE WHEN phase='validation' THEN exposure END),0) FROM calls").fetchone()
        if sums[0] + additional > 10 * USD or (self.phase == "validation" and sums[1] + additional > USD):
            raise BudgetExceeded("Next request exceeds the shared remaining budget")

    def request(self, stage, task, model, prompt, max_output_tokens, *, images=(),
                search_calls=0, fingerprint="", retries=0):
        if model not in RATES or not isinstance(prompt, str):
            raise ValueError("Use a dated, priced research model and a text prompt")
        if not _integer(max_output_tokens) or not 16 <= max_output_tokens <= 128000:
            raise ValueError("max_output_tokens must be 16..128000")
        if not (_integer(search_calls) and _integer(retries)):
            raise ValueError("search_calls and retries must be nonnegative integers")
        parts = [{"type": "input_text", "text": prompt}, *[_image(p) for p in images]]
        count_payload = {"model": model, "input": [{"role": "user", "content": parts}],
                         "reasoning": {"effort": "low"}}
        if search_calls:
            count_payload.update(tools=[{"type": "web_search", "search_context_size": "low"}],
                                 tool_choice="required", parallel_tool_calls=False)
        payload = {**count_payload, "max_output_tokens": max_output_tokens,
                   "service_tier": "default", "store": False}
        if search_calls:
            payload.update(max_tool_calls=search_calls, include=["web_search_call.action.sources"])
        identity = [payload, fingerprint, PRICE_DATE, hashlib.sha256(Path(__file__).read_bytes()).hexdigest()]
        digest = hashlib.sha256(_json(identity).encode()).hexdigest()
        config = {k: v for k, v in payload.items() if k != "input"}
        config.update(image_count=len(parts) - 1, fingerprint=fingerprint,
                      price_date=PRICE_DATE, retries=retries)
        common = (stage, task, digest, model)
        minimum = _reserve(model, 0, max_output_tokens, search_calls)
        count = self._exchange(common, "/input_tokens", count_payload, config, 0, retries, minimum)
        if count.get("object") != "response.input_tokens" or not _integer(count.get("input_tokens")):
            raise ResearchRequestError("Token counting returned no valid count; generation stopped")
        initial = count["input_tokens"]
        amount = _reserve(model, initial, max_output_tokens, search_calls)
        return self._exchange(common, "", payload, {**config, "counted_input": initial}, amount, retries, 0)

    def _exchange(self, common, endpoint, payload, config, reserve, retries, minimum):
        stage, task, digest, model = common
        key = (self.run_id, self.phase, stage, task, digest, endpoint)
        while True:
            with self._db() as con:
                existing = con.execute("SELECT * FROM calls WHERE (run_id,phase,stage,task,digest,endpoint)"
                                       "=(?,?,?,?,?,?) ORDER BY attempt DESC LIMIT 1", key).fetchone()
                if existing and existing["raw"] is not None and existing["status"] != "rate_limited":
                    return self._process(con, existing)
                if existing and existing["status"] != "rate_limited":
                    raise ResearchRequestError("Previous request is in flight or unknown; reservation retained")
                attempt = existing["attempt"] + 1 if existing else 1
                if attempt > retries + 1:
                    raise ResearchRequestError("Rate-limit retry allowance exhausted")
                self._budget(con, max(minimum, reserve))
                stamp = datetime.now(timezone.utc).isoformat()
                cursor = con.execute("INSERT INTO calls(run_id,phase,stage,task,digest,endpoint,attempt,"
                                     "requested,model,config,status,exposure,reserved) VALUES "
                                     "(?,?,?,?,?,?,?,?,?,?,'reserved',?,?)",
                                     (*key, attempt, stamp, model, _json(config), reserve, reserve))
                row_id = cursor.lastrowid
            try:
                status, request_id, raw = _post(endpoint, payload)
            except Exception as exc:
                with self._db() as con:
                    con.execute("UPDATE calls SET status='unknown',reason=? WHERE id=?",
                                (_json(type(exc).__name__), row_id))
                raise ResearchRequestError(f"No confirmed response to call {row_id}; see ledger") from None
            # This transaction durably records the exact body, including usage,
            # before any JSON decoding or output parsing.
            with self._db() as con:
                con.execute("UPDATE calls SET raw=?,http_status=?,request_id=?,status='received' WHERE id=?",
                            (raw, status, request_id, row_id))
            with self._db() as con:
                row = con.execute("SELECT * FROM calls WHERE id=?", (row_id,)).fetchone()
                try:
                    result = self._process(con, row)
                    error = None
                except ResearchRequestError as exc:
                    error = exc  # commit disposition before propagating
            if error is None:
                return result
            if status != 429 or attempt > retries:
                raise error
            time.sleep(min(2 ** (attempt - 1), 10))

    def _process(self, con, row):
        path = self.root / "raw" / f'{row["id"]:06d}.json'
        if not path.exists() or path.read_bytes() != row["raw"]:
            tmp = path.with_name(path.name + "." + uuid4().hex + ".tmp")
            tmp.write_bytes(row["raw"])
            os.replace(tmp, path)
        charge = None
        exposure = row["reserved"]
        status = row["http_status"]
        try:
            data = json.loads(row["raw"])
        except (UnicodeError, ValueError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        metrics = {}
        config = json.loads(row["config"])
        actual = data.get("model")
        response_status = data.get("status")
        reason = data.get("incomplete_details") or data.get("error")
        if status in (400, 401, 403, 404, 422, 429) and isinstance(data.get("error"), dict) and not data.get("usage"):
            state = "rate_limited" if status == 429 else "rejected"
            charge = exposure = 0
        elif status != 200 or not data:
            state = "unknown"
        elif row["endpoint"] == "/input_tokens":
            state = "completed" if _integer(data.get("input_tokens")) else "invalid_count"
            metrics = {"counted_input": data.get("input_tokens")}
            charge = exposure = 0
        else:
            state = response_status or "unknown"
            model = row["model"]
            if (actual == model or isinstance(actual, str) and re.fullmatch(re.escape(model) + r"-\d{4}-\d{2}-\d{2}", actual)) and data.get("service_tier", "default") == "default":
                metrics, charge, exposure = _settle(model, data, config.get("max_tool_calls", 0), exposure)
            else:
                state = "unpriced_response"
        con.execute("UPDATE calls SET status=?,exposure=?,estimated=?,metrics=?,actual_model=?,"
                    "response_id=?,response_status=?,reason=? WHERE id=?",
                    (state, exposure, charge, _json(metrics), actual, data.get("id"),
                     response_status, _json(reason) if reason else None, row["id"]))
        if state != "completed":
            raise ResearchRequestError(f'Call {row["id"]} ended in {state} (HTTP {status}); see {path}')
        return data

    def report(self):
        """Export per-call data, subtotals and outstanding conservative reservations."""
        with self._db() as con:
            # A crash after durable receipt but before parsing needs no new request.
            for pending in con.execute("SELECT * FROM calls WHERE status='received' AND raw IS NOT NULL").fetchall():
                try:
                    self._process(con, pending)
                except ResearchRequestError:
                    pass  # the recorded disposition and raw body are still reportable
            rows = con.execute("SELECT id,run_id,phase,stage,task,endpoint,attempt,requested,model,config,"
                               "status,reserved,exposure,estimated,http_status,request_id,response_id,"
                               "actual_model,response_status,reason,metrics, raw IS NOT NULL AS saved "
                               "FROM calls ORDER BY id").fetchall()
        calls, groups = [], {}
        for record in rows:
            row = dict(record)
            row["endpoint"] = "/v1/responses" + row["endpoint"]
            if row.pop("saved"):
                row["raw_response_path"] = f'raw/{row["id"]:06d}.json'
            for name in ("config", "metrics", "reason"):
                if row[name]:
                    row[name] = json.loads(row[name])
            estimated = row.pop("estimated")
            reserve = row.pop("reserved")
            exposure = row.pop("exposure")
            row.update(reserved_usd=reserve / USD, estimated_usd=None if estimated is None else estimated / USD,
                       outstanding_usd=(exposure - (estimated or 0)) / USD)
            group = (row["phase"], row["stage"], row["actual_model"] or row["model"], row["endpoint"])
            entry = groups.setdefault(group, {"calls": 0, "estimated": 0, "outstanding": 0})
            entry["calls"] += 1
            entry["estimated"] += estimated or 0
            entry["outstanding"] += exposure - (estimated or 0)
            for field, count in (row["metrics"] or {}).items():
                if _integer(count):
                    reported = entry.setdefault("reported_metrics", {})
                    metric = reported.setdefault(field, {"sum": 0, "reporting_calls": 0})
                    metric["sum"] += count
                    metric["reporting_calls"] += 1
            calls.append(row)
        return {"price_date": PRICE_DATE, "price_source": PRICING, "counting_source": COUNTING,
                "rates_nanodollars_per_token": {k: dict(zip(("input","cached","write_30m","write_12h","output"), v))
                                                 for k, v in RATES.items()},
                "search_fee_usd_per_billable_request": SEARCH_FEE / USD,
                "official_charges_usd": None, "official_billing_status": "待对账",
                "known_estimated_usd": sum(g["estimated"] for g in groups.values()) / USD,
                "outstanding_reserved_usd": sum(g["outstanding"] for g in groups.values()) / USD,
                "subtotals": [dict(phase=k[0],stage=k[1],model=k[2],endpoint=k[3],calls=v["calls"],
                                   known_estimated_usd=v["estimated"]/USD, outstanding_usd=v["outstanding"]/USD,
                                   reported_metrics=v.get("reported_metrics", {}))
                              for k,v in groups.items()], "calls": calls}

