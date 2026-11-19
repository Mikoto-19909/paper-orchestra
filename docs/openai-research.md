# OpenAI research CLI

This branch builds on Google Research main `d9bae997e793a015ed38bb1605be9eb58c00284f`
and the three routing/test commits through `44c92b8488d233c578d572ec9c9b11792a6fdfeb`
from [upstream PR 15](https://github.com/google-research/paper-orchestra/pull/15).
The research entry point uses the upstream outline and writing guidance, the AgentReview
criteria and the upstream stage order. It uses structured prose and a local LaTeX template
to keep compilation, citations and figure inclusion deterministic. Existing Gemini workflows
use the original CLI. OpenAI research runs use Responses at every stage.

## Environment

Verified on Windows with Python 3.12.3, MiKTeX 26.5 and Poppler. Put
`pdflatex`, `bibtex`, `pdftoppm` and `pdftotext` on PATH. Configure an existing project key
and its verified project ID in the process environment as `OPENAI_API_KEY` and
`OPENAI_PROJECT_ID`. The sender uses the official HTTPS endpoint and the system proxy.
Keep the environment, private materials, run directories and shared ledger outside the checkout.

```powershell
py -3.12 -m venv ..\paper-env
& ..\paper-env\Scripts\python.exe -m pip install -r requirements-research-windows.lock
$env:PYTHONUTF8 = '1'
$env:PAPER_TEX_TEST = '1'
& ..\paper-env\Scripts\python.exe -m unittest discover -s tests -v
```

## Input and run

Place UTF-8 `idea_sparse.md` and `experimental_log.md` inside a materials directory.
Add `claims.csv`, `guidelines.md`, `limitations.md`, `references.bib`,
`literature_evidence.md` and captioned images in `figures/` as available.
A custom `template.tex` has one `@@TITLE@@` and one `@@BODY@@` placeholder.
The validation phase supports a local `literature_pool.json` with a `papers` list.
Formal discovery needs a public research question in `public_search` and a positive
search-call cap. The discovery prompt contains that public question. The other
stages send the supplied research evidence to the configured OpenAI project.

Copy the example config and set the writing and review token limits, review count,
revision count, worker count, search limit and retry limit for your budget.
Use the same ledger for **both** commands. Use a separate run directory for each run.

```powershell
& ..\paper-env\Scripts\python.exe paper_writing_cli.py research --materials ..\micro-materials --run-dir ..\micro-run --ledger-root ..\ledger --run-id micro-1 --phase validation --config ..\micro-config.json
& ..\paper-env\Scripts\python.exe paper_writing_cli.py research --materials ..\materials --run-dir ..\paper-run --ledger-root ..\ledger --run-id paper-1 --phase formal --config ..\paper-config.json
& ..\paper-env\Scripts\python.exe -m utils.research_costs ..\ledger ..\costs.json
```

Repeat the original command to resume. Stage signatures include the prompt, model,
generation settings, image content and relevant implementation. The persisted raw response
can be parsed again locally. On a parsing error the CLI prints the path for an optional
corrected JSON file. Keep an explanation of any substantive correction with the run.
Initial and revised manuscripts are in `versions/`; `delivery.json` identifies the latest
checked PDF, and the content and visual checks have separate verdicts. Inspect the exits
and verdicts before reporting a complete run.

## Spending

The shared SQLite ledger enforces USD 10 across runs and USD 1 across validation runs.
Both limits apply to concurrent requests and survive restart. Keep the same ledger path.
Each paid request uses the official input-token count, an output limit, a bound on hosted
tool calls and a cost reservation. Unknown paid requests retain their reservations.
Only explicit rate-limit rejections can be retried automatically within the configured count.
Raw response bodies and usage are saved before application parsing. Incomplete responses
and parse failures preserve the received evidence.

Standard global short-context rates were recorded on 2026-11-19. Per million tokens
(input / cached read / 30-minute write / 12-hour write / output) the rates are
2 / 0.10 / 2.50 / 5 / 10 USD for Sol and 8 / 0.40 / 10 / 20 / 40 USD for Astra.
Web search has a 0.01 USD tool fee per billable call and retrieved tokens use model rates.
Reasoning tokens belong to the output total. Missing usage fields remain missing.
The report separates usage-based estimates, outstanding reservations and official charges.
Check pricing again before using these dated rates for another project.
[OpenAI pricing](https://developers.openai.com/api/docs/pricing) ·
[Web search](https://developers.openai.com/api/docs/guides/tools-web-search) ·
[Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)
