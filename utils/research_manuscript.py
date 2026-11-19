"""Typeset evidence-bounded, structured model output into a trusted local template."""
from pathlib import Path
import re
import shutil
import subprocess

TEMPLATE = r"""\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage[margin=1in]{geometry}
\usepackage{url}
\setlength{\emergencystretch}{2em}
\title{@@TITLE@@}
\author{}
\date{}
\begin{document}
\maketitle
@@BODY@@
\bibliographystyle{plain}
\bibliography{references}
\end{document}
"""
ESCAPES = dict(zip("\\&%$#_{}~^", [r"\textbackslash{}",r"\&",r"\%",r"\$",r"\#",r"\_",r"\{",r"\}",r"\textasciitilde{}",r"\textasciicircum{}"]))
CITE = re.compile(r"\[\[([A-Za-z][A-Za-z0-9_-]*)\]\]")
KEY = re.compile(r"[A-Za-z][A-Za-z0-9_-]*\Z")


def encode(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Expected nonempty manuscript text")
    return "".join(ESCAPES.get(c, c) for c in text)


def paragraph(text, keys, cited):
    out, start = [], 0
    for match in CITE.finditer(text):
        k = match[1]
        if k not in keys:
            raise ValueError(f"Unknown citation {k}")
        out.extend([encode(text[start:match.start()]) if text[start:match.start()].strip() else text[start:match.start()], r"\cite{" + k + "}"])
        cited.add(k)
        start = match.end()
    tail = text[start:]
    out.append(encode(tail) if tail.strip() else tail)
    result = "".join(out)
    if "[[" in result or "]]" in result:
        raise ValueError("Malformed citation")
    return result


def typeset(folder, draft, template, library, images, config):
    folder.mkdir(parents=True, exist_ok=True)
    keys, cited = {r["key"] for r in library}, set()
    body = [r"\begin{abstract}", paragraph(draft["abstract"], keys, cited), r"\end{abstract}"]
    sections = draft["sections"]
    if not sections:
        raise ValueError("No manuscript sections")
    included = set()
    for section in sections:
        body.append(r"\section{" + encode(section["heading"]) + "}")
        if not section["paragraphs"]:
            raise ValueError("Empty manuscript section")
        body += [paragraph(t, keys, cited) for t in section["paragraphs"]]
        for item in section.get("figures", []):
            name = item["name"]
            if name not in images or name in included:
                raise ValueError(f"Unknown or duplicate figure {name}")
            src = images[name]
            dest = f"figure-{len(included)+1}{src.suffix.lower()}"
            shutil.copyfile(src, folder / dest)
            included.add(name)
            body.append(r"\begin{figure}[htbp]\centering\includegraphics[width=.86\linewidth,height=.48\textheight,keepaspectratio]{"
                        + dest + r"}\caption{" + encode(item["caption"]) + r"}\end{figure}")
    if included != set(images):
        raise ValueError("Every provided figure requires a caption and placement")
    if library and not cited:
        raise ValueError("No sources cited")
    if any(template.count(marker) != 1 for marker in ("@@TITLE@@", "@@BODY@@")):
        raise ValueError("Template needs exactly one title and body placeholder")
    tex = template.replace("@@TITLE@@", encode(draft["title"])).replace("@@BODY@@", "\n\n".join(body))
    if not cited:
        tex = re.sub(r"\\bibliographystyle\{[^}]+\}|\\bibliography\{references\}", "", tex)
    (folder / "paper.tex").write_text(tex, encoding="utf8")
    entries = []
    for row in library:
        if row["key"] not in cited:
            continue
        # Values are untrusted text, including bibliographic fields.
        fields = {"title": row["title"], "author": row["authors"], "year": row["year"], "howpublished": row["url"]}
        if row.get("doi"):
            fields["doi"] = row["doi"]
        entries.append("@misc{" + row["key"] + ",\n" + ",\n".join(k + " = {" + encode(str(v)) + "}" for k,v in fields.items()) + "\n}")
    (folder / "references.bib").write_text("\n\n".join(entries), encoding="utf8")
    def cmd(args, label):
        result = subprocess.run(args, cwd=folder, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=150)
        (folder / (label + ".log")).write_bytes(result.stdout)
        if result.returncode:
            raise RuntimeError(f"{label} failed ({result.returncode}); inspect {folder}")
    engine = [config["latex"], "-no-shell-escape", "-halt-on-error", "-interaction=nonstopmode", "paper.tex"]
    cmd(engine, "latex1")
    if entries:
        cmd([config["bibtex"], "paper"], "bibtex-run")
        cmd(engine, "latex2")
    cmd(engine, "latex3")
    log = (folder / "paper.log").read_text(errors="replace")
    if re.search(r"There were undefined (references|citations)|Citation .+ undefined", log):
        raise ValueError("Unresolved citation or reference")
    cmd([config["pdftotext"], "-layout", "-enc", "UTF-8", "paper.pdf", "paper.txt"], "extract")
    pages = folder / "pages"
    pages.mkdir(exist_ok=True)
    cmd([config["pdftoppm"], "-scale-to", "1280", "-png", "paper.pdf", "pages/page"], "render")
    pdf_pages = (folder / "paper.txt").read_text(encoding="utf8").split("\f")
    if pdf_pages and not pdf_pages[-1].strip():
        pdf_pages.pop()
    rendered = sorted(pages.glob("page-*.png"), key=lambda p: int(p.stem.split("-")[-1]))
    if not rendered or len(rendered) != len(pdf_pages):
        raise ValueError("Page render count differs from PDF text page count")
    return rendered

