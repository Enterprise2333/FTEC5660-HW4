"""FTEC5660 Homework 4: the retrieval half of a RAG assistant over company reports.

Fill in the five functions marked ### YOUR CODE HERE. You may add imports, constants and helper
functions above them, but do not change the provided code below them (the models, the generator,
the loaders, the runner and the scoring).

    python3 hw4.py --data-folder public_test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek

load_dotenv()


# =============================================================================
# Your code: add imports, constants and helper functions here.
# =============================================================================


from dataclasses import dataclass
from functools import lru_cache
from collections import defaultdict

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi


def _text(tag):
    return " ".join(tag.get_text(" ", strip=True).split())


def _table_lines(table):
    caption = table.find("caption")
    caption = _text(caption) if caption else ""
    headers = []
    lines = []
    for row in table.find_all("tr"):
        cells = row.find_all(["th", "td"], recursive=False)
        if not cells:
            continue
        values = [_text(cell) for cell in cells]
        if all(cell.name == "th" and cell.get("scope") != "row" for cell in cells):
            headers = [value for value, cell in zip(values, cells)
                       for _ in range(int(cell.get("colspan", 1)))]
            continue
        if len(values) < 2 or len(headers) < len(values):
            lines.append(" | ".join(values))
            continue
        label = values[0]
        line = label
        for header, value in zip(headers[1:], values[1:]):
            pair = f"{header}: {value}"
            if len(line) + len(pair) + 3 > MAX_LINE_CHARS and line != label:
                lines.append(line)
                line = label
            line += f" | {pair}"
        if line != label:
            lines.append(line)
    return caption, lines


def table_to_text(table_html: str) -> str:
    table = BeautifulSoup(table_html, "lxml").find("table")
    if table is None:
        return ""
    caption, lines = _table_lines(table)
    return "\n".join(part for part in [caption, *lines] if part)


def _metadata(soup, file_name):
    heading = soup.find("h1")
    company = re.sub(r"\s*\([^()]*\)\s*$", "", _text(heading)) if heading else ""
    title = _text(soup.title) if soup.title else ""
    intro = " ".join(_text(el) for el in list(heading.next_elements)
                     if getattr(el, "name", None) in ("h2", "p")) if heading and not title else ""
    text = title or intro[:450]
    if re.search(r"results review|equity research", text, re.I) or soup.select_one(".broker"):
        kind = "research_report"
    elif re.search(r"earnings.*call|conference call", text, re.I):
        kind = "earnings_call"
    else:
        kind = "annual_report"
    year = re.search(r"(?:FY\s*|Annual Report\s+|Full Year\s+)(20\d{2})", text, re.I)
    if not year:
        year = re.search(r"(20\d{2})", text)
    return {"source": file_name, "company": company, "fiscal_year": int(year[1]) if year else None,
            "doc_type": kind}


def _split_text(text, limit):
    return RecursiveCharacterTextSplitter(chunk_size=limit, chunk_overlap=min(120, limit // 5)).split_text(text)


def _chunks(file_name, html, *, labelled_tables, section_chunks, register=False):
    soup = BeautifulSoup(html, "lxml")
    meta = _metadata(soup, file_name)
    if register and meta["company"]:
        _company_names[company_key(meta["company"])] = meta["company"]
    if not section_chunks:
        if labelled_tables:
            for table in soup.find_all("table"):
                table.replace_with("\n" + table_to_text(str(table)) + "\n")
        body = soup.get_text("\n", strip=True)
        return [Document(page_content=s, metadata=dict(meta)) for s in _split_text(body, MAX_CHUNK_CHARS)]

    chunks = []
    pending = []
    section = ""

    def emit(text, table=False):
        if text:
            chunks.append(Document(page_content=text, metadata={
                **meta, "section": section, "table": table,
            }))

    def flush():
        if pending:
            header = section + "\n" if section else ""
            for part in _split_text("\n".join(pending), MAX_CHUNK_CHARS - len(header)):
                emit(header + part)
            pending.clear()

    for element in (soup.body or soup).find_all(["h1", "h2", "h3", "p", "li", "table"]):
        if element.find_parent(["table", "nav", "li"]):
            continue
        if element.name in ("h1", "h2", "h3"):
            flush()
            section = _text(element)[:220]
        elif element.name == "table":
            flush()
            caption_tag = element.find("caption")
            caption = _text(caption_tag) if caption_tag and labelled_tables else ""
            rows = table_to_text(str(element)).splitlines() if labelled_tables else [_text(element)]
            if caption and rows and rows[0] == caption:
                rows = rows[1:]
            header = "\n".join(filter(None, [section, caption]))[: MAX_CHUNK_CHARS // 2]
            part = header
            for row in rows:
                for item in _split_text(row, MAX_CHUNK_CHARS - len(header) - 1):
                    if len(part) + len(item) + 1 > MAX_CHUNK_CHARS:
                        emit(part, table=True)
                        part = header
                    part = (part + "\n" + item).strip()
            if part != section or rows:
                emit(part, table=True)
        elif _text(element):
            pending.append(_text(element))
    flush()
    return chunks


_company_names = {}
_YEAR = re.compile(r"(?<!\d)(?:FY\s*)?(20\d{2})(?!\d)", re.I)
_COMPANY = r"[A-Z][\w&.-]*(?:[ \t]+(?:[A-Z][\w&.-]*|and|of|the|&))*"
_ENTITY_PATTERNS = [
    re.compile(rf"\b({_COMPANY})[’']s?(?=\s|$)"),
    re.compile(rf"\b(?:did|does|is|for|give|of)\s+({_COMPANY})(?=\s|[?,.]|$)"),
]
_ENTITY_FILLERS = {"USD", "FY", "Q4", "CEO", "CFO", "December", "January", "Which", "What"}


def chunk_document(file_name: str, html: str) -> list[Document]:
    return _chunks(file_name, html, labelled_tables=True, section_chunks=True, register=True)


def parse_query(question: str) -> dict:
    low = question.casefold()
    if re.search(r"\b(?:earnings|conference)\s+(?:conference\s+)?call\b", low):
        kind = "earnings_call"
    elif re.search(r"\b(?:rating|target price|forecast|results review|broker|estimate)\b", low):
        kind = "research_report"
    else:
        kind = "annual_report"
    year = _YEAR.search(question)
    names = []
    if ":" in question and re.search(r"\bwhich\b", low):
        options = re.split(r"\s+(?:or|versus|vs\.?)\s+", question.rsplit(":", 1)[1].strip(" ?."))
        if len(options) == 2 and all(re.fullmatch(_COMPANY, part) for part in options):
            names = options
    if not names:
        for pattern in _ENTITY_PATTERNS:
            candidates = [name.strip(" .") for name in pattern.findall(question)
                          if name.split()[0] not in _ENTITY_FILLERS and not _YEAR.fullmatch(name)]
            known = [name for name in candidates if company_key(name) in _company_names]
            if known or candidates:
                names = known or candidates[:1]
                break
    if not names:
        # Match whole names boundaries and prefer the longest name at a given location.
        padded = " " + company_key(question) + " "
        names = [name for key, name in sorted(_company_names.items(), key=lambda item: -len(item[0]))
                 if f" {key} " in padded]
    return {"companies": list(dict.fromkeys(names)), "fiscal_year": int(year[1]) if year else None,
            "doc_type": kind}


@lru_cache(maxsize=4096)
def _tokens(text):
    return tuple(text.casefold().split())


_WORD = re.compile(r"[a-z0-9]+")


def _words(text):
    return " ".join(_WORD.findall(text.casefold()))


_METRICS = (
    (r"cash and cash equivalents", ("cash and cash equivalents",)),
    (r"net cash from operating|operating cash flow", ("net cash from operating activities", "operating cash flow")),
    (r"accounts receivable|net receivables", ("accounts receivable net", "net accounts receivable")),
    (r"long.term borrow", ("long term borrowings", "long term debt")),
    (r"total assets", ("total assets",)),
    (r"total equity", ("total equity",)),
    (r"net income|net profit", ("net income", "net profit", "profit for the year")),
    (r"operating income|operating profit", ("operating income", "operating profit")),
    (r"gross profit", ("gross profit",)),
    (r"inventor", ("inventories", "inventory")),
    (r"employees|headcount", ("total employees", "number of employees", "employees")),
    (r"chief financial officer|\bcfo\b", ("chief financial officer", "cfo")),
    (r"chief executive officer|\bceo\b", ("chief executive officer", "ceo")),
    (r"audit firm|auditor|audited", ("independent auditor", "auditor", "audited")),
    (r"headquarter", ("headquarters", "headquartered")),
    (r"target price", ("target price",)),
    (r"rating", ("rating",)),
    (r"repurchase|buyback", ("share repurchase", "buyback")),
    (r"guid(?:e|ance)|outlook", ("guidance", "expect", "outlook")),
    (r"revenue|sales", ("revenue", "sales")),
)


def _metric(question):
    return next((terms for pattern, terms in _METRICS if re.search(pattern, question, re.I)), ())


@dataclass
class _Index:
    chunks: list[Document]
    bm25: Any
    by_company: dict
    words: list[str]


def build_index(chunks: list[Document]) -> _Index:
    _company_names.clear()
    groups = defaultdict(list)
    for i, chunk in enumerate(chunks):
        company = chunk.metadata.get("company", "")
        if company:
            key = company_key(company)
            groups[key].append(i)
            _company_names[key] = company
    bm25 = BM25Okapi([_tokens(c.page_content) for c in chunks]) if chunks else None
    return _Index(chunks, bm25, dict(groups), [_words(c.page_content) for c in chunks])


def _supported_year(chunk, year, kind):
    if year is None:
        return True
    published = chunk.metadata.get("fiscal_year")
    if kind == "annual_report":
        return published in (year, year + 1)
    return published == year or year in {int(m[1]) for m in _YEAR.finditer(chunk.page_content)}


def _evidence_boost(index, i, question, parsed):
    chunk = index.chunks[i]
    terms = _metric(question)
    lines = chunk.page_content.splitlines()
    year = parsed["fiscal_year"]
    published = chunk.metadata.get("fiscal_year")
    score = 0.0
    if published == year:
        score += 4
    qwords = _words(question)
    exact_rows = []
    for line in lines:
        row = _words(line.split("|", 1)[0])
        if any(f" {term} " in f" {row} " for term in terms):
            exact_rows.append(line)
    if exact_rows:
        score += 15
        if any("|" in line for line in exact_rows):
            score += 10
        if year and any(str(year) in line for line in exact_rows):
            score += 6
    if terms == ("revenue", "sales"):
        for line in lines:
            label = _words(line.split("|", 1)[0])
            if "|" in line and len(label) >= 3 and f" {label} " in f" {qwords} ":
                if label not in ("revenue", "sales", "total", "usd"):
                    score += 24
                    break
        if "|" in chunk.page_content and any(line.casefold().startswith("revenue |") for line in lines):
            score += 8
    if published != year and year and not any(str(year) in line for line in exact_rows):
        if parsed["doc_type"] == "annual_report":
            score -= 12
    return score


def _substantive_match(index, ranked, question):
    modifier = re.search(r"\bfrom (?:its|their) (.+?)\s+in\s+(?:FY|20)", question, re.I)
    if modifier:
        phrase = _words(modifier[1])
        return any(phrase in index.words[i] for i in ranked)
    # When the requested financial concept is unknown, require its distinctive
    # words together in the evidence instead of accepting generic spending text.
    if _metric(question):
        return True
    cleaned = re.sub(rf"{_COMPANY}[’']s?(?=\s|$)", "", question)
    match = re.search(r"\b(?:spend on|what (?:was|were))\s+(.+?)\s+(?:in|at)\s+(?:FY|20|\d)", cleaned, re.I)
    if not match:
        return True
    phrase = _words(match[1].strip())
    return len(phrase.split()) < 2 or any(phrase in index.words[i] for i in ranked)


def _retrieve(index, question, *, mode):
    if not index.chunks:
        return []
    scores = index.bm25.get_scores(_tokens(question))
    if mode < 3:
        return [index.chunks[i] for i in sorted(range(len(scores)), key=lambda i: -scores[i])[:MAX_RETRIEVED]]
    parsed = parse_query(question)
    companies = [company_key(name) for name in parsed["companies"]]
    if mode >= 6 and any(company not in index.by_company for company in companies):
        return []
    kind, year = parsed["doc_type"], parsed["fiscal_year"]
    groups = []
    for company in companies or [None]:
        pool = index.by_company.get(company, list(range(len(scores)))) if company else range(len(scores))
        typed = [i for i in pool if index.chunks[i].metadata.get("doc_type") == kind]
        dated = [i for i in typed if _supported_year(index.chunks[i], year, kind)]
        if mode >= 6 and (not dated or not _substantive_match(index, dated, question)):
            return []
        groups.append(dated or typed or list(pool))
    candidates = set(i for group in groups for i in group)
    ranking = {i: scores[i] + (_evidence_boost(index, i, question, parsed) if mode >= 4 else 0)
               for i in candidates}
    result = []
    if mode >= 5 and len(groups) > 1:
        for group in groups:
            if group:
                best = max(group, key=lambda i: (ranking[i], -i))
                if best not in result:
                    result.append(best)
    result.extend(i for i in sorted(candidates, key=lambda i: (-ranking[i], i)) if i not in result)
    return [index.chunks[i] for i in result[:MAX_RETRIEVED]]


def retrieve(index: Any, question: str) -> list[Document]:
    return _retrieve(index, question, mode=6)


# =============================================================================
# Provided code below. Do not change it.
# =============================================================================

LLM_MODEL = "deepseek-v4-flash"
DOC_TYPES = ("annual_report", "earnings_call", "research_report")

MAX_CHUNK_CHARS = 1500  # characters per chunk
MAX_RETRIEVED = 5       # chunks per question
TIME_LIMIT_SECONDS = 30 * 60

# USD per 1M tokens, only for the cost estimate that the runner prints.
PRICE_LLM_INPUT, PRICE_LLM_OUTPUT = 0.15, 0.60

# Points (Section 2 of the handout).
POINTS = {"1.1": 8, "1.2": 7, "2.1": 5, "2.2": 10, "2.3": 15, "3": 10, "4": 20}
RECALL_ZERO, RECALL_FULL = 0.50, 0.90  # Task 2.3: 0 points at or below RECALL_ZERO, full points from RECALL_FULL
MAX_LINE_CHARS = 400                   # Task 1.1
MIN_PROBES_FOUND = 0.9                 # Task 1.2


class UsageTracker(BaseCallbackHandler):
    """Counts LLM tokens, for the cost estimate. There is no budget."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = self.llm_input_tokens = self.llm_output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        tokens_in = tokens_out = 0
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
        with self._lock:
            self.llm_calls += 1
            self.llm_input_tokens += tokens_in
            self.llm_output_tokens += tokens_out

    def cost(self) -> float:
        return (self.llm_input_tokens * PRICE_LLM_INPUT + self.llm_output_tokens * PRICE_LLM_OUTPUT) / 1e6

    def summary(self) -> dict:
        return {"llm_calls": self.llm_calls, "llm_input_tokens": self.llm_input_tokens,
                "llm_output_tokens": self.llm_output_tokens, "cost_usd": round(self.cost(), 4)}


USAGE = UsageTracker()            # your calls
GENERATOR_USAGE = UsageTracker()  # the generator's calls


def _chat(usage: UsageTracker, temperature: float, thinking: bool, max_tokens: int | None) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=LLM_MODEL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=3,
        # Thinking mode ignores `temperature` and its tokens count as output.
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        callbacks=[usage],
    )


def make_llm(temperature: float = 0.0, thinking: bool = False, max_tokens: int | None = None) -> ChatDeepSeek:
    """deepseek-v4-flash. Use it for every LLM call you make (only in parse_query, build_index and retrieve)."""
    return _chat(USAGE, temperature, thinking, max_tokens)


# ----------------------------------------------------------------------------- the generator

GENERATOR_PROMPT = """You answer questions about company reports, using only the context below.

Context:
{context}

Question: {question}

Rules:
- Use only the context. The header of each passage says which document it comes from.
- Check that the passage you use is about the company and the fiscal year that the question asks about.
- If the context does not contain the answer, reply exactly NOT_FOUND.
- For a number, reply with that one number only, in the unit the question asks for, without units, words or other numbers (for example: 1913.4).
- For a name (a person, a company, a city, an audit firm or a rating), reply with the name only.

Answer:"""

_generator: ChatDeepSeek | None = None


def format_context(chunks: list[Document]) -> str:
    """The retrieved chunks as the generator sees them: a header from the metadata, then the text."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata or {}
        header = " | ".join(f"{key}: {str(meta[key])[:100]}"
                            for key in ("source", "company", "fiscal_year", "doc_type") if meta.get(key) not in (None, ""))
        parts.append(f"[{i}] {header}\n{chunk.page_content}")
    return "\n\n".join(parts)


def generate(question: str, chunks: list[Document]) -> str:
    """The fixed generator: deepseek-v4-flash answers from the retrieved chunks only."""
    global _generator
    if not chunks:
        return "NOT_FOUND"
    if _generator is None:
        _generator = _chat(GENERATOR_USAGE, 0.0, False, 64)
    prompt = GENERATOR_PROMPT.format(context=format_context(chunks), question=question)
    return str(_generator.invoke(prompt).content).strip()


# ----------------------------------------------------------------------------- loaders

def load_corpus(folder: str | Path) -> dict[str, str]:
    """{file name: raw HTML} for every .html file in the folder."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(folder).glob("*.html"))}


def load_json(path: str | Path) -> Any:
    """The parsed file, or None if it does not exist (the private set has no answers)."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- scoring

# One number: optional minus sign, optional thousands separators, optional decimals.
NUMBER_RE = re.compile(r"(?<![\d.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
TITLES = {"mr", "mrs", "ms", "dr", "prof", "professor", "sir"}
LEGAL_SUFFIXES = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}


def parse_number(text: str) -> float | None:
    """The single number in the text, or None if there is not exactly one."""
    found = NUMBER_RE.findall(text.replace("\u2212", "-"))
    if len(found) != 1:
        return None
    return float(found[0].replace(",", ""))


def normalize(text: str) -> str:
    """Lower case, no punctuation, no titles, padded with spaces for whole-word matching."""
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("_", " ")
    words = [w for w in re.sub(r"[^a-z0-9]+", " ", text).split() if w not in TITLES]
    return " " + " ".join(words) + " "


def company_key(name: Any) -> str:
    """A company name without case, punctuation or legal suffix: 'Northgate Robotics Inc.' -> 'northgate robotics'."""
    return " ".join(w for w in normalize(name).split() if w not in LEGAL_SUFFIXES)


def squash(text: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def as_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def answer_correct(answer: str, gold: dict) -> bool:
    answer = str(answer or "")
    said_not_found = " not found " in normalize(answer)
    if gold["type"] == "not_found":
        return said_not_found
    if said_not_found:
        return False
    if gold["type"] == "number":
        value = parse_number(answer)
        if value is None:
            return False
        return abs(value - gold["answer"]) <= max(0.01 * abs(gold["answer"]), 0.05)
    names = [gold["answer"], *gold.get("aliases", [])]
    if not any(normalize(name) in normalize(answer) for name in names):
        return False
    return not any(normalize(other) in normalize(answer) for other in gold.get("reject", []))


def table_score(output: Any, table: dict) -> float:
    """Task 1.1, one table: the share of cells found on a line with their row and column labels and
    without the label of another row, halved if the caption is missing."""
    if not isinstance(output, str) or not table["cells"]:
        return 0.0
    lines = [squash(line) for line in output.splitlines()]
    lines = [line for line in lines if line and len(line) <= MAX_LINE_CHARS]
    labels = {squash(row) for row, _, _ in table["cells"]}

    def kept(row: str, col: str, value: str) -> bool:
        row, col, value = squash(row), squash(col), squash(value)
        others = [o for o in labels if o not in row]  # e.g. "total liabilities" may be in "total liabilities and equity"
        return any(row in line and col in line and value in line and not any(o in line for o in others)
                   for line in lines)

    share = sum(kept(*cell) for cell in table["cells"]) / len(table["cells"])
    return share if squash(table["caption"]) in squash(output) else share / 2


def document_problems(chunks: Any, error: str | None, meta: dict, file_name: str) -> list[str]:
    """Task 1.2, one document: why its chunks fail the checks (an empty list means they pass)."""
    if error:
        return [f"error: {error}"]
    if not isinstance(chunks, list) or not chunks:
        return ["no chunks"]
    if not all(isinstance(c, Document) for c in chunks):
        return ["not a list of Document"]
    problems = []
    longest = max(len(c.page_content) for c in chunks)
    if longest > MAX_CHUNK_CHARS:
        problems.append(f"a chunk has {longest} characters")
    for key, ok in (("source", lambda v: v == file_name),
                    ("company", lambda v: company_key(v) == company_key(meta["company"])),
                    ("fiscal_year", lambda v: as_year(v) == meta["fiscal_year"]),
                    ("doc_type", lambda v: v == meta["doc_type"])):
        wrong = [c.metadata.get(key) for c in chunks if not ok(c.metadata.get(key))]
        if wrong:
            problems.append(f"{key} {wrong[0]!r} (expected {meta.get(key, file_name)!r}) on {len(wrong)} chunk(s)")
    text = squash("\n".join(c.page_content for c in chunks))
    missing = [p for p in meta["probes"] if squash(p) not in text]
    if len(missing) > (1 - MIN_PROBES_FOUND) * len(meta["probes"]):
        problems.append(f"text missing from the chunks: {missing[:3]}")
    return problems


def parse_score(parsed: Any, gold: dict) -> float:
    """Task 2.2, one question: the share of the three fields that are right."""
    if not isinstance(parsed, dict):
        return 0.0
    expected = gold["query"]
    companies = parsed.get("companies")
    if isinstance(companies, str):
        companies = [companies]
    right = 0
    if isinstance(companies, (list, tuple)) and {company_key(c) for c in companies} == {company_key(c) for c in expected["companies"]}:
        right += 1
    if "fiscal_year" in parsed and as_year(parsed["fiscal_year"]) == expected["fiscal_year"] and \
            (parsed["fiscal_year"] is None) == (expected["fiscal_year"] is None):
        right += 1
    if parsed.get("doc_type") == expected["doc_type"]:
        right += 1
    return right / 3


def contains(chunk_text: str, printed: str) -> bool:
    """The fact as printed in the document is in the chunk (a number only as a whole number)."""
    text, printed = squash(chunk_text), squash(printed)
    if re.fullmatch(r"[\d,.%-]+", printed):
        return re.search(rf"(?<![\d.,]){re.escape(printed)}(?![\d]|[.,]\d)", text) is not None
    return printed in text


def retrieval_hit(chunks: list[Document], gold: dict) -> bool:
    """For every fact the question needs, a retrieved chunk of a document that states it contains it."""
    return bool(gold["evidence"]) and all(
        any(c.metadata.get("source") == file and contains(c.page_content, printed)
            for c in chunks for file, printed in group)
        for group in gold["evidence"])


def clip(x: float) -> float:
    return max(0.0, min(1.0, x))


# ----------------------------------------------------------------------------- runner

def main() -> None:
    parser = argparse.ArgumentParser(description="FTEC5660 HW4: retrieval for RAG over company reports")
    parser.add_argument("--data-folder", default="public_test",
                        help="folder with corpus/, questions.json and (optionally) the answer files")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = load_corpus(folder / "corpus")
    questions = load_json(folder / "questions.json")
    ground_truth = load_json(folder / "ground_truth.json")
    docs_meta = load_json(folder / "docs.json")
    tables = load_json(folder / "tables.json")
    print(f"{len(corpus)} documents, {len(questions)} questions")
    start = time.perf_counter()
    points: dict[str, float] = {}

    # Task 1.1: tables (no model calls allowed)
    if tables:
        before = USAGE.llm_calls
        scores = []
        for table in tables:
            try:
                scores.append(table_score(table_to_text(table["html"]), table))
            except Exception as e:
                print(f"table_to_text failed on a table of {table['source']}: {e!r}")
                scores.append(0.0)
        if USAGE.llm_calls != before:
            print("WARNING: table_to_text called the LLM, which is not allowed (Task 1.1 = 0).")
            scores = [0.0]
        points["1.1"] = POINTS["1.1"] * sum(scores) / len(scores)
        print(f"\nTask 1.1 table_to_text: {sum(scores) / len(scores):.1%} of the table cells kept with their labels")

    # Task 1.2: chunks (no model calls allowed)
    before = USAGE.llm_calls
    chunks_by_file: dict[str, list] = {}
    errors: dict[str, str] = {}
    for file_name, html in corpus.items():
        try:
            chunks_by_file[file_name] = list(chunk_document(file_name, html) or [])
        except Exception as e:
            traceback.print_exc()
            chunks_by_file[file_name], errors[file_name] = [], repr(e)
    called_models = USAGE.llm_calls != before
    if called_models:
        print("WARNING: chunk_document called the LLM, which is not allowed (Task 1.2 = 0).")
    if docs_meta:
        problems = {f: document_problems(chunks_by_file.get(f), errors.get(f), m, f) for f, m in docs_meta.items()}
        passed = sum(not p for p in problems.values()) / len(problems)
        points["1.2"] = 0.0 if called_models else POINTS["1.2"] * passed
        print(f"Task 1.2 chunk_document: {passed:.1%} of the documents pass")
        for f, p in [(f, p) for f, p in problems.items() if p][:5]:
            print(f"  {f}: {'; '.join(p)}")

    # the chunks that retrieve() may return
    all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
                  if isinstance(c, Document) and len(c.page_content) <= MAX_CHUNK_CHARS]
    by_text: dict[str, list[Document]] = {}
    for c in all_chunks:
        by_text.setdefault(c.page_content, []).append(c)
    print(f"{len(all_chunks)} chunks to index")

    # Task 2.2: query parsing
    parsed: dict[str, Any] = {}
    for qid, question in questions.items():
        try:
            parsed[qid] = parse_query(question)
        except Exception as e:
            parsed[qid] = f"error: {e!r}"

    # Tasks 2.1, 2.3 and 3: retrieval
    index = build_index(all_chunks)
    index_seconds = time.perf_counter() - start
    print(f"Index built in {index_seconds:.1f}s | {USAGE.summary()}")
    retrieved: dict[str, list[Document]] = {}
    valid: dict[str, bool] = {}
    for qid, question in questions.items():
        try:
            result = retrieve(index, question)
            ok = isinstance(result, (list, tuple)) and len(result) <= MAX_RETRIEVED
            kept = []
            for chunk in list(result or []) if isinstance(result, (list, tuple)) else []:
                matches = by_text.get(getattr(chunk, "page_content", None), [])
                if not matches:
                    ok = False
                    continue
                source = (getattr(chunk, "metadata", None) or {}).get("source")
                kept.append(next((m for m in matches if m.metadata.get("source") == source), matches[0]))
            retrieved[qid], valid[qid] = kept[:MAX_RETRIEVED], ok
        except Exception:
            traceback.print_exc()
            retrieved[qid], valid[qid] = [], False
    retrieval_seconds = time.perf_counter() - start - index_seconds

    # Task 4: the fixed generator
    def answer(qid: str) -> str:
        try:
            return generate(questions[qid], retrieved[qid])
        except Exception as e:
            print(f"Generator failed on {qid}: {e!r}")
            return ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = dict(zip(questions, pool.map(answer, questions)))
    total_seconds = time.perf_counter() - start

    rows = []
    for qid, question in questions.items():
        sources = sorted({c.metadata.get("source", "") for c in retrieved[qid]})
        row = {"id": qid, "category": "", "question": question, "parsed": json.dumps(parsed[qid], default=str),
               "parse_score": "", "retrieved": ";".join(sources), "valid": valid[qid], "retrieval_hit": "",
               "answer": answers[qid], "correct": "", "expected": ""}
        if ground_truth and qid in ground_truth:
            gold = ground_truth[qid]
            row["category"] = gold["category"]
            row["parse_score"] = round(parse_score(parsed[qid], gold), 3)
            row["retrieval_hit"] = retrieval_hit(retrieved[qid], gold) if gold["evidence"] else ""
            row["correct"] = answer_correct(answers[qid], gold)
            if not row["correct"]:
                row["expected"] = gold["answer"]
        rows.append(row)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")

    if ground_truth:
        graded = [r for r in rows if r["id"] in ground_truth]
        answerable = [r for r in graded if ground_truth[r["id"]]["evidence"]]
        unanswerable = [r for r in graded if not ground_truth[r["id"]]["evidence"]]
        parse_mean = sum(r["parse_score"] for r in graded) / len(graded)
        non_empty = sum(r["valid"] and bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        recall = sum(r["retrieval_hit"] for r in answerable) / max(len(answerable), 1)
        kept = sum(bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        rejected = sum(not r["retrieved"] for r in unanswerable) / max(len(unanswerable), 1)
        accuracy = sum(r["correct"] for r in graded) / len(graded)
        points["2.1"] = POINTS["2.1"] * non_empty
        points["2.2"] = POINTS["2.2"] * parse_mean
        points["2.3"] = POINTS["2.3"] * clip((recall - RECALL_ZERO) / (RECALL_FULL - RECALL_ZERO))
        points["3"] = POINTS["3"] * clip(kept + rejected - 1)
        points["4"] = POINTS["4"] * accuracy
        print(f"Task 2.1 valid, non-empty retrieval on answerable questions: {non_empty:.1%}")
        print(f"Task 2.2 parse_query fields right: {parse_mean:.1%}")
        print(f"Task 2.3 retrieval recall@{MAX_RETRIEVED} on answerable questions: {recall:.1%}")
        print(f"Task 3   answerable questions with chunks: {kept:.1%} | unanswerable questions with []: {rejected:.1%}")
        print(f"Task 4   answer accuracy: {accuracy:.1%}")
        by_category: dict[str, list[tuple]] = {}
        for r in graded:
            by_category.setdefault(r["category"], []).append((r["correct"], r["retrieval_hit"]))
        print("\nBy category       accuracy   recall")
        for category, results in sorted(by_category.items()):
            hits = [h for _, h in results if h != ""]
            recall_text = f"{sum(hits) / len(hits):.1%}" if hits else "-"
            print(f"  {category:<14}{sum(c for c, _ in results) / len(results):>9.1%}{recall_text:>9}")

    if points:
        print("\nEstimated points (the grade uses the private set):")
        for task in POINTS:
            if task in points:
                print(f"  Task {task:<4}{points[task]:6.2f} / {POINTS[task]}")
        print(f"  Total    {sum(points.values()):6.2f} / {sum(POINTS[t] for t in points)}")

    print(f"\nYour usage: {USAGE.summary()}")
    print(f"Generator usage: {GENERATOR_USAGE.summary()}")
    print(f"Time: {total_seconds:.1f}s (index {index_seconds:.1f}s, retrieval {retrieval_seconds:.1f}s)")
    if total_seconds > TIME_LIMIT_SECONDS:
        print(f"WARNING: the run took longer than the {TIME_LIMIT_SECONDS // 60}-minute limit.")


if __name__ == "__main__":
    main()
