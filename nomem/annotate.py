"""Deterministic, LLM-free annotation of a message.

Every function here is a rule, a pattern or a hash: no model, no network, no
randomness. The output is stored *beside* the verbatim message (never instead
of it), so a wrong annotation can cost a little ranking quality but can never
corrupt or lose what the user actually said.

- :func:`entities`    capitalised spans, acronyms, numbers with units
- :func:`keyphrases`  RAKE-style content-word phrases
- :func:`parse_times` "yesterday", "two weeks ago", "7 May 2023", "last Friday"
- :func:`extract_slots` first-person facts as (subject, attribute, value)
- :func:`importance`  a feature score in [0, 1]
- :func:`simhash` / :func:`norm_hash`  near-duplicate and exact-duplicate keys
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

# ---------------------------------------------------------------------------
# Tokenising

STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can can't cannot could couldn't did
didn't do does doesn't doing don't down during each few for from further get got
had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers
herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's
its itself just let's me more most mustn't my myself no nor not now of off on once
only or other ought our ours ourselves out over own really same shan't she she'd
she'll she's should shouldn't so some such than that that's the their theirs them
themselves then there there's these they they'd they'll they're they've this those
through to too under until up us very was wasn't we we'd we'll we're we've were
weren't what what's when when's where where's which while who who's whom why why's
will with won't would wouldn't yeah yes you you'd you'll you're you've your yours
yourself yourselves also oh ok okay hey hi hello thanks thank wow lol well like
going go went gonna sure great good nice cool awesome thing things lot lots much
many something anything everything one ones did do does done make made know think
""".split())

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def words(text: str) -> list[str]:
    return _WORD.findall(text)


def content_terms(text: str) -> list[str]:
    """Lower-cased non-stopword terms."""
    return [w.lower() for w in words(text) if w.lower() not in STOPWORDS and len(w) > 1]


def sentences(text: str) -> list[str]:
    return [s for s in _SENT_SPLIT.split(text.strip()) if s]


# ---------------------------------------------------------------------------
# Entities

_COMMON_CAPS = frozenset("""
i i'm i've i'll i'd the a an and but so oh hey hi hello yes no yeah wow thanks
thank that this it what when where who why how my we you they he she just
really well ok okay sure great good nice cool awesome congrats congratulations
monday tuesday wednesday thursday friday saturday sunday
january february march april may june july august september october november december
""".split())
_CAP_SPAN = re.compile(r"\b([A-Z][a-zA-Z'\-]+(?:\s+(?:of|the|de|van|von|&)?\s*[A-Z][a-zA-Z'\-]+)*)")
_ACRONYM = re.compile(r"\b[A-Z]{2,}[A-Z0-9+]*\b")
_NUMBER = re.compile(r"(?:[$£€]\s?)?\b\d[\d,.]*\s?(?:%|k|m|bn|dollars|euros|pounds|years?|kg|km|miles)?\b")
_SPEAKER_PREFIX = re.compile(r"^\s*([A-Z][\w'\-]*(?: [A-Z][\w'\-]*)?):\s+")


def entities(text: str) -> list[str]:
    """Named-thing candidates, lower-cased and de-duplicated, in order."""
    out: list[str] = []
    body = _SPEAKER_PREFIX.sub("", text)
    for span in _CAP_SPAN.findall(body):
        span = span.strip(" '")
        first = span.split()[0].lower()
        if first in _COMMON_CAPS and len(span.split()) == 1:
            continue
        if first in _COMMON_CAPS:
            span = " ".join(span.split()[1:])
        if span and span.lower() not in STOPWORDS:
            out.append(span.lower())
    out += [a.lower() for a in _ACRONYM.findall(body)]
    seen: set[str] = set()
    return [e for e in out if not (e in seen or seen.add(e))]


def numbers(text: str) -> list[str]:
    return [n.strip() for n in _NUMBER.findall(text) if n.strip()]


# ---------------------------------------------------------------------------
# Keyphrases (RAKE-style: runs of content words between stopwords/punctuation)

def keyphrases(text: str, max_words: int = 3, limit: int = 8) -> list[str]:
    phrases: list[str] = []
    run: list[str] = []
    for tok in re.findall(r"[A-Za-z0-9'\-]+|[^\sA-Za-z0-9]", _SPEAKER_PREFIX.sub("", text)):
        low = tok.lower()
        if not tok[0].isalnum() or low in STOPWORDS or len(low) < 2:
            if run:
                phrases.append(" ".join(run[:max_words]))
            run = []
        else:
            run.append(low)
    if run:
        phrases.append(" ".join(run[:max_words]))
    # longer phrases first (more specific), then keep order
    seen: set[str] = set()
    uniq = [p for p in phrases if not (p in seen or seen.add(p))]
    uniq.sort(key=lambda p: -len(p.split()))
    return uniq[:limit]


# ---------------------------------------------------------------------------
# Time expressions -> (start, end) epoch ranges, anchored to the message time

_MONTHS = {m: i for i, m in enumerate(
    "january february march april may june july august september october november december".split(), 1)}
_MONTHS.update({m[:3]: i for m, i in list(_MONTHS.items())})
_MONTHS["sept"] = 9
_WEEKDAYS = {d: i for i, d in enumerate("monday tuesday wednesday thursday friday saturday sunday".split())}
_NUMWORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10, "few": 3, "couple of": 2, "couple": 2,
             "several": 4}
_MONTH_RE = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_UNIT_RE = r"(day|week|month|year|weekend)s?"


def _day(dt: datetime) -> tuple[float, float]:
    start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return start.timestamp(), (start + timedelta(days=1)).timestamp() - 1


def _month(y: int, m: int) -> tuple[float, float]:
    start = datetime(y, m, 1)
    end = datetime(y + (m == 12), m % 12 + 1, 1)
    return start.timestamp(), end.timestamp() - 1


def _span(unit: str, anchor: datetime) -> tuple[float, float]:
    if unit == "day":
        return _day(anchor)
    if unit in ("week", "weekend"):
        start = anchor - timedelta(days=anchor.weekday())
        if unit == "weekend":
            start += timedelta(days=5)
            return _day(start)[0], _day(start + timedelta(days=1))[1]
        return _day(start)[0], _day(start + timedelta(days=6))[1]
    if unit == "month":
        return _month(anchor.year, anchor.month)
    return datetime(anchor.year, 1, 1).timestamp(), datetime(anchor.year + 1, 1, 1).timestamp() - 1


def _shift(anchor: datetime, unit: str, n: int) -> datetime:
    if unit in ("day",):
        return anchor + timedelta(days=n)
    if unit in ("week", "weekend"):
        return anchor + timedelta(weeks=n)
    if unit == "month":
        m = anchor.month - 1 + n
        return anchor.replace(year=anchor.year + m // 12, month=m % 12 + 1, day=1)
    return anchor.replace(year=anchor.year + n, month=1, day=1)


def parse_times(text: str, ref: float) -> list[tuple[float, float]]:
    """Time ranges a message talks about, resolved against its timestamp ``ref``."""
    t = text.lower()
    now = datetime.fromtimestamp(ref)
    out: list[tuple[float, float]] = []

    for word, delta in (("yesterday", -1), ("last night", -1), ("today", 0), ("tonight", 0),
                        ("this morning", 0), ("tomorrow", 1)):
        if re.search(rf"\b{word}\b", t):
            out.append(_day(now + timedelta(days=delta)))

    num = r"(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|few|couple of|couple|several)"
    for n, unit in re.findall(rf"\b{num}\s+{_UNIT_RE}\s+ago\b", t):
        k = int(n) if n.isdigit() else _NUMWORDS[n]
        out.append(_span(unit, _shift(now, unit, -k)))

    for which, unit in re.findall(rf"\b(last|this|next|past)\s+{_UNIT_RE}\b", t):
        k = {"last": -1, "past": -1, "this": 0, "next": 1}[which]
        out.append(_span(unit, _shift(now, unit, k)))

    for which, day in re.findall(r"\b(last|this|next|on)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", t):
        diff = _WEEKDAYS[day] - now.weekday()
        if which in ("last", "on"):
            diff = diff - 7 if diff >= 0 else diff
        elif which == "next":
            diff = diff + 7 if diff <= 0 else diff
        out.append(_day(now + timedelta(days=diff)))

    # "7 May 2023", "7th of May", "May 7, 2023", "May 7th"
    for d, m, y in re.findall(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH_RE}\b,?\s*(\d{{4}})?", t):
        out.append(_safe_day(y or now.year, _MONTHS[m[:4] if m.startswith("sept") else m[:3]], d))
    for m, d, y in re.findall(rf"\b{_MONTH_RE}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b,?\s*(\d{{4}})?", t):
        out.append(_safe_day(y or now.year, _MONTHS[m[:4] if m.startswith("sept") else m[:3]], d))
    # "in May 2023", "in May", "May 2023"
    for m, y in re.findall(rf"\b(?:in|during|since|until)\s+{_MONTH_RE}\b(?:\s+(\d{{4}}))?", t):
        out.append(_month(int(y) if y else now.year, _MONTHS[m[:4] if m.startswith("sept") else m[:3]]))
    for y in re.findall(r"\b(?:in|since|during)\s+((?:19|20)\d{2})\b", t):
        out.append((datetime(int(y), 1, 1).timestamp(), datetime(int(y) + 1, 1, 1).timestamp() - 1))

    seen: set = set()
    return [r for r in out if r and not (r in seen or seen.add(r))]


def _safe_day(y, m: int, d) -> Optional[tuple[float, float]]:
    try:
        return _day(datetime(int(y), m, int(d)))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Slot facts: first-person statements -> (subject, attribute, value)

@dataclass(frozen=True)
class Slot:
    subject: str
    attribute: str
    value: str
    multi: bool = False      # several values may hold at once (likes, allergies)
    negated: bool = False    # "I no longer like X" closes a "likes: X" slot


_VALUE = r"([^.,;!?\n]{2,60}?)(?=\s+(?:and|but|because|since|so|which|who|now|anymore|these days|nowadays|at|for|with|last|this|next|yesterday|today|tomorrow|recently|ago|on|in \d{4})\b|[.,;!?\n]|$)"
_SINGLE_PATTERNS = [
    (r"\bmy name is\s+" + _VALUE, "name"),
    (r"\b(?:i'm|i am)\s+(\d{1,3})\s+years?\s+old\b", "age"),
    (r"\bi\s+(?:now\s+)?(?:live|am living|'m living|reside)\s+in\s+" + _VALUE, "location"),
    (r"\bi(?:'ve| have)?\s+(?:just\s+)?(?:moved|relocated)\s+to\s+" + _VALUE, "location"),
    (r"\bi(?:'m| am)\s+(?:now\s+)?(?:based|living)\s+in\s+" + _VALUE, "location"),
    (r"\bi\s+(?:now\s+)?work\s+(?:as)\s+(?:an?\s+)?" + _VALUE, "job"),
    (r"\bi(?:'m| am)\s+(?:now\s+)?(?:working\s+)?as\s+(?:an?\s+)?" + _VALUE, "job"),
    (r"\bmy job is\s+(?:an?\s+)?" + _VALUE, "job"),
    (r"\bi\s+(?:now\s+)?work(?:\s+as\s+(?:an?\s+)?[a-z]+)?\s+(?:at|for)\s+" + _VALUE, "employer"),
    (r"\bi(?:'m| am)\s+(?:a\s+)?(vegetarian|vegan|pescatarian|keto)\b", "diet"),
    (r"\bmy\s+(?:new\s+)?(?:favou?rite\s+)?([a-z]+(?:\s[a-z]+)?)\s+is\s+(?:called\s+|named\s+)?" + _VALUE, None),
]
_MULTI_PATTERNS = [
    (r"\bi(?:'m| am)\s+(?:severely\s+|very\s+|really\s+|mildly\s+)?allergic\s+to\s+" + _VALUE, "allergy"),
    (r"\bi\s+(?:really\s+|absolutely\s+)?(?:love|like|enjoy|adore|prefer)\s+" + _VALUE, "likes"),
    (r"\bi\s+(?:really\s+)?(?:hate|dislike|can't stand|cannot stand)\s+" + _VALUE, "dislikes"),
]
_NEG_LIKE = re.compile(r"\bi\s+(?:don't|do not|no longer|don't really|stopped)\s+(?:like|love|enjoy|liking|eating|drinking)\s+" + _VALUE)
_NEG_BEFORE = re.compile(r"\b(?:not|n't|never|no longer)\b[^.]{0,12}$")
_GENERIC_ATTR_BLOCK = frozenset("""
day week life time point goal question guess bad fault pleasure turn idea plan way
thing problem issue dream hope pick favorite favourite god heart mind work job name
""".split())


def _clean_value(v: str) -> str:
    v = v.strip(" '\"")
    v = re.sub(r"^(?:the|a|an|my)\s+", "", v, flags=re.I)
    return v


def speaker_of(text: str) -> Optional[str]:
    """``"Caroline: ..."`` -> ``"Caroline"`` (transcripts that prefix the speaker)."""
    m = _SPEAKER_PREFIX.match(text)
    return m.group(1) if m else None


def extract_slots(text: str, subject: str = "user") -> list[Slot]:
    """First-person facts. Conservative by design: precision over recall."""
    body = _SPEAKER_PREFIX.sub("", text)
    out: list[Slot] = []
    for sent in sentences(body):
        low = sent.lower()
        if low.rstrip().endswith("?"):
            continue  # questions are not statements of fact
        for m in _NEG_LIKE.finditer(low):
            out.append(Slot(subject, "likes", _clean_value(m.group(1)), multi=True, negated=True))
        for pattern, attr in _SINGLE_PATTERNS:
            for m in re.finditer(pattern, low):
                if _NEG_BEFORE.search(low[: m.start() + 6]):
                    continue
                if attr is None:
                    a, v = m.group(1).strip(), m.group(2)
                    if a in _GENERIC_ATTR_BLOCK or a in STOPWORDS:
                        continue
                    attr_name = a.replace(" ", "_")
                else:
                    attr_name, v = attr, m.group(1)
                v = _clean_value(_restore_case(sent, v))
                if v and len(v.split()) <= 6:
                    out.append(Slot(subject, attr_name, v))
        for pattern, attr in _MULTI_PATTERNS:
            for m in re.finditer(pattern, low):
                if _NEG_BEFORE.search(low[max(0, m.start() - 12): m.start() + 2]):
                    continue
                v = _clean_value(_restore_case(sent, m.group(1)))
                if v and len(v.split()) <= 6:
                    out.append(Slot(subject, attr, v, multi=True))
    seen: set = set()
    return [s for s in out if not ((s.attribute, s.value.lower(), s.negated) in seen
                                   or seen.add((s.attribute, s.value.lower(), s.negated)))]


def _restore_case(original: str, lowered_value: str) -> str:
    i = original.lower().find(lowered_value)
    return original[i:i + len(lowered_value)] if i >= 0 else lowered_value


# ---------------------------------------------------------------------------
# Importance (replaces the "rate this memory 1-10" LLM call)

_CUES = re.compile(r"\b(remember|important|never|always|allergic|allergy|birthday|anniversary|"
                   r"deadline|password|address|diagnosed|married|divorced|pregnant|promoted|"
                   r"moved|started|quit|graduated|passed away|died|born|favou?rite|prefer)\b", re.I)
_FILLER = re.compile(r"^\W*(ok(ay)?|thanks?( you)?|thank you|nice|cool|great|sure|yes|no|yeah|lol|"
                     r"haha|wow|hi|hello|hey|bye|got it|sounds good)\W*$", re.I)
_FIRST_PERSON = re.compile(r"\b(i|i'm|i've|i'd|my|me|mine)\b", re.I)


def importance(text: str, slots: Optional[list[Slot]] = None, ents: Optional[list[str]] = None) -> float:
    body = _SPEAKER_PREFIX.sub("", text).strip()
    if not body or _FILLER.match(body):
        return 0.0
    score = 0.2
    if _FIRST_PERSON.search(body):
        score += 0.1
    if slots:
        score += 0.3
    if _CUES.search(body):
        score += 0.2
    score += min(0.15, 0.05 * len(ents or []))
    if numbers(body):
        score += 0.05
    if body.endswith("?"):
        score -= 0.1
    n = len(words(body))
    if n < 4:
        score -= 0.1
    return round(max(0.0, min(1.0, score)), 3)


# ---------------------------------------------------------------------------
# Duplicate keys

def norm_hash(text: str) -> str:
    norm = " ".join(w.lower() for w in words(_SPEAKER_PREFIX.sub("", text)))
    return hashlib.blake2b(norm.encode(), digest_size=8).hexdigest()


def simhash(text: str, bits: int = 64) -> int:
    """64-bit SimHash over word 2-shingles (signed, so it fits SQLite INTEGER)."""
    toks = [w.lower() for w in words(_SPEAKER_PREFIX.sub("", text))]
    shingles = [" ".join(toks[i:i + 2]) for i in range(max(1, len(toks) - 1))] or [""]
    acc = [0] * bits
    for sh in shingles:
        h = int.from_bytes(hashlib.blake2b(sh.encode(), digest_size=8).digest(), "big")
        for b in range(bits):
            acc[b] += 1 if (h >> b) & 1 else -1
    v = sum(1 << b for b in range(bits) if acc[b] > 0)
    return v - (1 << 64) if v >= (1 << 63) else v


def hamming(a: int, b: int) -> int:
    return bin((a ^ b) & ((1 << 64) - 1)).count("1")


# ---------------------------------------------------------------------------

@dataclass
class Annotation:
    entities: list[str]
    keyphrases: list[str]
    times: list[tuple[float, float]]
    slots: list[Slot]
    importance: float
    norm_hash: str
    simhash: int


def annotate(text: str, ref_time: float, subject: str = "user") -> Annotation:
    ents = entities(text)
    sl = extract_slots(text, subject)
    return Annotation(
        entities=ents,
        keyphrases=keyphrases(text),
        times=parse_times(text, ref_time),
        slots=sl,
        importance=importance(text, sl, ents),
        norm_hash=norm_hash(text),
        simhash=simhash(text),
    )
