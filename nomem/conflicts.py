"""LLM-free contradiction detection between memories.

When something the user says now disagrees with something they said before
("my flight is at 9am" ... "actually the flight moved to 11am"), NoMem does
not pick a winner or rewrite either memory. It records a *conflict*: a link
from the older memory to the newer one, with the kind of evidence, the
values on each side and a confidence. At read time both memories are shown
with their dates and a conflict marker, and the agent's own model decides.

Three detectors, cheapest first, all deterministic:

``slot``   A single-valued slot fact (``location``, ``job``, ``address`` ...)
           got a new value, or a multi-valued one was negated ("I no longer
           like sushi"). Highest confidence: the rules already know the two
           statements are about the same attribute.
``frame``  The same sentence frame with different values: "I have 2 kids" /
           "I have 3 kids", "The meeting is at 3pm" / "The meeting is at
           5pm". Values (numbers, times, dates, names) are masked out and the
           rest of the sentence is hashed, so finding earlier sentences with
           the same frame is one indexed lookup.
``update`` A sentence with an update cue ("moved", "rescheduled", "now",
           "no longer", "actually", "instead" ...) that shares a topic word
           with an earlier sentence by the same speaker and carries a
           different value of the same type. Candidates come from the
           keyword index and, when an embedder is configured, from vector
           similarity, which also raises the confidence.

Recency decides direction only: the later memory is marked as the update,
the earlier as possibly outdated. Nothing is deleted or hidden.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from . import annotate as A

# ---------------------------------------------------------------------------
# Cues and vocabularies

UPDATE_CUE = re.compile(
    r"\b(now|no longer|not anymore|anymore|any more|actually|instead|changed|change|"
    r"switched|switching|moved|moving|relocated|rescheduled|postponed|pushed back|"
    r"brought forward|bumped|cancell?ed|updated?|correction|turns out|"
    r"i meant|scratch that|used to|from now on|these days|nowadays)\b",
    re.I,
)
# Cues that say outright "the earlier statement is wrong or out of date".
STRONG_CUE = re.compile(
    r"\b(no longer|not anymore|anymore|actually|instead|changed|switched|moved|relocated|"
    r"rescheduled|postponed|pushed back|brought forward|cancell?ed|updated?|correction|"
    r"i meant|scratch that|from now on|now)\b",
    re.I,
)
_CUE_WORDS = frozenset("""
now longer anymore any more actually instead changed change switched switching moved
moving relocated rescheduled postponed pushed back brought forward bumped cancelled
canceled update updated correction new newly turns meant scratch used nowadays days
""".split())

# Verbs that state a current fact, not an event: a frame needs one of these
# to be compared ("I ran 5 km" twice is two runs, not a contradiction).
_STATIVE = frozenset("""
is are am 'm 're 's have has own owns live lives work works earn earns weigh weighs
cost costs drive drives use uses leaves leave starts start begins begin ends end due
""".split())

# Topic words too vague to anchor two sentences to the same fact.
_WEAK_ANCHORS = A.STOPWORDS | _CUE_WORDS | frozenset("""
time today tomorrow yesterday week weeks month months year years day days morning
evening night tonight weekend am pm i'm i've it's that's really totally next last
soon ago still always sometimes finally able hope look looks looking makes love
happy sounds sound great fun awesome glad keep trying find perfect lot bit way
""".split())
_GREETING = re.compile(r"\b(?:hey|hi|hello|thanks|thank you|wow|congrats|congratulations|sure|yeah|"
                       r"oh|ok|okay|bye|dear|cheers|sorry|yo|aw|aww|omg|haha|good luck|go)\W*$", re.I)
_CLAUSE_SPLIT = re.compile(r"\s*(?:[,;]|\s-\s|\s—\s|\bbut\b)\s*")


def clauses(sent: str) -> list[str]:
    return [c for c in _CLAUSE_SPLIT.split(sent) if c and c.strip()]

_TIME_OF_DAY = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s?(am|pm|a\.m\.|p\.m\.)(?![a-z])|\b(\d{1,2}):(\d{2})\b", re.I)
_MONEY_OR_NUM = re.compile(r"(?:[$£€]\s?)?\b\d[\d,]*(?:\.\d+)?\b(?:\s?(?:%|k\b|m\b|bn\b))?")
_DATE_WORDS = re.compile(
    r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?)?" + A._MONTH_RE + r"\b(?:\s+\d{1,2}(?:st|nd|rd|th)?)?(?:,?\s*\d{4})?"
    r"|\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?\b",
    re.I,
)
# A weekday that names a schedule ("on Wednesday", "Thursdays", "every
# Sunday"), not a past day ("last Friday"): compared by name, never dated.
_SCHEDULE_DAY = re.compile(
    r"(?<!\blast )(?<!\bthis )(?<!\bnext )\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?\b", re.I)


def stem(word: str) -> str:
    """Crude singular form, so "banks" meets "bank" and "dogs" meets "dog"."""
    w = word.lower()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("ches", "shes", "sses", "xes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is", "'s")):
        return w[:-1]
    return w


# ---------------------------------------------------------------------------
# Values: the parts of a sentence that can change while the topic stays

@dataclass
class Values:
    num: set = field(default_factory=set)
    time: set = field(default_factory=set)
    day: set = field(default_factory=set)
    date: set = field(default_factory=set)
    ent: set = field(default_factory=set)

    def kinds(self) -> dict:
        return {"num": self.num, "time": self.time, "day": self.day, "date": self.date, "ent": self.ent}

    def any(self) -> bool:
        return bool(self.num or self.time or self.day or self.date or self.ent)

    def update(self, other: "Values") -> None:
        for k, v in other.kinds().items():
            self.kinds()[k].update(v)


def _norm_time(m: re.Match) -> str:
    if m.group(3):
        h, mi, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3).lower()[0]
        h = h % 12 + (12 if ap == "p" else 0)
    else:
        h, mi = int(m.group(4)), int(m.group(5))
    return f"{h:02d}:{mi:02d}"


def sentence_values(sent: str, ref: float) -> Values:
    """Values in a sentence, clause by clause (see :func:`clause_values`)."""
    v = Values()
    for c in clauses(sent):
        v.update(clause_values(c, ref))
    return v


def clause_values(sent: str, ref: float) -> Values:
    """Numbers, times of day, upcoming dates and names in one clause.

    Only dates from the day before ``ref`` onwards count: a plan can be
    moved, a past event cannot. A capitalised word that opens the clause
    or follows a greeting ("Wow, Dave", "Hey John") is not a value.
    """
    v = Values()
    rest = sent
    for m in _TIME_OF_DAY.finditer(sent):
        v.time.add(_norm_time(m))
    rest = _TIME_OF_DAY.sub(" ", rest)
    for m in _SCHEDULE_DAY.finditer(rest):
        v.day.add(m.group(1).lower())
    rest = _SCHEDULE_DAY.sub(" ", rest)
    for a, b in A.parse_times(rest, ref):
        if b < ref:
            continue
        v.date.add(datetime.fromtimestamp(a).strftime("%Y-%m-%d") + (
            "" if b - a < 86400 else "/" + datetime.fromtimestamp(b).strftime("%Y-%m-%d")))
    rest = _DATE_WORDS.sub(" ", rest)
    for n in _MONEY_OR_NUM.findall(rest):
        n = n.replace(",", "").replace(" ", "").lower()
        if n:
            v.num.add(n)
    low = rest.lower()
    for e in A.entities(rest):
        if any(ch.isdigit() for ch in e) or any(w in A.STOPWORDS or w in _CUE_WORDS for w in e.split()):
            continue
        i = low.find(e)
        before = low[:i].strip()
        if i < 0 or not before or not re.search(r"[a-z0-9]", before) or _GREETING.search(before):
            continue
        v.ent.add(e)
    return v


def values_conflict(old: Values, new: Values, strong: bool = False) -> Optional[tuple[str, str, str]]:
    """``(kind, old_values, new_values)`` when the same kind of value changed.

    Normally both sides need a value the other lacks. With a ``strong`` cue
    ("correction: it's Muffin, Biscuit was my old cat") a new value is
    enough even when the old one is mentioned again.
    """
    for kind in ("time", "day", "date", "num", "ent"):
        a, b = old.kinds()[kind], new.kinds()[kind]
        if a and b and (b - a) and ((a - b) or strong):
            return kind, ", ".join(sorted(a - b) or sorted(a)), ", ".join(sorted(b - a))
    return None


# ---------------------------------------------------------------------------
# Frames: a sentence with its values masked, hashed

_MASK = re.compile(r"\x00+(?:\s+\x00+)*")


def frame(sent: str) -> Optional[str]:
    """A hashable frame for a stative sentence that carries a value, else None."""
    body = A._SPEAKER_PREFIX.sub("", sent).strip()
    if not body or body.rstrip().endswith("?"):
        return None
    masked = _TIME_OF_DAY.sub("\x00", body)
    masked = _DATE_WORDS.sub("\x00", masked)
    masked = _MONEY_OR_NUM.sub("\x00", masked)
    # Capitalised spans past the first word are names/places.
    masked = re.sub(r"(?<=[\w,;:] )[A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+)*", "\x00", masked)
    if "\x00" not in masked:
        return None
    toks = [t if t == "\x00" else stem(t) for t in re.findall(r"\x00|[a-z][a-z']*", masked.lower())
            if t not in _CUE_WORDS]
    toks = [t for t in toks if t not in {"so", "well", "oh", "hey", "btw", "just", "also", "still"}]
    collapsed: list[str] = []
    for t in toks:
        if t == "\x00" and collapsed and collapsed[-1] == "\x00":
            continue
        collapsed.append(t)
    words = [t for t in collapsed if t != "\x00"]
    if len(collapsed) < 3 or not (set(words) & _STATIVE):
        return None
    content = [w for w in words if w not in A.STOPWORDS and w not in _STATIVE]
    if not content:
        return None
    key = " ".join("<v>" if t == "\x00" else t for t in collapsed)
    return hashlib.blake2b(key.encode(), digest_size=8).hexdigest()


_PEOPLE = frozenset("""
friend friends sister sisters brother brothers mom mum mother dad father parents wife husband
partner son daughter kids children boss manager colleague coworker neighbour neighbor roommate
flatmate cousin aunt uncle grandma grandmother grandpa grandfather niece nephew he she they his
her their him them girlfriend boyfriend fiance fiancee baby teacher doctor
""".split())


def people(sent: str) -> set[str]:
    """Who else a sentence is about ("my friend", "she"): a fact about
    someone else never contradicts a fact about the speaker."""
    return {w.lower() for w in A.words(sent) if w.lower() in _PEOPLE}


_OWN = re.compile(r"\b(?:i|i'm|i've|my|we|we're|our)\b", re.I)


def _plain_statement(text: str) -> bool:
    """Short, first-person, not a question or an exclamation of feeling."""
    body = A._SPEAKER_PREFIX.sub("", text).strip()
    ws = A.words(body)
    return (0 < len(ws) <= 10 and not body.endswith(("?", "!")) and bool(_OWN.search(body))
            and not ({w.lower() for w in ws} & _FEELING))


_FEELING = frozenset("""
feel feels felt love loving loved hope wish think thought glad happy sad excited thanks thank
appreciate miss wait can't cannot want wanna gonna sure agree right definitely totally
""".split())


_KEEP_PREPS = frozenset("by from at in on to with for".split())


def _seq(text: str) -> list[str]:
    """Content words plus the prepositions that frame them, in order, stemmed.

    Capitalised words are left out: names here are greetings ("Hey James"),
    and real names are values the other detectors compare.
    """
    out = []
    for w in A.words(A._SPEAKER_PREFIX.sub("", text)):
        low = w.lower()
        if w[0].isupper() and low not in ("i", "i'm", "i've"):
            continue
        if low in _CUE_WORDS or any(ch.isdigit() for ch in low):
            continue
        if low in _KEEP_PREPS or (low not in A.STOPWORDS and low not in _WEAK_ANCHORS and len(low) > 1):
            out.append(stem(low))
    return out


def term_swap(old: str, new_clauses: list[str], shared: set[str]) -> Optional[tuple[str, str, str]]:
    """One short phrase replaced in an otherwise identical short statement.

    "My commute is by car" / "I commute by train now instead" line up as
    ``commute by [car]`` / ``commute by [train]``. Both statements must be
    short, plain and first-person, the words around the swap must include
    the shared topic word, and the swapped phrases are one or two words.
    """
    if not _plain_statement(old):
        return None
    a = _seq(old)
    for c in new_clauses:
        if not _plain_statement(c) or re.search(r"\b(?:not|no|never)\b|n't\b", c, re.I):
            continue
        b = _seq(c)
        if not (2 <= len(a) <= 5 and 2 <= len(b) <= 5):
            continue
        p = 0
        while p < min(len(a), len(b)) and a[p] == b[p]:
            p += 1
        q = 0
        while q < min(len(a), len(b)) - p and a[-1 - q] == b[-1 - q]:
            q += 1
        kept = a[:p] + a[len(a) - q:]
        gone, came = a[p:len(a) - q], b[p:len(b) - q]
        if not (set(kept) & shared) or not (1 <= len(gone) <= 2 and 1 <= len(came) <= 2):
            continue
        if set(gone) & set(came) or all(w in _KEEP_PREPS for w in gone + came):
            continue
        return "term", " ".join(gone), " ".join(came)
    return None


def anchors(sent: str, vals: Values) -> set[str]:
    """Topic words: content terms that are not values, cues or filler."""
    ent_words = {w for e in vals.ent for w in e.split()}
    return {stem(t) for t in A.content_terms(A._SPEAKER_PREFIX.sub("", sent))
            if t not in _WEAK_ANCHORS and t not in ent_words and not any(ch.isdigit() for ch in t)
            and len(t) > 2}


# ---------------------------------------------------------------------------

@dataclass
class Conflict:
    older_id: int
    newer_id: int
    kind: str              # "slot" | "frame" | "update"
    detail: str            # e.g. "location: Pune -> Bangalore"
    confidence: float
    attribute: Optional[str] = None


@dataclass
class ConflictConfig:
    """Knobs for the write-time detectors."""

    enabled: bool = True
    frame: bool = True
    update: bool = True
    candidates: int = 20          # earlier messages checked per update sentence
    min_confidence: float = 0.6
    vector_min_similarity: float = 0.55  # cosine that counts as "same topic"
    max_per_message: int = 3
    # Read side: how conflicts are shown on retrieved memories.
    show_min_confidence: float = 0.6
    show_per_memory: int = 2      # conflict markers per memory (0 hides them)
    quote_chars: int = 90         # longest quote of the other memory


def speaker_key(role: str, content: str, metadata: Optional[dict]) -> str:
    sp = (metadata or {}).get("speaker") or A.speaker_of(content)
    return f"{role}:{sp or ''}"


def detect(store, user_id: str, session_id: str, msg, speaker: str,
           config: ConflictConfig, embedder=None) -> list[Conflict]:
    """Frame and update-cue conflicts between ``msg`` and earlier messages."""
    out: list[Conflict] = []
    seen: set[int] = set()
    sents = A.sentences(A._SPEAKER_PREFIX.sub("", msg.content))
    frames_to_add: list[tuple[str, str]] = []
    for sent in sents:
        if sent.rstrip().endswith("?"):
            continue
        vals = sentence_values(sent, msg.created_at)
        if not vals.any() and not STRONG_CUE.search(sent):
            continue
        # -- frame: identical frame, different value --------------------
        # Frames are per clause: "I have 3 kids now, the baby arrived!"
        # still matches "I have 2 kids".
        for part in (clauses(sent) if config.frame else []):
            fh = frame(part)
            if not fh:
                continue
            frames_to_add.append((fh, part))
            pvals = sentence_values(part, msg.created_at)
            for row in store.frames_like(user_id, fh, speaker, msg.id):
                if row["message_id"] in seen:
                    continue
                old = sentence_values(row["sentence"], row["created_at"])
                diff = values_conflict(old, pvals)
                if diff:
                    conf = 0.7 + (0.15 if UPDATE_CUE.search(sent) else 0.0)
                    out.append(Conflict(row["message_id"], msg.id, "frame",
                                        f"{diff[1]} -> {diff[2]}", round(conf, 2)))
                    seen.add(row["message_id"])
                    break  # the most recent earlier statement is the one updated
        # -- update cue: shared topic, different value ------------------
        if config.update and UPDATE_CUE.search(sent):
            strong = bool(STRONG_CUE.search(sent))
            # The changed value must sit in the clause that carries the cue:
            # "the party starts at 9pm instead", not "Wow Dave, ... now".
            new_vals = Values()
            cue_clauses = [c for c in clauses(sent) if UPDATE_CUE.search(c)]
            for c in cue_clauses:
                new_vals.update(clause_values(c, msg.created_at))
            anc = anchors(sent, vals)
            # With no value at all, only a strong cue can carry a change of
            # wording ("I commute by train now instead").
            if not anc or not (new_vals.any() or strong):
                continue
            who = people(sent)
            cand = store.fts_ranked(user_id, session_id, " ".join(sorted(anc)), config.candidates)
            sims: dict[int, float] = {}
            if embedder is not None:
                try:
                    near = embedder.similar(store, user_id, session_id, sent, config.candidates)
                except Exception:  # keyword candidates still work without the model
                    near = []
                for i, s in near:
                    sims[i] = s
                    if i not in cand:
                        cand.append(i)
            cand = [i for i in cand if i != msg.id and i not in seen]
            if not cand:
                continue
            msgs = store.get_messages(cand)
            best: Optional[tuple[float, float, Conflict]] = None
            for i in cand:
                m = msgs.get(i)
                if m is None or m.created_at > msg.created_at or (m.created_at == msg.created_at and m.id > msg.id):
                    continue
                if speaker_key(m.role, m.content, m.metadata) != speaker:
                    continue
                for osent in A.sentences(A._SPEAKER_PREFIX.sub("", m.content)):
                    if osent.rstrip().endswith("?"):
                        continue
                    ovals = sentence_values(osent, m.created_at)
                    shared = anc & anchors(osent, ovals)
                    if not shared or people(osent) != who:
                        continue
                    diff = values_conflict(ovals, new_vals, strong)
                    if diff:
                        conf = 0.4 + 0.1 * min(3, len(shared)) + (0.1 if strong else 0.0)
                    elif strong:
                        diff = term_swap(osent, cue_clauses, shared)
                        if not diff:
                            continue
                        conf = 0.6
                    else:
                        continue
                    sim = sims.get(i)
                    if sim is not None:
                        conf += 0.1 if sim >= config.vector_min_similarity else 0.0
                    conf = round(min(conf, 0.95), 2)
                    if conf < config.min_confidence:
                        continue
                    c = Conflict(i, msg.id, "update",
                                 f"{'/'.join(sorted(shared))}: {diff[1]} -> {diff[2]}", conf)
                    key = (conf, m.created_at)  # ties go to the latest earlier statement
                    if best is None or key > best[:2]:
                        best = (conf, m.created_at, c)
            if best is not None:
                out.append(best[2])
                seen.add(best[2].older_id)
    for fh, sent in frames_to_add:
        store.add_frame(user_id, session_id, fh, msg.id, speaker, sent, msg.created_at)
    out.sort(key=lambda c: -c.confidence)
    return out[: config.max_per_message]


# Attributes with their own extraction rule; any other attribute comes from
# the generic "my <thing> is <value>" pattern, which is noisier.
KNOWN_ATTRIBUTES = frozenset("name age location job employer diet allergy likes dislikes".split())


def slot_conflict_confidence(attribute: str, old: str, new: Optional[str]) -> float:
    """How sure a superseded slot is a real update, not an extraction slip.

    "My address is 12 Baker Street" -> "my address is 5 Elm Road" is solid;
    "my family is my motivation" -> "my family is awesome" is two opinions.
    A generic attribute counts when both values look like values: a name, a
    number, or a short phrase with no filler words.
    """
    if attribute in KNOWN_ATTRIBUTES or new is None:
        return 0.9

    def value_like(v: str) -> bool:
        ws = A.words(v)
        if any(w[0].isupper() or w[0].isdigit() for w in ws):
            return True
        return 0 < len(ws) <= 2 and not any(w.lower() in A.STOPWORDS for w in ws)

    return 0.75 if value_like(old) and value_like(new) else 0.5


def snippet(text: str, limit: int = 90) -> str:
    """A short, verbatim quote of a memory for a conflict marker."""
    body = text.strip()
    if len(body) <= limit:
        return body
    cut = body[:limit].rsplit(" ", 1)[0]
    return cut + " ..."
