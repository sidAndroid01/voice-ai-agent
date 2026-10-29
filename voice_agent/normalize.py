"""Text -> spoken-form normalization applied to every LLM token stream before TTS.

TTS engines mispronounce structured text ("$45.50", "15/09/2026", "AB12345",
"order 45678"). The LLM is told to write in spoken form, but it does not always
comply, so this layer guarantees it. The transcript shown to the user is not
affected (only the TTS input is transformed).
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterable, AsyncIterator, Callable
from typing import Any

from num2words import num2words

_DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
_CURRENCIES = {
    "$": ("dollar", "dollars", "cent", "cents"),
    "USD": ("dollar", "dollars", "cent", "cents"),
    "€": ("euro", "euros", "cent", "cents"),
    "EUR": ("euro", "euros", "cent", "cents"),
    "£": ("pound", "pounds", "penny", "pence"),
    "GBP": ("pound", "pounds", "penny", "pence"),
    "₹": ("rupee", "rupees", "paisa", "paise"),
    "Rs.": ("rupee", "rupees", "paisa", "paise"),
    "Rs": ("rupee", "rupees", "paisa", "paise"),
    "INR": ("rupee", "rupees", "paisa", "paise"),
}
_SCALES = {
    "k": 1_000,
    "K": 1_000,
    "m": 1_000_000,
    "M": 1_000_000,
    "mn": 1_000_000,
    "million": 1_000_000,
    "b": 1_000_000_000,
    "B": 1_000_000_000,
    "bn": 1_000_000_000,
    "billion": 1_000_000_000,
    "lakh": 100_000,
    "crore": 10_000_000,
}
_UNIT_SUFFIX = re.compile(
    r"^\d+(?:\.\d+)?(kg|km|g|m|cm|mm|ml|l|gb|mb|tb|kb|hz|khz|mhz|ghz|x|s|ms|h|hr|hrs|min|mins|k|mp)$",
    re.I,
)
# Words that, right before a number, mean "read this digit by digit".
_ID_CONTEXT = re.compile(
    r"(?:\b(?:order|id|number|no\.?|code|account|acct|pin|otp|reference|ref|ticket|"
    r"tracking|phone|mobile|card|booking|confirmation|invoice|policy|case)\b|#)\s*(?:is\s*|:\s*)?$",
    re.I,
)


def _words(n: int | float, lang: str) -> str:
    try:
        return num2words(n, lang=lang)
    except NotImplementedError:
        return num2words(n, lang="en")


def spell_digits(digits: str, group: int = 3) -> str:
    """'4567890' -> 'four five six, seven eight nine zero' (pauses between groups)."""
    digits = re.sub(r"\D", "", digits)
    groups = [digits[i : i + group] for i in range(0, len(digits), group)]
    if len(groups) > 1 and len(groups[-1]) == 1:  # never leave a lone trailing digit
        tail = groups.pop()
        groups[-1] += tail
    return ", ".join(" ".join(_DIGIT_WORDS[int(d)] for d in g) for g in groups)


def spell_id(token: str) -> str:
    """'AB12-345' -> 'A B one two, three four five'."""
    parts = [p for p in re.split(r"[-_/\s]+", token) if p]
    out = []
    for part in parts:
        chars = [_DIGIT_WORDS[int(c)] if c.isdigit() else c.upper() for c in part]
        chunks = [" ".join(chars[i : i + 3]) for i in range(0, len(chars), 3)]
        if len(chunks) > 1 and len(chars) % 3 == 1:  # never leave a lone trailing char
            tail = chunks.pop()
            chunks[-1] += " " + tail
        out.append(", ".join(chunks))
    return ", ".join(out)


def _amount(value: str, scale: str | None, cur: tuple[str, str, str, str], lang: str) -> str:
    num = float(value.replace(",", ""))
    if scale:
        num *= _SCALES[scale]
        whole = int(round(num))
        return f"{_words(whole, lang)} {cur[1]}"
    whole = int(num)
    cents = int(round((num - whole) * 100))
    if cents == 100:
        whole, cents = whole + 1, 0
    s = f"{_words(whole, lang)} {cur[0] if whole == 1 else cur[1]}"
    if cents:
        s += f" and {_words(cents, lang)} {cur[2] if cents == 1 else cur[3]}"
    return s


def _date(d: int, m: int, y: int, lang: str) -> str | None:
    if not (1 <= m <= 12 and 1 <= d <= 31):
        return None
    year = num2words(y, to="year", lang="en") if 1000 <= y <= 2999 else _words(y, lang)
    return f"the {num2words(d, to='ordinal', lang='en')} of {_MONTHS[m - 1]}, {year}"


def _time(h: int, mnt: int, suffix: str | None) -> str | None:
    if h > 23 or mnt > 59:
        return None
    if suffix:
        ampm = "AM" if suffix.lower().startswith("a") else "PM"
        h12 = h % 12 or 12
    else:
        ampm = "AM" if h < 12 else "PM"
        h12 = h % 12 or 12
    if mnt == 0:
        mins = ""
    elif mnt < 10:
        mins = f" oh {_DIGIT_WORDS[mnt]}"
    else:
        mins = " " + num2words(mnt, lang="en")
    return f"{num2words(h12, lang='en')}{mins} {ampm}"


_URL = re.compile(r"\bhttps?://([^\s,]+?)([.,!?]?)(?=\s|$)")
_EMAIL = re.compile(r"\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
_CURRENCY = re.compile(
    r"(?P<cur>[$€£₹]|\bRs\.?|\bINR|\bUSD|\bEUR|\bGBP)\s?(?P<num>\d[\d,]*(?:\.\d+)?)"
    r"(?:\s?(?P<scale>k|K|m|M|mn|bn|B|million|billion|lakh|crore)\b)?"
)
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s?%")
_DATE_SLASH = re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b")
_DATE_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_TIME = re.compile(r"\b(\d{1,2}):(\d{2})(?:\s?([AaPp])\.?\s?[Mm]\.?)?(?![\w:])")
_TIME_AMPM = re.compile(r"\b(\d{1,2})\s?([AaPp])\.?[Mm]\.?(?!\w)")
_PHONE = re.compile(r"(?<![\w.])(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]?\d{4}(?![\w])")
_PHONE_IN = re.compile(r"(?<![\w.])(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?![\w])")
_ORDINAL = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.I)
_ALNUM_ID = re.compile(r"\b(?=[A-Za-z0-9-]*\d)(?=[A-Za-z0-9-]*[A-Za-z])[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\b")
_NUMBER = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?![\w])")


def normalize_for_speech(text: str, *, date_order: str = "DMY", lang: str = "en") -> str:
    """Pure function: rewrite structured tokens in ``text`` into speakable words."""

    def url(m: re.Match[str]) -> str:
        body = m.group(1).rstrip("/")
        body = re.sub(r"^www\.", "", body)
        return body.replace(".", " dot ").replace("/", " slash ") + m.group(2)

    def email(m: re.Match[str]) -> str:
        user = m.group(1).replace(".", " dot ").replace("_", " underscore ").replace("-", " dash ")
        return f"{user} at {m.group(2).replace('.', ' dot ')}"

    def currency(m: re.Match[str]) -> str:
        cur = _CURRENCIES[m.group("cur").strip()]
        return _amount(m.group("num"), m.group("scale"), cur, lang)

    def date_slash(m: re.Match[str]) -> str:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        d, mo = (a, b) if date_order.upper() == "DMY" else (b, a)
        if mo > 12 and d <= 12:  # obviously the other order
            d, mo = mo, d
        return _date(d, mo, y, lang) or m.group(0)

    def date_iso(m: re.Match[str]) -> str:
        return _date(int(m.group(3)), int(m.group(2)), int(m.group(1)), lang) or m.group(0)

    def time_hm(m: re.Match[str]) -> str:
        return _time(int(m.group(1)), int(m.group(2)), m.group(3)) or m.group(0)

    def time_ampm(m: re.Match[str]) -> str:
        return _time(int(m.group(1)), 0, m.group(2)) or m.group(0)

    def phone(m: re.Match[str]) -> str:
        groups = [g for g in re.split(r"\D+", m.group(0)) if g]
        return ", ".join(" ".join(_DIGIT_WORDS[int(c)] for c in g) for g in groups)

    def ordinal(m: re.Match[str]) -> str:
        return num2words(int(m.group(1)), to="ordinal", lang="en")

    def alnum(m: re.Match[str]) -> str:
        tok = m.group(0)
        if _UNIT_SUFFIX.match(tok) or len(tok) < 3:
            return tok
        return spell_id(tok)

    def number(m: re.Match[str]) -> str:
        raw = m.group(1)
        digits = raw.replace(",", "")
        prefix = text_so_far(m)
        if (
            "." not in raw
            and "," not in raw
            and (_ID_CONTEXT.search(prefix) or digits.startswith("0") and len(digits) > 1 or len(digits) >= 7)
        ):
            return spell_digits(digits)
        if "." in digits:
            whole, frac = digits.split(".", 1)
            return f"{_words(int(whole), lang)} point {' '.join(_DIGIT_WORDS[int(c)] for c in frac)}"
        return _words(int(digits), lang)

    current = {"s": ""}

    def text_so_far(m: re.Match[str]) -> str:
        return current["s"][max(0, m.start() - 30) : m.start()]

    s = text
    s = _URL.sub(url, s)
    s = _EMAIL.sub(email, s)
    s = _CURRENCY.sub(currency, s)
    s = _PERCENT.sub(lambda m: f"{m.group(1)} percent", s)
    s = _DATE_ISO.sub(date_iso, s)
    s = _DATE_SLASH.sub(date_slash, s)
    s = _TIME.sub(time_hm, s)
    s = _TIME_AMPM.sub(time_ampm, s)
    s = _PHONE_IN.sub(phone, s)
    s = _PHONE.sub(phone, s)
    s = _ORDINAL.sub(ordinal, s)
    s = re.sub(r"#\s?(?=\d)", "number ", s)
    s = _ALNUM_ID.sub(alnum, s)
    current["s"] = s
    s = _NUMBER.sub(number, s)
    s = re.sub(r"\s&\s", " and ", s)
    return s


# A clause boundary: punctuation followed by whitespace, or a newline.
_BOUNDARY = re.compile(r"(?:[.!?;:,](?=\s)|\n)")
_MAX_BUFFER = 160


def speech_normalizer(
    *, date_order: str = "DMY", lang: str = "en"
) -> Callable[[AsyncIterable[Any]], AsyncIterator[Any]]:
    """Streaming TTS text transform.

    LLM tokens split structured values arbitrarily ("$4" + "5.50", "+1 555" + " 123 4567"),
    so text is buffered up to the next clause boundary before normalizing. At ~100 tok/s
    this costs a few tens of ms, far less than the TTS's own sentence buffering.
    """

    def norm(t: str) -> str:
        return normalize_for_speech(t, date_order=date_order, lang=lang)

    async def transform(stream: AsyncIterable[Any]) -> AsyncIterator[Any]:
        buf = ""
        async for chunk in stream:
            if not isinstance(chunk, str):  # control sentinel: flush what we have first
                if buf:
                    yield norm(buf)
                    buf = ""
                yield chunk
                continue
            buf += chunk
            last = max(_BOUNDARY.finditer(buf), key=lambda m: m.end(), default=None)
            if last is not None:
                cut = last.end()
                yield norm(buf[:cut])
                buf = buf[cut:]
            elif len(buf) > _MAX_BUFFER and " " in buf:
                cut = buf.rfind(" ")
                yield norm(buf[:cut])
                buf = buf[cut:]
        if buf:
            yield norm(buf)

    return transform
