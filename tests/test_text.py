from __future__ import annotations

import pytest

from voice_agent.normalize import normalize_for_speech as n
from voice_agent.normalize import speech_normalizer
from voice_agent.telemetry import redact
from voice_agent.tools import digits_from_speech


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Total is $45.50.", "Total is forty-five dollars and fifty cents."),
        ("That's $1.", "That's one dollar."),
        ("₹1,250 due", "one thousand, two hundred and fifty rupees due"),
        ("Raised $2.5M", "Raised two million, five hundred thousand dollars"),
        ("on 15/09/2026", "on the fifteenth of September, twenty twenty-six"),
        ("on 2026-10-09", "on the ninth of October, twenty twenty-six"),
        ("at 14:30", "at two thirty PM"),
        ("at 9:05 am", "at nine oh five AM"),
        ("by 3pm", "by three PM"),
        ("Order 45678 shipped", "Order four five six, seven eight shipped"),
        ("Order #45678", "Order number four five six, seven eight"),
        ("ID AB12345", "ID A B one, two three four five"),
        ("OTP is 0423", "OTP is zero four two three"),
        ("call (555) 123-4567", "call five five five, one two three, four five six seven"),
        ("call 98765 43210", "call nine eight seven six five, four three two one zero"),
        ("you're 1st", "you're first"),
        ("12.5% off", "twelve point five percent off"),
        ("3 items", "three items"),
        ("a 5kg box", "a 5kg box"),
        ("mail john.doe@acme.com", "mail john dot doe at acme dot com"),
        ("see https://acme.com/help.", "see acme dot com slash help."),
        ("returns & refunds", "returns and refunds"),
    ],
)
def test_normalize(text: str, expected: str) -> None:
    assert n(text) == expected


def test_mdy_dates() -> None:
    assert n("09/15/2026", date_order="MDY") == "the fifteenth of September, twenty twenty-six"
    # impossible month in DMY order -> read the other way
    assert n("09/15/2026", date_order="DMY") == "the fifteenth of September, twenty twenty-six"


async def _collect(chunks: list[str]) -> str:
    async def gen():
        for c in chunks:
            yield c

    return "".join([c async for c in speech_normalizer()(gen())])


async def test_streaming_handles_values_split_across_tokens() -> None:
    chunks = ["Your total", " is $4", "5.", "50 and", " it arrives 15/0", "9/2026", ". Order ", "456", "78."]
    out = await _collect(chunks)
    assert out == (
        "Your total is forty-five dollars and fifty cents and it arrives the fifteenth of "
        "September, twenty twenty-six. Order four five six, seven eight."
    )


async def test_streaming_emits_early_at_clause_boundaries() -> None:
    seen: list[str] = []

    async def gen():
        for c in ["Sure, ", "let me", " check."]:
            yield c

    async for out in speech_normalizer()(gen()):
        seen.append(out)
    assert seen[0] == "Sure,"  # first clause goes to TTS without waiting for the sentence


def test_redact() -> None:
    s = redact(
        "card 4111 1111 1111 1111, mail a.b@x.io, phone +1 555 123 4567, acct 99887766, "
        "otp four five six seven eight nine"
    )
    assert "4111" not in s and "a.b@x.io" not in s and "555" not in s and "99887766" not in s
    assert "four five six" not in s
    assert redact("order 45 shipped") == "order 45 shipped"


@pytest.mark.parametrize(
    "spoken, digits",
    [
        ("four five six, seven eight", "45678"),
        ("45 678", "45678"),
        ("double five six oh", "5560"),
        ("my order is 12345", "12345"),
    ],
)
def test_digits_from_speech(spoken: str, digits: str) -> None:
    assert digits_from_speech(spoken) == digits
