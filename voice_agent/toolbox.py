"""The assistant's everyday tools: web search, weather, shell, calculator, timers.

All of them are keyless. Each returns short plain text that the LLM turns into a spoken answer.
"""

from __future__ import annotations

import ast
import asyncio
import logging
import math
import operator
import os
import re
import shlex
from typing import Any

import httpx
from livekit.agents import RunContext, ToolError, function_tool

from .config import load_settings
from .state import SessionState
from .telemetry import redact
from .tools import end_call, get_current_datetime, require_confirmation, voice_tool

logger = logging.getLogger("voice-agent")

_HTTP_HEADERS = {"User-Agent": "voice-ai-agent/0.1 (local voice assistant)"}

# ------------------------------------------------------------------ web search


@voice_tool(filler="Let me look that up.", timeout=10)
async def web_search(ctx: RunContext[SessionState], query: str) -> str:
    """Search the web. Use for news, current events, recent facts, prices, sports results,
    anything after your training data, or anything you are not sure about.

    Args:
        query: A short search query, like you would type into a search engine.
    """

    def search() -> list[dict[str, str]]:
        from ddgs import DDGS

        # "auto" sometimes comes back empty; fall back to specific engines before giving up
        for backend in ("auto", "bing,yahoo", "google"):
            try:
                hits = DDGS(timeout=5).text(query, max_results=4, backend=backend)
            except Exception as e:  # DDGSException("No results found."), rate limits, timeouts
                logger.info("web search via %s failed: %s", backend, e)
                continue
            if hits:
                return hits
        return []

    hits = await asyncio.to_thread(search)
    if not hits:
        return "No results found. Tell the user you couldn't find anything for that."
    lines = [f"- {h.get('title', '')}: {h.get('body', '')[:300]}" for h in hits]
    return (
        "Search results (summarize the answer in one or two spoken sentences, "
        "and mention the source site only if it matters):\n" + "\n".join(lines)
    )


# ------------------------------------------------------------------ weather

_WMO = {
    0: "clear sky",
    1: "mostly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "heavy freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light showers",
    81: "showers",
    82: "violent showers",
    85: "snow showers",
    86: "heavy snow showers",
    95: "thunderstorms",
    96: "thunderstorms with hail",
    99: "thunderstorms with heavy hail",
}


@voice_tool(filler="Checking the weather.", timeout=10)
async def get_weather(ctx: RunContext[SessionState], location: str) -> str:
    """Get current weather and today's and tomorrow's forecast for a place.

    Args:
        location: City or place name, optionally with country, e.g. "Bengaluru" or "Paris, France".
    """
    async with httpx.AsyncClient(timeout=6, headers=_HTTP_HEADERS) as http:
        # OpenStreetMap geocoding ranks by importance, so "Bangalore" -> Bengaluru, India
        geo = await http.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": location, "format": "json", "limit": 1},
        )
        geo.raise_for_status()
        places = geo.json()
        if not places:
            raise ToolError(f"I couldn't find a place called {location}. Ask the user to say it differently.")
        place = places[0]
        name = ", ".join(place["display_name"].split(", ")[:1] + place["display_name"].split(", ")[-1:])
        fx = await http.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["lat"],
                "longitude": place["lon"],
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,"
                "wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
                "forecast_days": 2,
                "timezone": "auto",
                "temperature_unit": load_settings().temperature_unit,
            },
        )
        fx.raise_for_status()
        data = fx.json()

    unit = "degrees Fahrenheit" if load_settings().temperature_unit == "fahrenheit" else "degrees Celsius"
    cur, day = data["current"], data["daily"]

    def deg(v: float) -> str:
        return f"{round(v)} {unit}"

    def daily(i: int, label: str) -> str:
        rain = day["precipitation_probability_max"][i]
        rain_txt = f", {rain} percent chance of rain" if rain is not None else ""
        return (
            f"{label}: {_WMO.get(day['weather_code'][i], 'mixed')}, "
            f"high {deg(day['temperature_2m_max'][i])}, "
            f"low {deg(day['temperature_2m_min'][i])}{rain_txt}."
        )

    return (
        f"Weather in {name} right now: {_WMO.get(cur['weather_code'], 'mixed')}, "
        f"{deg(cur['temperature_2m'])}, "
        f"feels like {deg(cur['apparent_temperature'])}, humidity {cur['relative_humidity_2m']} percent, "
        f"wind {round(cur['wind_speed_10m'])} kilometres per hour. "
        f"{daily(0, 'Today')} {daily(1, 'Tomorrow')} "
        "Answer only what the user asked (e.g. just the current temperature), in one or two short sentences."
    )


# ------------------------------------------------------------------ calculator

_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS: dict[str, Any] = {
    name: getattr(math, name)
    for name in (
        "sqrt",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "log",
        "log10",
        "log2",
        "exp",
        "factorial",
    )
}
_FUNCS.update(abs=abs, round=round, min=min, max=max)
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


def safe_eval(expr: str) -> float | int:
    """Evaluate an arithmetic expression without exec/eval. Raises ValueError on anything else."""
    expr = expr.replace("^", "**").replace("×", "*").replace("÷", "/").replace(",", "")

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise ValueError("exponent too large")
            return _BIN[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
            args = [ev(a) for a in node.args]
            if node.func.id == "factorial" and args and args[0] > 1000:
                raise ValueError("factorial too large")
            return _FUNCS[node.func.id](*args)
        raise ValueError(f"unsupported expression: {ast.dump(node)[:60]}")

    return ev(ast.parse(expr.strip(), mode="eval"))


@function_tool
async def calculate(ctx: RunContext[SessionState], expression: str) -> str:
    """Calculate an arithmetic expression exactly. Use for ANY maths instead of doing it yourself:
    percentages, tips, unit conversions, interest, etc.

    Args:
        expression: A Python-style expression, e.g. "18 / 100 * 2450", "sqrt(2)", "(5 + 3) ** 2".
    """
    try:
        value = safe_eval(expression)
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, TypeError) as e:
        raise ToolError(
            f"Couldn't calculate that ({e}). Rephrase it as a plain arithmetic expression."
        ) from None
    if isinstance(value, float):
        value = round(value, 6)
        if value.is_integer():
            value = int(value)
    return f"{expression} = {value}"


# ------------------------------------------------------------------ timers

_TIMERS: dict[str, set[asyncio.Task[None]]] = {}


_UNIT_SECONDS = {
    "s": 1,
    "sec": 1,
    "second": 1,
    "m": 60,
    "min": 60,
    "minute": 60,
    "h": 3600,
    "hr": 3600,
    "hour": 3600,
}
_NUM_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
    "forty-five": 45, "sixty": 60, "ninety": 90, "half an": 0.5, "half a": 0.5,
}  # fmt: skip


def parse_duration(text: str) -> int:
    """'5 seconds' -> 5, '2 minutes' -> 120, '1 hour 30 minutes' -> 5400, 'half an hour' -> 1800."""
    t = re.sub(r"(\d+) and a half", r"\1.5", text.lower())
    total, found = 0.0, False
    pattern = (
        r"(\d+(?:\.\d+)?|half an|half a|forty-five|[a-z]+)\s*"
        r"(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h)\b"
    )
    for num, unit in re.findall(pattern, t):
        value = float(num) if num[0].isdigit() else _NUM_WORDS.get(num)
        if value is None:
            continue
        key = unit.rstrip("s") if unit not in ("s", "secs") else "s"
        total += value * _UNIT_SECONDS.get(key, _UNIT_SECONDS.get(key[:3], 1))
        found = True
    if not found and re.fullmatch(r"\s*\d+\s*", t):
        return int(t)  # bare number: seconds
    return int(total)


@function_tool
async def set_timer(ctx: RunContext[SessionState], duration: str, label: str = "") -> str:
    """Set a timer or short reminder. When it goes off, you will say so out loud.

    Args:
        duration: How long, in words exactly as the user said it, e.g. "5 seconds", "10 minutes",
            "1 hour 30 minutes".
        label: Optional short name, e.g. "tea" or "call mom".
    """
    seconds = parse_duration(duration)
    if not 1 <= seconds <= 24 * 3600:
        raise ToolError("Couldn't understand that duration, or it's over 24 hours. Ask the user again.")
    session, call_id = ctx.session, ctx.userdata.call_id
    what = f"Your {label} timer" if label else "Your timer"

    async def ring() -> None:
        await asyncio.sleep(seconds)
        if not ctx.userdata.ending:
            session.say(f"{what} is done!", allow_interruptions=True)

    task = asyncio.create_task(ring())
    timers = _TIMERS.setdefault(call_id, set())
    timers.add(task)
    task.add_done_callback(timers.discard)
    named = f", label {label}" if label else ""
    return f"Timer set for {duration} ({seconds} seconds){named}. Confirm in a few words."


def cancel_timers(call_id: str) -> None:
    for task in _TIMERS.pop(call_id, set()):
        task.cancel()


# ------------------------------------------------------------------ shell

# Commands that only read state run without asking. Everything else needs a spoken "yes".
_READ_ONLY = {
    "date", "cal", "uptime", "whoami", "hostname", "pwd", "ls", "cat", "head", "tail", "wc",
    "grep", "egrep", "find", "du", "df", "ps", "echo", "which", "sw_vers", "uname", "sysctl",
    "system_profiler", "ifconfig", "ping", "top", "pmset", "networksetup", "git", "file", "stat",
    "sort", "uniq", "cut", "tr", "awk", "env", "printenv", "vm_stat", "mdfind", "say", "open",
    "python3", "node", "brew", "ollama", "osascript", "defaults", "diskutil",
}  # fmt: skip
# For these, only specific read-only forms skip confirmation.
_READ_ONLY_FORMS = {
    "git": re.compile(r"^git (status|log|diff|branch|show|remote -v)\b"),
    "pmset": re.compile(r"^pmset -g\b"),
    "networksetup": re.compile(r"^networksetup -(get|list)"),
    "ping": re.compile(r"^ping .*-c ?\d"),
    "top": re.compile(r"^top .*-l ?\d"),
    "python3": re.compile(r"^python3 (--version|-V)$"),
    "node": re.compile(r"^node (--version|-v)$"),
    "brew": re.compile(r"^brew (list|info|search|--version|outdated)\b"),
    "ollama": re.compile(r"^ollama (list|ps|--version)\b"),
    "defaults": re.compile(r"^defaults read\b"),
    "diskutil": re.compile(r"^diskutil (list|info)\b"),
    "osascript": re.compile(
        r"^osascript -e ['\"]?(get volume settings|output volume of \(get volume settings\))"
    ),
    "find": re.compile(r"^find (?!.*-(delete|exec|ok)\b)"),
    "awk": re.compile(r"^awk (?!.*(system|>))"),
}
# Never run these, even with confirmation.
_BLOCKED = re.compile(
    r"\b(sudo|su|doas|mkfs|fdisk|dd|shutdown|reboot|halt|launchctl|csrutil|nvram|passwd|security|"
    r"chown|kextload|spctl)\b"
    r"|rm\s+(-\w*[rf]\w*\s+)+(/|~|\$HOME|\*)(\s|$)"
    r"|diskutil\s+(erase|partition|zero|secure)"
    r"|(curl|wget)[^|]*\|\s*(ba|z)?sh"
    r"|:\(\)\s*\{"
    r"|>\s*/dev/(disk|sd|rdisk)"
    r"|chmod\s+-R"
    r"|\.ssh|\.aws|\.gnupg|Keychains|\.env\b|id_rsa|id_ed25519",
)
_SEGMENT_SPLIT = re.compile(r"\|\||&&|\||;")
_MAX_OUTPUT = 1500


def classify_command(command: str) -> str:
    """'blocked', 'read_only' or 'needs_confirmation'."""
    cmd = command.strip()
    if not cmd or _BLOCKED.search(cmd):
        return "blocked"
    # redirection, command substitution and background jobs can write or hide anything
    if re.search(r">|`|\$\(|&(?!&)|\n", cmd):
        return "needs_confirmation"
    for segment in _SEGMENT_SPLIT.split(cmd):
        segment = segment.strip()
        try:
            tokens = shlex.split(segment)
        except ValueError:
            return "needs_confirmation"
        if not tokens:
            continue
        prog = os.path.basename(tokens[0])
        if prog not in _READ_ONLY:
            return "needs_confirmation"
        form = _READ_ONLY_FORMS.get(prog)
        if form and not form.search(" ".join([prog, *tokens[1:]])):
            return "needs_confirmation"
    return "read_only"


@function_tool
async def run_shell_command(ctx: RunContext[SessionState], command: str) -> str:
    """Run a shell command on the user's Mac (macOS, zsh) and get its output. Use it for system
    info (battery, disk space, Wi-Fi, running apps), files and folders, opening apps or websites
    ("open -a Spotify", "open https://..."), volume, and anything else the user asks you to do on
    their computer. Read-only commands run immediately; others ask the user to confirm first.

    Args:
        command: The exact shell command to run.
    """
    settings = load_settings()
    if not settings.shell_tool:
        raise ToolError("Running commands is turned off in this setup.")
    kind = classify_command(command)
    if kind == "blocked":
        raise ToolError("That command is blocked for safety. Tell the user you can't run it.")
    if kind == "needs_confirmation":
        pending = require_confirmation(ctx, "run_shell_command", {"command": command})
        if pending:
            return pending
    if ctx.speech_handle.interrupted:
        return "Not run: the user interrupted."

    logger.info("shell", extra={"json": {"command": redact(command), "kind": kind}})
    proc = await asyncio.create_subprocess_shell(
        command,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=os.path.expanduser("~"),
        start_new_session=True,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=settings.shell_timeout)
    except TimeoutError:
        proc.kill()
        raise ToolError(
            f"The command took longer than {settings.shell_timeout:.0f} seconds and was stopped."
        ) from None
    text = out.decode(errors="replace").strip()
    if len(text) > _MAX_OUTPUT:
        text = text[:_MAX_OUTPUT] + f"\n...[{len(text) - _MAX_OUTPUT} more characters cut]"
    status = "ok" if proc.returncode == 0 else f"exit code {proc.returncode}"
    return (
        f"Command finished ({status}). Output:\n{text or '(no output)'}\n"
        "Tell the user the answer in plain speech; never read raw output, paths or codes verbatim."
    )


ALL_TOOLS = [
    get_current_datetime,
    web_search,
    get_weather,
    calculate,
    set_timer,
    run_shell_command,
    end_call,
]
