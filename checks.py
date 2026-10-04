"""
Small shared helpers that look at the GOAL TEXT and at a finished run.

Used by:
  - nodes/evaluator.py  (safety checks that can overrule the AI's "done")
  - nodes/executor.py   (blocks write_file when the goal never asked for a file)
  - app.py / main.py    (decide whether a run counts as a genuine success)
"""
import re

# ---------------------------------------------------------------------------
# Does the goal ask for a saved file?
#
# Matches WHOLE WORDS only, so "pro-FILE" and "EXCEL-lent" do not count.
#   (?<![a-z0-9])  = the clue must not be glued to a letter/digit before it
#   (?![a-z0-9])   = ...or after it
# "save/saved/saving" are ignored when followed by money/time/costs
# (e.g. "save money on coffee" is not a request to save a file).
# File formats only count when written like a filename (".json", ".pdf"),
# because words like "PDF" or "JSON" can also just be the topic of the goal.
# ---------------------------------------------------------------------------
FILE_REQUEST_PATTERN = re.compile(
    r"(?<![a-z0-9])(?:csv|xlsx|xls|excel|spreadsheet|files?)(?![a-z0-9])"
    r"|\.(?:json|txt|pdf|md|docx)(?![a-z0-9])"
    r"|(?<![a-z0-9])(?:save|saved|saving)(?![a-z0-9])(?!\s+(?:money|time|costs?|up|on)\b)"
)

# ---------------------------------------------------------------------------
# Does the goal contain a COUNT of items to find?
#
# A number only counts as "how many items" if it is NOT:
#   - followed by a unit  ("5 minutes", "2 cups", "10 dollars")
#   - a limit             ("under 5", "within 10")
#   - glued to letters    ("4k", "mp3", "output2.csv")
#   - part of a hyphen    ("2-in-1", "3-day"), a price ("$5") or a percent
# ---------------------------------------------------------------------------
UNIT_WORDS = {
    "minute", "minutes", "min", "mins", "hour", "hours", "hr", "hrs",
    "day", "days", "week", "weeks", "month", "months", "year", "years",
    "second", "seconds", "sec", "secs",
    "dollar", "dollars", "usd", "euro", "euros", "rupee", "rupees", "inr", "percent",
    "kg", "kgs", "g", "gram", "grams", "lb", "lbs", "pound", "pounds", "oz", "ounce", "ounces",
    "ml", "liter", "liters", "litre", "litres", "cup", "cups", "tbsp", "tsp",
    "tablespoon", "tablespoons", "teaspoon", "teaspoons", "calories", "cal",
    "cm", "mm", "inch", "inches", "ft", "feet", "km", "mile", "miles",
    "gb", "tb", "mb", "mah", "hz", "watt", "watts", "star", "stars",
    "people", "persons", "guests", "servings", "serving",
}
LIMIT_WORDS_BEFORE = {
    "under", "below", "over", "above", "within", "than", "around", "about",
    "upto", "max", "maximum", "minimum", "budget",
}


def goal_requires_file_output(goal: str) -> bool:
    """True if the goal's wording asks for a saved file."""
    return bool(FILE_REQUEST_PATTERN.search(goal.lower()))


def extract_required_item_count(goal: str) -> int | None:
    """
    Return the number of items the goal asks for (e.g. 3 in "find 3 recipes"),
    or None if the goal has no real item count. Bounded to 2-20.
    """
    text = goal.lower()
    for m in re.finditer(r"\d+", text):
        n = int(m.group())
        if not (2 <= n <= 20):
            continue
        before = text[:m.start()]
        after = text[m.end():]

        # glued to letters ("4k", "mp3", "output2") -> not a count
        if before[-1:].isalpha() or after[:1].isalpha():
            continue
        # price, percent, or hyphenated ("$5", "50%", "2-in-1") -> not a count
        if before.endswith(("$", "\u20ac", "\u00a3")) or after.startswith(("-", "%")):
            continue
        # followed by a unit ("5 minutes") -> not a count
        next_word = re.match(r"\s*([a-z]+)", after)
        if next_word and next_word.group(1) in UNIT_WORDS:
            continue
        # a limit ("under 5") -> not a count
        prev_word = re.search(r"([a-z]+)\s*$", before)
        if prev_word and prev_word.group(1) in LIMIT_WORDS_BEFORE:
            continue

        return n
    return None


def is_genuine_success(state) -> bool:
    """
    True only if the run really finished its goal. False for runs that were
    aborted (API down), hit the 15-step limit, or were stopped because the
    user declined a file save. Used so that only real successes are written
    into memory and counted as successes.
    """
    if not state.done or state.aborted:
        return False
    answer = (state.final_answer or "").strip().lower()
    if not answer:
        return False
    if answer.startswith("stopped") or answer.startswith("run stopped"):
        return False
    if "declined" in answer:
        return False
    return True