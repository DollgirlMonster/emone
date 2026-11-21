#!/usr/bin/env python3
"""Fixed inputs for phases 2 and 3 of the divergence overview (Emone-8d0).

Phase 2 (token flips): 24 three-turn conversations. Twelve open on a ~16K-char
slice of the repository's documentation at HEAD (so follow-up turns are short
prefills deep in the cache: the key split, small QMM and resident rows all
engage there); twelve are short chats. Follow-ups are a few words each.

Phase 3 (outcomes): 30 multi-step arithmetic word problems generated from a
fixed seed with their answers computed here, and 15 small Python functions
checked by running tests. Everything is deterministic: same seed, same text.
"""

import random
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DOCS = [
    "docs/m1-prefill-spike.md",
    "docs/adding-a-model.md",
    "docs/frontier-prefix-cache.md",
    "docs/repository-layout.md",
    "docs/release-process.md",
    "docs/task-table-standard.md",
]

DOC_QUESTIONS = [
    ("Summarise the main point of this document in three sentences.",
     ["Which part of that is most likely to break?", "Say it again as one sentence."]),
    ("List the three most important numbers in this document and what each measures.",
     ["Which of those would you check first?", "Why?"]),
    ("What problem is this document trying to solve?",
     ["Name one risk it mentions.", "How would you test for that risk?"]),
    ("Write a short checklist someone could follow from this document.",
     ["Shorten it to three items.", "Which item takes longest?"]),
]

CHATS = [
    ("Explain what a hash table is to a new programmer.", ["Give a tiny example.", "What is a collision?"]),
    ("Why does the sky look blue?", ["Why are sunsets red then?", "Answer in one line."]),
    ("Write a haiku about a lighthouse.", ["Now one about fog.", "Which do you prefer?"]),
    ("What are three good habits for writing commit messages?", ["Show an example message.", "Make it shorter."]),
    ("Compare TCP and UDP in a few sentences.", ["When would you pick UDP?", "Give a real product that uses it."]),
    ("How do vaccines train the immune system?", ["What is a booster for?", "Summarise in one sentence."]),
    ("Describe a simple recipe for pancakes.", ["How do I make them fluffier?", "Can I skip the eggs?"]),
    ("What is the difference between weather and climate?", ["Give an example of each.", "Keep it to ten words."]),
    ("Suggest a name for a cat that likes boxes.", ["Three more, please.", "Pick your favourite."]),
    ("Explain recursion using a real-world analogy.", ["Where does the analogy break?", "Write a recursive factorial in Python."]),
    ("What should I pack for a two-day hike?", ["What if it rains?", "Which item is most often forgotten?"]),
    ("Explain what an API is in plain words.", ["Give an everyday example.", "And a bad API?"]),
]


def doc_slices(count: int, chars: int = 16_000) -> list[str]:
    text = ""
    for path in DOCS:
        text += subprocess.run(["git", "show", f"HEAD:{path}"], cwd=ROOT, capture_output=True,
                               text=True, check=True).stdout + "\n\n"
    text = "".join(c for c in text if c in "\t\n\r" or 32 <= ord(c) < 127)
    step = max(1, (len(text) - chars) // max(1, count - 1))
    return [text[i * step: i * step + chars] for i in range(count)]


def conversations() -> list[dict]:
    convs = []
    for i, doc in enumerate(doc_slices(12)):
        q, follow = DOC_QUESTIONS[i % len(DOC_QUESTIONS)]
        convs.append({"id": f"doc{i:02d}", "turns": [f"{doc}\n\n---\n\n{q}", *follow]})
    for i, (q, follow) in enumerate(CHATS):
        convs.append({"id": f"chat{i:02d}", "turns": [q, *follow]})
    return convs


def math_tasks(count: int = 30) -> list[dict]:
    # The generator is sequential: the first 30 are the same for any count.
    rng = random.Random(8_002)
    out = []
    names = ["Ada", "Ben", "Cleo", "Dev", "Esme", "Femi"]
    for i in range(count):
        kind = i % 6
        n = names[i % len(names)]
        if kind == 0:
            boxes, per, sold, price = rng.randint(4, 12), rng.randint(6, 18), rng.randint(5, 30), rng.randint(2, 9)
            q = (f"{n} has {boxes} boxes of {per} apples. {n} sells {sold} apples at ${price} each and gives "
                 f"half of the remaining apples (rounded down) to a neighbour. How many apples does {n} keep?")
            a = (boxes * per - sold) - (boxes * per - sold) // 2
        elif kind == 1:
            speed, hours, back = rng.randint(40, 90), rng.randint(2, 6), rng.randint(30, 70)
            q = (f"A train travels at {speed} km/h for {hours} hours, then returns at {back} km/h. "
                 f"How many minutes does the return trip take? Round to the nearest whole minute.")
            a = round(speed * hours / back * 60)
        elif kind == 2:
            w, h, tile = rng.randint(6, 20), rng.randint(5, 15), rng.choice([2, 3, 4])
            spare = rng.randint(1, 9)
            q = (f"A floor is {w} m by {h} m. Square tiles are {tile} m on a side and a tile covers at most "
                 f"its area (cut pieces are wasted). How many tiles are needed to cover it, plus {spare} spares?")
            a = -(-w // tile) * -(-h // tile) + spare
        elif kind == 3:
            start, rate, years = rng.randint(1000, 9000), rng.choice([2, 3, 4, 5, 10]), rng.randint(2, 4)
            q = (f"${start} earns {rate}% interest per year, compounded yearly. What is the balance after "
                 f"{years} years, rounded to the nearest dollar?")
            a = round(start * (1 + rate / 100) ** years)
        elif kind == 4:
            a1, d, k = rng.randint(1, 20), rng.randint(2, 9), rng.randint(8, 25)
            q = (f"A sequence starts at {a1} and each term is {d} more than the last. What is the sum of the "
                 f"first {k} terms?")
            a = k * (2 * a1 + (k - 1) * d) // 2
        else:
            x, y = rng.randint(3, 30), rng.randint(3, 30)
            s, diff = x + y, abs(x - y)
            q = (f"Two numbers add up to {s} and differ by {diff}. What is their product?")
            a = x * y
        out.append({"id": f"math{i:02d}", "kind": "math", "prompt":
                    q + "\nWork it out step by step, then write the final answer on the last line as "
                    "'ANSWER: <number>'.", "answer": a})
    return out


CODE = [
    ("is_palindrome(s)", "returns True if the string s reads the same backwards, ignoring case and "
     "non-alphanumeric characters",
     ["is_palindrome('A man, a plan, a canal: Panama') is True", "is_palindrome('abc') is False",
      "is_palindrome('') is True"]),
    ("rle(s)", "run-length encodes s as letter followed by count, e.g. 'aaab' -> 'a3b1'",
     ["rle('aaab') == 'a3b1'", "rle('') == ''", "rle('abc') == 'a1b1c1'"]),
    ("to_roman(n)", "converts an integer 1..3999 to a Roman numeral",
     ["to_roman(1994) == 'MCMXCIV'", "to_roman(4) == 'IV'", "to_roman(3999) == 'MMMCMXCIX'"]),
    ("flatten(xs)", "flattens arbitrarily nested lists into one list",
     ["flatten([1,[2,[3,[4]]],5]) == [1,2,3,4,5]", "flatten([]) == []"]),
    ("balanced(s)", "returns True if the brackets ()[]{} in s are balanced",
     ["balanced('([]{})') is True", "balanced('([)]') is False", "balanced('') is True"]),
    ("merge_intervals(xs)", "merges overlapping [start, end] intervals and returns them sorted",
     ["merge_intervals([[1,3],[2,6],[8,10]]) == [[1,6],[8,10]]", "merge_intervals([]) == []",
      "merge_intervals([[1,4],[4,5]]) == [[1,5]]"]),
    ("second_largest(xs)", "returns the second largest distinct value, or None if there is none",
     ["second_largest([3,1,4,4]) == 3", "second_largest([7]) is None", "second_largest([2,2]) is None"]),
    ("caesar(s, k)", "shifts letters by k (wrapping, preserving case), leaving other characters",
     ["caesar('abc XYZ!', 3) == 'def ABC!'", "caesar('a', -1) == 'z'"]),
    ("top_words(text, k)", "returns the k most frequent lowercase words, ties broken alphabetically",
     ["top_words('b a b c a b', 2) == ['b', 'a']", "top_words('x y z', 2) == ['x', 'y']"]),
    ("transpose(m)", "transposes a rectangular list of lists",
     ["transpose([[1,2,3],[4,5,6]]) == [[1,4],[2,5],[3,6]]", "transpose([]) == []"]),
    ("binary_search(xs, x)", "returns the index of x in sorted list xs, or -1",
     ["binary_search([1,3,5,7], 5) == 2", "binary_search([1,3,5,7], 4) == -1", "binary_search([], 1) == -1"]),
    ("anagram_groups(words)", "groups anagrams; returns a list of sorted groups, sorted by first word",
     ["anagram_groups(['eat','tea','tan','ate','nat','bat']) == [['ate','eat','tea'],['bat'],['nat','tan']]"]),
    ("digits_sum(n)", "repeatedly sums the decimal digits of n until one digit remains",
     ["digits_sum(9875) == 2", "digits_sum(0) == 0", "digits_sum(5) == 5"]),
    ("fizzbuzz(n)", "returns the list of FizzBuzz strings for 1..n",
     ["fizzbuzz(5) == ['1','2','Fizz','4','Buzz']", "fizzbuzz(15)[-1] == 'FizzBuzz'"]),
    ("count_islands(grid)", "counts 4-connected groups of 1s in a grid of 0/1 lists",
     ["count_islands([[1,1,0],[0,1,0],[1,0,1]]) == 3", "count_islands([]) == 0"]),
]


def code_tasks() -> list[dict]:
    out = []
    for i, (sig, what, tests) in enumerate(CODE):
        out.append({"id": f"code{i:02d}", "kind": "code", "prompt":
                    f"Write a Python function `{sig}` that {what}. Use only the standard library. "
                    "Reply with a single ```python code block containing the function and nothing else.",
                    "tests": tests})
    return out


def tasks() -> list[dict]:
    return math_tasks() + code_tasks()


if __name__ == "__main__":
    c, t = conversations(), tasks()
    print(len(c), "conversations,", sum(len(x["turns"][0]) for x in c), "first-turn chars;", len(t), "tasks")
    print(t[0]["prompt"], "->", t[0]["answer"])
