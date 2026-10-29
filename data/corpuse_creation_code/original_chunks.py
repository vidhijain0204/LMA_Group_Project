#!/usr/bin/env python3
"""Hand-written teaching chunks for the Python tutor corpus.

All original (non-scraped) content lives in this file, in two lists:

GAP_CHUNKS
    Fill subconcepts the scraped sources do not cover well. Running this script verifies them and
    writes `original_chunks.jsonl`, which `scrape_corpus.py` merges into its output.

FINALIZE_CHUNKS
    Misconception explanations, self-contained file examples and late gap fills. They are imported and
    added by `finalize_corpus.py` *after* its cleaning step, so the cleaning rules (written for scraped
    textbook text) never modify them.

Every ```python block is executed and, when followed by an "Output:" block, the printed text must match
exactly; a chunk that fails is rejected. Chunks marked verify=False show deliberately broken code.

Usage:
    python3 build_original_chunks.py [--topics topics.json] [--out original_chunks.jsonl]
"""
import argparse
import contextlib
import hashlib
import io
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent

GAP_CHUNKS = []
FINALIZE_CHUNKS = []


def C(topic, subs, ctype, difficulty, tags, title, text):
    """Add a gap-filling chunk (merged by the scraper)."""
    GAP_CHUNKS.append(dict(topic=topic, subs=subs, type=ctype, difficulty=difficulty, tags=tags,
                           title=title, text=text.strip()))


def F(topic, subs, ctype, difficulty, tags, title, text, verify=True):
    """Add a chunk that finalize_corpus.py appends after cleaning."""
    FINALIZE_CHUNKS.append(dict(topic=topic, subs=subs, type=ctype, difficulty=difficulty, tags=tags,
                                title=title, text=text.strip(), verify=verify))


# =============================================================================================
# GAP_CHUNKS
# =============================================================================================
# ======================================================================= T03.1
C("T03", ["T03.1"], "explanation", 1, [], "print() with sep and end", r'''
`print()` shows values on the screen. Give it several values separated by commas and it prints them with one space between them, then starts a new line.

```python
print("Score:", 42, "points")
```

Output:
```text
Score: 42 points
```

Two keyword arguments change this formatting:
- `sep` is the text placed between the values (default: one space).
- `end` is the text printed after the last value (default: a newline, `"\n"`).

```python
print("2024", "06", "15", sep="-")
print("Loading", end="...")
print("done")
```

Output:
```text
2024-06-15
Loading...done
```

`sep` and `end` must be written by name (`sep=`, `end=`). Without the name, an extra string is simply printed as one more value: `print("a", "b", "-")` prints `a b -`.
''')

C("T03", ["T03.1"], "example", 2, [], "Building a line of output with sep and end", r'''
Building one line of output from several print calls. With `end=""`, the next `print` continues on the same line:

```python
print("Loading", end="")
print(".", end="")
print(".", end="")
print(" done")
```

Output:
```text
Loading.. done
```

The last `print` uses the default `end`, so it finishes the line.

Joining values without a gap, and with a custom divider:

```python
print("a", "b", "c", sep="")
print("Name", "Age", sep=" | ")
```

Output:
```text
abc
Name | Age
```
''')

# ======================================================================= T03.5
C("T03", ["T03.5"], "explanation", 2, [], "Reading several values from one line", r'''
`input()` always returns one string, even if the user types several numbers on the line. To read several values from one line, split the string into pieces with `.split()` and convert each piece to the type you need.

By default `.split()` cuts the string at any run of spaces. Give it an argument such as `.split(",")` to cut at a different separator.

```python
line = "10 25"        # what input() returns when the user types: 10 25
first, second = line.split()
total = int(first) + int(second)
print(total)
```

Output:
```text
35
```

Unpacking (`first, second = ...`) works only when the number of pieces equals the number of names on the left; otherwise Python raises a `ValueError`. If the number of values can vary, keep the pieces in a list instead: `parts = line.split()`, and use `len(parts)` to see how many there are.
''')

C("T03", ["T03.5", "T03.4"], "example", 2, [], "Examples: splitting a line of input", r'''
Reading three comma-separated numbers and adding them up:

```python
line = "4,8,15"        # same as line = input() when the user types: 4,8,15
a, b, c = line.split(",")
print(int(a) + int(b) + int(c))
```

Output:
```text
27
```

Reading a name and a number from one line:

```python
line = "Asha 17"
name, age_text = line.split()
age = int(age_text)
print(name, "will be", age + 1, "next year")
```

Output:
```text
Asha will be 18 next year
```
''')

C("T03", ["T03.5"], "common_mistake", 2, ["unconverted_split_values"], "Using split() pieces as numbers", r'''
**Common mistake: using the pieces of `.split()` as numbers**

`.split()` always returns a list of strings. Adding two of them joins the text instead of adding numbers:

```python
a, b = "3 4".split()
print(a + b)
```

Output:
```text
34
```

Convert before doing arithmetic:

```python
a, b = "3 4".split()
print(int(a) + int(b))
```

Output:
```text
7
```

Checklist: after `split()`, which of the pieces should be numbers? Wrap each of those in `int()` (or `float()`) before calculating with them.
''')

# ======================================================================= T05.6
C("T05", ["T05.6"], "explanation", 2, [], "Truthy and falsy values", r'''
In a condition, Python does not need a real `True` or `False`: every value counts as either truthy or falsy.

Falsy values: `False`, `None`, the number zero (`0`, `0.0`), and empty collections (`""`, `[]`, `{}`, `()`, `set()`). Everything else is truthy, including non-zero numbers, non-empty strings, and non-empty lists.

`bool(value)` shows how Python treats a value:

```python
print(bool(0), bool(""), bool([]), bool(None))
print(bool(7), bool("hi"), bool([0]))
```

Output:
```text
False False False False
True True True
```

Note that `[0]` is truthy: the list is not empty, even though its only item is zero.

This allows short conditions: `if items:` means "if the list has at least one item", the same as `if len(items) > 0:`.
''')

C("T05", ["T05.6"], "example", 2, [], "Using truthiness in conditions and defaults", r'''
Using truthiness in a condition and to supply a default:

```python
names = []
if names:
    print("have names")
else:
    print("no names yet")

nickname = ""
display = nickname or "Anonymous"
print(display)
```

Output:
```text
no names yet
Anonymous
```

`a or b` gives `a` if it is truthy, otherwise `b`. So an empty nickname falls back to the default.
''')

C("T05", ["T05.6"], "common_mistake", 3, ["truthiness_confusion"], "Treating a valid zero as missing", r'''
**Common mistake: treating a valid zero (or empty value) as "missing"**

`0` is falsy, so a check like `if score:` is false when the score is a perfectly good 0.

```python
score = 0
if score:
    print("has a score")
else:
    print("no score")
```

Output:
```text
no score
```

If the real question is "was a value provided at all?", compare with `None` instead:

```python
score = 0
if score is not None:
    print("has a score")
else:
    print("no score")
```

Output:
```text
has a score
```

Also remember that any non-empty string is truthy, even `"False"` and `"0"`:

```python
print(bool("False"), bool("0"))
```

Output:
```text
True True
```
''')

# ======================================================================= T06.5
C("T06", ["T06.5"], "explanation", 3, [], "Nested loops", r'''
A nested loop is a loop inside another loop. For every single iteration of the outer loop, the inner loop runs from start to finish.

```python
for row in range(3):
    for col in range(2):
        print(row, col)
```

Output:
```text
0 0
0 1
1 0
1 1
2 0
2 1
```

The outer loop runs 3 times and the inner loop runs 2 times for each of those, so the inner body runs 3 x 2 = 6 times.

Nested loops are used for grids and tables, and for comparing every item with every other item. The inner loop's range can depend on the outer variable, which produces triangle-shaped patterns.
''')

C("T06", ["T06.5"], "example", 3, [], "A triangle of stars", r'''
A triangle of stars, where the inner loop depends on the outer variable:

```python
for line in range(1, 5):
    row = ""
    for _ in range(line):
        row += "*"
    print(row)
```

Output:
```text
*
**
***
****
```

Tracing tip: write down the value of the outer variable, then list every value the inner variable takes for it, before moving to the next outer value.
''')

C("T06", ["T06.5", "T06.4"], "explanation", 3, [], "break inside a nested loop", r'''
`break` inside a nested loop only leaves the **inner** loop. The outer loop carries on with its next iteration.

```python
for row in range(3):
    for col in range(3):
        if col == 2:
            break
        print(row, col)
```

Output:
```text
0 0
0 1
1 0
1 1
2 0
2 1
```

Each time `col` reaches 2 the inner loop stops, but the outer loop still visits all three rows. To stop both loops at once, set a flag variable that the outer loop also checks, or move the loops into a function and use `return`.
''')

C("T06", ["T06.5"], "common_mistake", 3, ["loop_variable_misuse"], "Same variable name in both loops", r'''
**Common mistake: using the same variable name in both loops**

The inner loop assigns to its loop variable too. If it has the same name as the outer one, the outer value is overwritten:

```python
for i in range(3):
    for i in range(2):
        pass
    print(i)
```

Output:
```text
1
1
1
```

After the inner loop finishes, `i` is 1 (its last value), so that is what gets printed, not the outer loop's 0, 1, 2. Give the two loops different names, such as `row` and `col`, or `i` and `j`.
''')

# ======================================================================= T11
C("T11", ["T11.1"], "explanation", 2, [], "Basic list comprehension syntax", r'''
A list comprehension builds a new list from an existing sequence in a single expression. The pattern is:

`[expression for item in iterable]`

Read it as: "for each `item` in the iterable, compute `expression` and collect the results".

```python
numbers = [1, 2, 3, 4]
squares = [n * n for n in numbers]
print(squares)
```

Output:
```text
[1, 4, 9, 16]
```

It does exactly what this loop does:

```python
squares = []
for n in numbers:
    squares.append(n * n)
print(squares)
```

Output:
```text
[1, 4, 9, 16]
```

The square brackets tell Python to build a list. A comprehension always creates a new list and leaves the original unchanged.
''')

C("T11", ["T11.1"], "example", 2, [], "List comprehension examples", r'''
More examples with different iterables:

```python
words = ["tea", "coffee", "milk"]
print([len(w) for w in words])
print([w.upper() for w in words])
print([i * 10 for i in range(4)])
```

Output:
```text
[3, 6, 4]
['TEA', 'COFFEE', 'MILK']
[0, 10, 20, 30]
```

The loop variable (`w`, `i`) can have any name, and it exists only inside the comprehension.
''')

C("T11", ["T11.2"], "explanation", 2, [], "Filtering with an if clause", r'''
Add an `if` at the end of a comprehension to keep only the items that pass a test:

`[expression for item in iterable if condition]`

```python
scores = [72, 45, 88, 91, 60]
passed = [s for s in scores if s >= 60]
print(passed)
```

Output:
```text
[72, 88, 91, 60]
```

The condition decides **which items are used**; the expression at the front decides **what is stored**. They can be combined:

```python
boosted = [s + 5 for s in scores if s < 60]
print(boosted)
```

Output:
```text
[50]
```
''')

C("T11", ["T11.3"], "explanation", 3, [], "Conditional expressions in a comprehension", r'''
A conditional expression has the form `value_if_true if condition else value_if_false`. Inside a comprehension it goes at the **front**, where it chooses what to store for every item. No item is dropped.

```python
temps = [18, 25, 31, 12]
labels = ["warm" if t >= 20 else "cool" for t in temps]
print(labels)
```

Output:
```text
['cool', 'warm', 'warm', 'cool']
```

Compare with a filter, where `if` goes at the **end** and removes items:

```python
print([t for t in temps if t >= 20])
```

Output:
```text
[25, 31]
```

Rule of thumb: an `if` alone at the end filters; `if ... else` at the front transforms.
''')

C("T11", ["T11.3", "T11.2"], "common_mistake", 3, ["filter_transform_placement"], "if-else at the end of a comprehension", r'''
**Common mistake: putting `if ... else` at the end of a comprehension**

Writing `[t for t in temps if t >= 20 else 0]` is a `SyntaxError`: a trailing `if` is a filter and cannot have an `else`. The `if ... else` choice belongs in front of the `for`:

```python
temps = [18, 25, 31, 12]
print([t if t >= 20 else 0 for t in temps])
```

Output:
```text
[0, 25, 31, 0]
```

Ask yourself: "Do I want to skip some items (filter, `if` at the end) or change them (conditional expression at the front)?"
''')

C("T11", ["T11.4"], "explanation", 3, [], "Nested loops in a comprehension", r'''
A comprehension can contain more than one `for`. The clauses are read in the same order as nested loops, left to right: the first `for` is the outer loop.

```python
pairs = [(r, c) for r in range(2) for c in range(3)]
print(pairs)
```

Output:
```text
[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
```

This is the same as writing:

```python
pairs = []
for r in range(2):
    for c in range(3):
        pairs.append((r, c))
```

A common use is flattening a list of lists:

```python
matrix = [[1, 2], [3, 4], [5, 6]]
flat = [x for row in matrix for x in row]
print(flat)
```

Output:
```text
[1, 2, 3, 4, 5, 6]
```

If a comprehension needs more than two `for` clauses or becomes hard to read, use ordinary loops instead.
''')

C("T11", ["T11.5", "T11.2"], "explanation", 3, [], "Converting a loop into a comprehension", r'''
A loop that builds a list can be turned into a comprehension, and back, by moving four parts:

1. The empty list before the loop disappears (the comprehension creates it).
2. The loop header (`for w in words`) becomes the `for` clause.
3. An `if` that guards the `append` becomes the filter at the end.
4. The expression inside `append(...)` moves to the front.

```python
words = ["tea", "coffee", "milk", "ale", "juice"]

result = []
for w in words:
    if len(w) > 3:
        result.append(w.upper())
print(result)

result = [w.upper() for w in words if len(w) > 3]
print(result)
```

Output:
```text
['COFFEE', 'MILK', 'JUICE']
['COFFEE', 'MILK', 'JUICE']
```

Going the other way, write the empty list, copy the `for` clause as a loop header, turn the trailing `if` into an `if` inside the loop, and `append` the front expression.
''')

C("T11", ["T11.5"], "common_mistake", 3, ["side_effects_in_comprehension"], "Using a comprehension just to run code", r'''
**Common mistake: using a comprehension just to run code**

A comprehension is for building a list. Using it only for its side effects (such as printing) builds a useless list of `None` values:

```python
result = [print(x) for x in [1, 2]]
print(result)
```

Output:
```text
1
2
[None, None]
```

If you only want to do something with each item, use a normal `for` loop. Keep comprehensions for when you need the resulting list.
''')

C("T11", ["T11.1"], "common_mistake", 2, ["wrong_brackets"], "Wrong brackets around a comprehension", r'''
**Common mistake: wrong brackets**

Only square brackets build a list. The same expression in round brackets creates a generator, and in curly braces a set:

```python
print(type([n * n for n in range(3)]).__name__)
print(type((n * n for n in range(3))).__name__)
print(type({n * n for n in range(3)}).__name__)
```

Output:
```text
list
generator
set
```

If you print a generator you will not see the values. Use square brackets, or wrap it in `list(...)`, when you want a list.
''')


# =============================================================================================
# FINALIZE_CHUNKS
# =============================================================================================
# --- Self-contained file examples (T13)

F('T13', ['T13.2', 'T13.3', 'T13.4', 'T13.6'], 'example', 2, [], 'A complete file example you can run', r'''
This example is self-contained: it first creates the file it then reads, so it runs anywhere.

```python
with open("notes.txt", "w") as f:
    f.write("first line\n")
    f.write("second line\n")

with open("notes.txt") as f:
    for line in f:
        print(line.rstrip())
```

Output:
```text
first line
second line
```

- `open(name, "w")` creates the file (or empties it if it already exists) for writing.
- `with` closes the file automatically when the indented block ends, even if an error happens.
- Looping over a file object gives one line at a time. Each line still ends with `"\n"`, so `.rstrip()` removes it before printing.
''')

F('T13', ['T13.4', 'T13.3'], 'common_mistake', 2, ['write_mode_overwrites', 'read_returns_string'], 'Write mode erases the file; read() returns one string', r'''
**Common mistake: expecting `"w"` to add to a file**

Opening a file with `"w"` erases what was there. Use `"a"` (append) to add to the end:

```python
with open("log.txt", "w") as f:
    f.write("run 1\n")
with open("log.txt", "w") as f:
    f.write("run 2\n")
with open("log.txt") as f:
    print(repr(f.read()))
```

Output:
```text
'run 2\n'
```

`"run 1"` is gone. Appending keeps the old content:

```python
with open("log.txt", "a") as f:
    f.write("run 3\n")
with open("log.txt") as f:
    print(f.read().splitlines())
```

Output:
```text
['run 2', 'run 3']
```

Also note that `f.read()` returns the whole file as **one string**; `.splitlines()` turns it into a list of lines.
''')

# --- Common mistakes for loops (T06). These show deliberately broken code, including an infinite loop,
# so they are not executed

F('T06', ['T06.1'], 'common_mistake', 2, ['off_by_one_range'], 'Off-by-one errors with range()', r'''
**Common mistake: off-by-one errors with `range()`**

`range(a, b)` stops *before* `b`. Students often expect the end value to be included.

```python
for i in range(1, 5):
    print(i)              # prints 1 2 3 4, NOT 1 2 3 4 5
```

How to avoid it:
- Count the values: `range(a, b)` produces `b - a` numbers.
- To include `b` when counting up, write `range(a, b + 1)`.
- When counting down, the stop value is also excluded, so `range(5, 0, -1)` gives 5 4 3 2 1 and never reaches 0.
- When looping over positions in a list, use `range(len(items))`, not `range(len(items) + 1)`; the latter causes an `IndexError`.
''', verify=False)

F('T06', ['T06.3'], 'common_mistake', 2, ['infinite_loop_no_update'], 'A while loop that never ends', r'''
**Common mistake: a `while` loop that never ends**

If nothing inside the loop changes what the condition tests, the condition never becomes `False`.

```python
n = 5
while n > 0:
    print(n)              # prints 5 forever: n never changes
```

The fix is to update the variable the condition depends on:

```python
n = 5
while n > 0:
    print(n)
    n -= 1
```

Checklist: (1) Which variable does the condition test? (2) Does the body change it? (3) Does it change in the direction that makes the condition false? An infinite loop can also come from updating the wrong variable or moving in the wrong direction (for example `n += 1` when the condition is `n > 0`). In a terminal, Ctrl+C interrupts a runaway loop.
''', verify=False)

F('T06', ['T06.2'], 'common_mistake', 3, ['modify_list_while_iterating'], 'Changing a list while looping over it', r'''
**Common mistake: changing a list while looping over it**

Removing items from a list during a `for` loop shifts the remaining items left, so the loop skips the element that moved into the current position.

```python
nums = [2, 4, 6, 7]
for n in nums:
    if n % 2 == 0:
        nums.remove(n)
print(nums)               # [4, 7]  (4 was skipped)
```

Safer approaches:
- Build a new list containing only the items you want to keep.
- Loop over a copy: `for n in nums[:]:`.
- Loop over the indexes from the end towards the start if you must delete in place.

The same problem occurs with `insert()` and `append()` while iterating over the same list.
''', verify=False)

F('T06', ['T06.6'], 'common_mistake', 2, ['accumulator_reset_in_loop'], 'Resetting the accumulator inside the loop', r'''
**Common mistake: resetting the accumulator inside the loop**

```python
for price in [5, 8, 2]:
    total = 0             # reset on every iteration!
    total += price
print(total)              # 2, not 15
```

Because `total = 0` runs on every pass, earlier additions are thrown away. Move the initialisation above the loop:

```python
total = 0
for price in [5, 8, 2]:
    total += price
print(total)              # 15
```

A related slip is initialising a "best-so-far" with a value that can be beaten wrongly, such as starting `best = 0` when all the data may be negative. Starting from the first item avoids this.
''', verify=False)

F('T06', ['T06.4'], 'common_mistake', 2, ['break_continue_confusion'], 'Mixing up break and continue', r'''
**Common mistake: mixing up `break` and `continue`**

- `break` leaves the whole loop.
- `continue` only skips the rest of the current iteration.

```python
for n in [1, 2, 3, 4]:
    if n == 2:
        break
    print(n)              # 1

for n in [1, 2, 3, 4]:
    if n == 2:
        continue
    print(n)              # 1 3 4
```

Ask yourself: "Do I want to stop looping altogether, or just ignore this one item?" Also remember that code placed after `continue` in the same iteration never runs, and that in a nested loop `break` exits only the inner loop.
''', verify=False)

F('T06', ['T06.2'], 'common_mistake', 2, ['loop_variable_misuse'], 'Misunderstanding what the loop variable holds', r'''
**Common mistake: misunderstanding what the loop variable holds**

When looping over a list directly, the loop variable is the *item*, not its index.

```python
colors = ["red", "green", "blue"]
for c in colors:
    print(colors[c])      # TypeError: list indices must be integers
```

Here `c` is already `"red"`, `"green"`, or `"blue"`. Either use the item directly (`print(c)`) or loop over indexes:

```python
for i in range(len(colors)):
    print(colors[i])
```

Another pitfall: the loop variable keeps its last value after the loop ends, and reusing the same name for an inner and an outer loop makes the outer variable get overwritten.
''', verify=False)

# --- One common-mistake chunk for each misconception the scraped sources do not cover

F('T01', ['T01.2'], 'common_mistake', 1, ['repl_script_confusion'], 'Expecting a script to show values like the prompt does', r'''
**Common mistake: expecting a script to show values the way the interactive prompt does**

At the interactive prompt (`>>>`), typing an expression shows its value:

```text
>>> 2 + 3
5
```

In a script file, a bare expression is calculated and then thrown away, so nothing appears. Use `print()` to display it:

```python
2 + 3          # in a script: computed, but nothing is shown
print(2 + 3)
```

Output:
```text
5
```

Rule: in a script, only `print()` (or another output function) displays anything.
''')

F('T02', ['T02.1', 'T02.5'], 'common_mistake', 1, ['assignment_as_equation'], "Reading = as 'is equal to'", r'''
**Common mistake: reading `=` as "is equal to"**

In Python, `=` means "store the value on the right in the name on the left". It is an instruction, not a claim that both sides are equal. That is why this works:

```python
count = 3
count = count + 1
print(count)
```

Output:
```text
4
```

Read the second line as "take the current value of `count`, add 1, and store the result back in `count`". In maths `x = x + 1` would be impossible; in Python it is the normal way to update a variable.

The direction matters: `5 = count` is a `SyntaxError` ("cannot assign to literal"), because the left side must be a name. To *compare* two values, use `==`.
''')

F('T03', ['T03.1'], 'common_mistake', 1, ['print_syntax_error'], 'print without parentheses or with mismatched quotes', r'''
**Common mistake: `print` without parentheses, or with mismatched quotes**

In Python 3, `print` is a function, so it needs parentheses. Old Python 2 tutorials show `print "hello"`, which is now an error:

```text
>>> print "hello"
SyntaxError: Missing parentheses in call to 'print'. Did you mean print(...)?
```

The quotes around a string must also match: a string that starts with `"` must end with `"`. To put one kind of quote inside a string, surround it with the other kind, or escape it with a backslash:

```python
print("hello")
print('It\'s fine')
print("She said 'hi'")
```

Output:
```text
hello
It's fine
She said 'hi'
```
''')

F('T03', ['T03.1'], 'common_mistake', 1, ['print_vs_variable_value'], 'Thinking print() changes or stores a value', r'''
**Common mistake: thinking `print()` changes or stores a value**

`print()` only displays a value. It does not change any variable, and it does not hand the value back for later use.

```python
price = 10
print(price * 2)
print(price)
```

Output:
```text
20
10
```

`price` is still 10: the doubled value was shown, not stored. To keep a result, assign it, for example `doubled = price * 2`.

`print()` itself returns `None`, so saving its result gives you nothing useful:

```python
result = print("hi")
print(result)
```

Output:
```text
hi
None
```
''')

F('T04', ['T04.3', 'T05.5'], 'common_mistake', 2, ['chained_or_misuse'], 'Writing x == 1 or 2', r'''
**Common mistake: writing `x == 1 or 2`**

It reads like English ("x is 1 or 2"), but Python evaluates `x == 1` first and then `or 2`. Because `2` is truthy, the whole condition is always true:

```python
x = 7
if x == 1 or 2:
    print("matched")
```

Output:
```text
matched
```

Compare `x` with each value explicitly, or test membership with `in`:

```python
x = 7
print(x == 1 or x == 2)
print(x in (1, 2))
```

Output:
```text
False
False
```
''')

F('T05', ['T05.1', 'T04.2'], 'common_mistake', 1, ['assign_in_condition'], 'Using = instead of == in a condition', r'''
**Common mistake: `=` instead of `==` in a condition**

`=` stores a value; `==` compares two values. A condition needs a comparison, so this is an error:

```text
>>> if x = 5:
SyntaxError: invalid syntax. Maybe you meant '==' or ':=' instead of '='?
```

The correct version:

```python
x = 5
if x == 5:
    print("x is five")
```

Output:
```text
x is five
```

(The exact wording of the error message varies a little between Python versions, but it always points at the `=`.)
''')

F('T05', ['T05.3'], 'common_mistake', 2, ['multiple_if_vs_elif'], 'Separate ifs where only one branch should run', r'''
**Common mistake: separate `if` statements where only one branch should run**

```python
score = 95
if score >= 90:
    print("A")
if score >= 80:
    print("B")
if score >= 70:
    print("C")
```

Output:
```text
A
B
C
```

Each `if` is checked on its own, so all three print. With `elif`, Python stops at the first condition that is true:

```python
score = 95
if score >= 90:
    print("A")
elif score >= 80:
    print("B")
elif score >= 70:
    print("C")
```

Output:
```text
A
```
''')

F('T07', ['T07.4'], 'common_mistake', 2, ['split_join_confusion'], 'Mixing up split() and join()', r'''
**Common mistake: mixing up `split()` and `join()`**

`split()` breaks a string into a **list**. `join()` does the opposite, and it is called on the **separator** string, with the list as its argument:

```python
words = "red green blue".split()
print(words)
print("-".join(words))
```

Output:
```text
['red', 'green', 'blue']
red-green-blue
```

Writing `words.join("-")` fails, because lists have no `join` method. `join` also only accepts strings, so convert numbers first:

```python
nums = [1, 2, 3]
parts = []
for n in nums:
    parts.append(str(n))
print(", ".join(parts))
```

Output:
```text
1, 2, 3
```
''')

F('T08', ['T08.1', 'T08.5'], 'common_mistake', 1, ['list_index_out_of_range'], 'Using an index that does not exist', r'''
**Common mistake: using an index that does not exist**

A list with n items has indexes 0 to n - 1. Asking for index n raises `IndexError: list index out of range`.

```python
colors = ["red", "green", "blue"]
print(len(colors))
print(colors[2])
print(colors[-1])
```

Output:
```text
3
blue
blue
```

`colors[3]` would fail: there is no fourth item. For the last item use `colors[-1]` (or `colors[len(colors) - 1]`), and prefer `for c in colors:` over managing indexes by hand.
''')

F('T09', ['T09.4', 'T09.2'], 'common_mistake', 3, ['mutate_dict_while_iterating'], 'Deleting keys while looping over a dictionary', r'''
**Common mistake: adding or deleting keys while looping over a dictionary**

```text
>>> stock = {"apple": 0, "pear": 4, "plum": 0}
>>> for fruit in stock:
...     if stock[fruit] == 0:
...         del stock[fruit]
...
RuntimeError: dictionary changed size during iteration
```

Loop over a copy of the keys instead (or build a new dictionary):

```python
stock = {"apple": 0, "pear": 4, "plum": 0}
for fruit in list(stock):
    if stock[fruit] == 0:
        del stock[fruit]
print(stock)
```

Output:
```text
{'pear': 4}
```
''')

F('T09', ['T09.1', 'T09.4'], 'common_mistake', 2, ['keys_values_confusion'], 'Mixing up keys and values', r'''
**Common mistake: mixing up keys and values**

A dictionary maps **keys** to **values**. `d[key]` looks up by key, and `in` checks keys only; neither searches the values.

```python
ages = {"Ana": 31, "Ben": 25}
print(ages["Ana"])
print(31 in ages)
print(31 in ages.values())
```

Output:
```text
31
False
True
```

To find which key has a given value, loop over the items:

```python
for name, age in ages.items():
    if age == 25:
        print(name)
```

Output:
```text
Ben
```
''')

F('T09', ['T09.1', 'T09.2'], 'common_mistake', 2, ['duplicate_key_overwrite'], 'Expecting a dictionary to keep repeated keys', r'''
**Common mistake: expecting a dictionary to keep repeated keys**

Each key can appear only once. Assigning to an existing key replaces its value without any warning:

```python
phone = {"Ana": "111", "Ana": "222"}
print(phone)
phone["Ana"] = "333"
print(phone)
```

Output:
```text
{'Ana': '222'}
{'Ana': '333'}
```

If one key needs several values, store a list as the value:

```python
phone = {"Ana": ["111", "222"]}
phone["Ana"].append("333")
print(phone)
```

Output:
```text
{'Ana': ['111', '222', '333']}
```
''')

F('T10', ['T10.1'], 'common_mistake', 1, ['defined_not_called'], 'Defining a function but never calling it', r'''
**Common mistake: defining a function but never calling it**

`def` only creates the function. Its body runs when you call it, with parentheses.

```python
def greet():
    print("Hello!")

greet          # only names the function: nothing happens
greet()        # calls it
```

Output:
```text
Hello!
```

A program made only of `def` blocks prints nothing at all. Also check the indentation: a call indented under the `def` becomes part of the function body instead of calling it.
''')

F('T10', ['T10.2'], 'common_mistake', 2, ['argument_count_mismatch'], 'Passing the wrong number of arguments', r'''
**Common mistake: passing the wrong number of arguments**

A call must supply one argument for every parameter that has no default value:

```text
>>> def area(width, height):
...     return width * height
...
>>> area(5)
TypeError: area() missing 1 required positional argument: 'height'
```

With both arguments it works:

```python
def area(width, height):
    return width * height

print(area(5, 3))
```

Output:
```text
15
```

Arguments are matched to parameters by position. For `area` the order does not change the answer, but for a function such as `power(base, exponent)` it does. Use keyword arguments, for example `power(base=2, exponent=3)`, when the order is easy to mix up.
''')

F('T10', ['T10.3'], 'common_mistake', 2, ['early_return_in_loop'], 'return inside the loop, too early', r'''
**Common mistake: `return` inside the loop, too early**

`return` ends the function immediately. In the wrong place it stops the loop after the first item:

```python
def all_even(nums):
    for n in nums:
        if n % 2 != 0:
            return False
        else:
            return True

print(all_even([2, 3, 4]))
```

Output:
```text
True
```

The function only looked at `2`. The "every item passed" answer belongs **after** the loop, once all items have been checked:

```python
def all_even(nums):
    for n in nums:
        if n % 2 != 0:
            return False
    return True

print(all_even([2, 3, 4]))
```

Output:
```text
False
```
''')

F('T11', ['T11.4'], 'common_mistake', 3, ['nested_loop_order'], 'for clauses of a nested comprehension in the wrong order', r'''
**Common mistake: putting the `for` clauses of a nested comprehension in the wrong order**

The `for` clauses run in the same order as nested loops written out: the first one is the outer loop.

```python
matrix = [[1, 2], [3, 4]]
print([x for row in matrix for x in row])
```

Output:
```text
[1, 2, 3, 4]
```

Swapping them, `[x for x in row for row in matrix]`, fails with `NameError: name 'row' is not defined`, because `row` is used before the loop that creates it. (If a variable called `row` already exists, it silently gives a wrong result instead.) Write the clauses in the order you would nest the loops:

```python
flat = []
for row in matrix:
    for x in row:
        flat.append(x)
print(flat)
```

Output:
```text
[1, 2, 3, 4]
```
''')

F('T11', ['T11.5'], 'common_mistake', 3, ['overly_complex_comprehension'], 'Cramming too much into one comprehension', r'''
**Common mistake: cramming too much into one comprehension**

This works, but it is hard to read and hard to debug:

```python
data = [[3, -1], [4, 10], [-2, 8]]
result = [x * 2 if x > 5 else x for row in data if sum(row) > 0 for x in row if x != 0]
print(result)
```

Output:
```text
[3, -1, 4, 20, -2, 16]
```

A plain loop says the same thing step by step and is easier to change:

```python
result = []
for row in data:
    if sum(row) > 0:
        for x in row:
            if x != 0:
                result.append(x * 2 if x > 5 else x)
print(result)
```

Output:
```text
[3, -1, 4, 20, -2, 16]
```

Guideline: if a comprehension needs more than one `for` and one `if`, or does not fit comfortably on one line, use a loop.
''')

F('T12', ['T12.2'], 'common_mistake', 2, ['swallowed_exception'], 'Catching an error and doing nothing', r'''
**Common mistake: catching an error and doing nothing**

```python
def to_number(text):
    try:
        return int(text)
    except ValueError:
        pass

print(to_number("12"))
print(to_number("twelve"))
```

Output:
```text
12
None
```

The error disappeared, and the program carries on with `None`, which usually causes a confusing failure later. If you catch an exception, handle it: show a message, use a sensible default, or ask again.

```python
def to_number(text):
    try:
        return int(text)
    except ValueError:
        print(f"'{text}' is not a whole number, using 0")
        return 0

print(to_number("twelve"))
```

Output:
```text
'twelve' is not a whole number, using 0
0
```
''')

F('T12', ['T12.2'], 'common_mistake', 3, ['oversized_try_block'], 'Putting too much code inside try', r'''
**Common mistake: putting too much code inside `try`**

```python
values = ["4", "x", "2"]
total = 0
try:
    for v in values:
        total += int(v)
    print("average:", total / len(values))
except ValueError:
    print("bad value")
```

Output:
```text
bad value
```

One bad item stopped the whole calculation, and the message does not say which value failed. Keep the `try` around only the line that can fail:

```python
total, count = 0, 0
for v in values:
    try:
        n = int(v)
    except ValueError:
        print("skipping", repr(v))
        continue
    total += n
    count += 1
print("average:", total / count)
```

Output:
```text
skipping 'x'
average: 3.0
```
''')

F('T12', ['T12.3', 'T12.1'], 'common_mistake', 2, ['wrong_exception_type'], 'Catching the wrong exception type', r'''
**Common mistake: catching the wrong exception type**

An `except` clause only catches the type it names. `int("abc")` raises `ValueError`, not `TypeError`, so this handler never runs and the program still crashes:

```text
>>> try:
...     n = int("abc")
... except TypeError:
...     print("not a number")
...
ValueError: invalid literal for int() with base 10: 'abc'
```

Name the type that is actually raised:

```python
try:
    n = int("abc")
except ValueError:
    print("not a number")
```

Output:
```text
not a number
```

The last line of a traceback tells you the type. Common ones: `int("abc")` gives `ValueError`; `"a" + 1` gives `TypeError`; `[1, 2][5]` gives `IndexError`; `{"a": 1}["b"]` gives `KeyError`; `open("missing.txt")` gives `FileNotFoundError`.
''')

F('T13', ['T13.3'], 'common_mistake', 2, ['read_pointer_exhausted'], 'Reading the same file object twice', r'''
**Common mistake: reading the same file object twice**

A file object remembers how far it has read. After everything has been read, a second read returns an empty string:

```python
with open("data.txt", "w") as f:
    f.write("a\nb\n")

with open("data.txt") as f:
    first = f.read()
    second = f.read()
print(repr(first))
print(repr(second))
```

Output:
```text
'a\nb\n'
''
```

To read again, open the file again (or call `f.seek(0)` to go back to the start), or keep the text you already read in a variable and reuse it.
''')

# --- Subconcepts that had fewer than 3 chunks after label correction

F('T02', ['T02.3'], 'explanation', 1, [], 'Converting between types', r'''
Python never changes a value's type by itself, but you can ask for a converted copy with `int()`, `float()`, `str()`, and `bool()`:

```python
print(int("42") + 1)
print(float("3.5") * 2)
print(str(7) + " days")
print(int(9.99))
print(bool(0), bool(5), bool(""), bool("no"))
```

Output:
```text
43
7.0
7 days
9
False True False True
```

Points to remember:
- `int()` on a float **cuts off** the decimal part (`int(9.99)` is `9`); it does not round. Use `round()` to round.
- `str()` lets you join a number to text with `+`.
- `bool()` gives `False` for zero and for empty text, and `True` for everything else, even the string `"no"`.
''')

F('T02', ['T02.3'], 'example', 2, ['string_number_confusion'], 'Conversions that fail, and how to avoid them', r'''
`int()` only accepts text that looks like a whole number. Anything else raises a `ValueError`:

```text
>>> int("3.5")
ValueError: invalid literal for int() with base 10: '3.5'
>>> int("ten")
ValueError: invalid literal for int() with base 10: 'ten'
```

For text containing a decimal point, convert to `float` first. Spaces around the number are allowed:

```python
print(int(float("3.5")))
print(int("  12  "))
print(float("2") + 0.5)
```

Output:
```text
3
12
2.5
```

A string such as `"5"` and the number `5` are different values: `"5" + "5"` is `"55"`, while `5 + 5` is `10`. Convert first, then calculate.
''')

F('T02', ['T02.4'], 'explanation', 2, [], 'type() and dynamic typing', r'''
In Python a **value** has a type; a variable is just a name that can refer to any value. The same name can refer to an `int` now and a `str` later. `type()` tells you the type of the value a name currently refers to:

```python
x = 5
print(type(x))
x = "five"
print(type(x))
print(type(3.0), type(True))
```

Output:
```text
<class 'int'>
<class 'str'>
<class 'float'> <class 'bool'>
```

This is called **dynamic typing**: types are checked while the program runs, not declared in advance. It is flexible, but it also means a mistake such as adding text to a number is only discovered when that line runs.
''')

F('T02', ['T02.4', 'T02.3'], 'example', 2, [], "Checking a value's type", r'''
`type()` is useful when a value does not behave as you expect, for example after reading input, which always gives text:

```python
value = "12"
print(type(value))
number = int(value)
print(type(number), number + 1)
```

Output:
```text
<class 'str'>
<class 'int'> 13
```

To test a type in a condition, compare with the type name or use `isinstance`:

```python
value = "12"
print(type(value) == str)
print(isinstance(value, int))
```

Output:
```text
True
False
```
''')

F('T04', ['T04.5'], 'explanation', 2, ['precedence_error'], 'Operator precedence', r'''
When an expression has several operators, Python applies them in a fixed order, not simply left to right:

```python
print(2 + 3 * 4)
print((2 + 3) * 4)
print(2 ** 3 ** 2)
print(-2 ** 2)
print(10 - 4 - 3)
```

Output:
```text
14
20
512
-4
3
```

- `*`, `/`, `//` and `%` are applied before `+` and `-`.
- `**` is applied first of all, and it groups from the right: `2 ** 3 ** 2` is `2 ** 9`.
- `-2 ** 2` is `-(2 ** 2)`, because `**` binds more tightly than the minus sign.
- Operators of the same level, such as `-` and `-`, go left to right: `(10 - 4) - 3`.
- Comparisons come after arithmetic, and `not`, `and`, `or` come last, in that order.

When in doubt, add parentheses: they make the order explicit and the code easier to read.
''')

F('T10', ['T10.5'], 'explanation', 3, ['local_scope_access'], 'Local and global variables', r'''
A variable assigned inside a function is **local**: it exists only while the function runs, and it is separate from any variable with the same name outside.

```python
count = 10

def show():
    count = 99
    print("inside:", count)

show()
print("outside:", count)
```

Output:
```text
inside: 99
outside: 10
```

A function can **read** a variable defined outside it, but assigning to that name creates a new local variable instead. To change the outer (global) variable you would need the `global` keyword:

```python
total = 0

def add(amount):
    global total
    total = total + amount

add(5)
print(total)
```

Output:
```text
5
```

Prefer returning a value (`total = add(total, 5)`) over `global`: it keeps each function self-contained and easier to test.
''')

F('T11', ['T11.3'], 'example', 3, [], 'More conditional expressions in comprehensions', r'''
A conditional expression in front of the `for` decides what to store for **every** item:

```python
scores = [45, 80, 62]
print(["pass" if s >= 50 else "fail" for s in scores])

nums = [3, -1, 4]
print([n if n > 0 else 0 for n in nums])

words = ["apple", "kiwi", "banana"]
print([w.upper() if len(w) > 4 else w for w in words])
```

Output:
```text
['fail', 'pass', 'pass']
[3, 0, 4]
['APPLE', 'kiwi', 'BANANA']
```

Each result list has the same length as the input: nothing is filtered out, every item is transformed one way or the other.
''')

F('T12', ['T12.3'], 'explanation', 2, ['wrong_exception_type'], 'Handling different exceptions differently', r'''
A `try` statement can have several `except` clauses, one for each kind of error you expect. Python runs the first one that matches:

```python
values = ["10", "abc", "0"]
for v in values:
    try:
        print(100 / int(v))
    except ValueError:
        print(v, "is not a number")
    except ZeroDivisionError:
        print("cannot divide by zero")
```

Output:
```text
10.0
abc is not a number
cannot divide by zero
```

If several errors need the same handling, list them in parentheses: `except (ValueError, ZeroDivisionError):`. Name only the errors you expect; an unexpected error should still stop the program so you notice it.
''')

F('T12', ['T12.4'], 'explanation', 3, ['else_finally_confusion'], 'else and finally', r'''
After `except`, a `try` statement can have two more clauses:
- `else` runs only when **no** exception happened in the `try` block.
- `finally` runs **always**, whether there was an exception or not. It is used for clean-up.

```python
for text in ["5", "x"]:
    try:
        n = int(text)
    except ValueError:
        print("bad input:", text)
    else:
        print("converted:", n)
    finally:
        print("done with", text)
```

Output:
```text
converted: 5
done with 5
bad input: x
done with x
```

Putting the "success" code in `else` keeps the `try` block small, so the `except` only catches errors from the line that can actually fail.
''')

F('T12', ['T12.5'], 'explanation', 3, [], 'Raising your own exceptions', r'''
`raise` signals an error deliberately, for example when a function receives a value it cannot work with. Choose a built-in exception type that fits and give a clear message:

```python
def set_age(age):
    if age < 0:
        raise ValueError("age cannot be negative")
    return age

print(set_age(30))
try:
    set_age(-4)
except ValueError as e:
    print("Error:", e)
```

Output:
```text
30
Error: age cannot be negative
```

Raising an error is better than silently returning a wrong value: the caller finds out immediately, and the traceback points to the cause.
''')


# =============================================================================================
# Verification and output
# =============================================================================================
FENCE = re.compile(r"```python\n(.*?)\n```", re.S)
OUTPUT = re.compile(r"\n\nOutput:\n```text\n(.*?)\n```", re.S)


def normalise(s):
    return "\n".join(line.rstrip() for line in s.strip().splitlines())


def check_examples(text):
    """Run every python block of `text` in one namespace; return a list of problems."""
    problems, namespace = [], {}
    for m in FENCE.finditer(text):
        printed = io.StringIO()
        try:
            with contextlib.redirect_stdout(printed):
                exec(m.group(1), namespace)
        except Exception as e:  # noqa: BLE001 - any failure means the example is wrong
            problems.append(f"raised {type(e).__name__}: {e}")
            continue
        shown = OUTPUT.match(text, m.end())
        if shown and normalise(shown.group(1)) != normalise(printed.getvalue()):
            problems.append(f"output mismatch: shown {normalise(shown.group(1))!r}, "
                            f"actual {normalise(printed.getvalue())!r}")
    return problems


def check_ids(chunk, sub_ids, tag_ids):
    problems = [f"unknown subconcept {s}" for s in chunk["subs"] if s not in sub_ids]
    problems += [f"unknown misconception tag {t}" for t in chunk["tags"] if t not in tag_ids]
    if chunk["type"] == "common_mistake" and not chunk["tags"]:
        problems.append("a common_mistake chunk needs misconception tags")
    return problems


def to_record(chunk, topic_name, timestamp):
    """Convert an authored chunk to the corpus record format (chunk_id is assigned later)."""
    text = f"**{topic_name} > {chunk['title']}**\n\n{chunk['text']}"
    code_words = sum(len(m.group(1).split()) for m in FENCE.finditer(text))
    return {
        "chunk_id": None, "topic_id": chunk["topic"], "subconcept_ids": chunk["subs"], "type": chunk["type"],
        "difficulty": chunk["difficulty"], "misconception_tags": chunk["tags"], "text": text,
        "source": {"id": "original", "name": "Original teaching material written for this project", "url": None,
                   "page_title": chunk["title"], "section_path": [topic_name, chunk["title"]],
                   "license": "Project-original (team to choose a licence)",
                   "attribution": "Written for the Personalized Python Tutor project", "redistributable": True},
        "scraped_at": timestamp, "word_count": len(text.split()), "has_code": code_words > 0,
        "mapping": {"method": "authored", "confidence": 1.0, "candidate_topics": None, "scores": {}},
        "flags": [], "content_hash": hashlib.sha256(re.sub(r"\s+", " ", text.lower()).encode()).hexdigest()[:16],
    }


def main():
    default_topics = HERE / "topics.json" if (HERE / "topics.json").exists() else HERE.parent / "topics.json"
    ap = argparse.ArgumentParser(description="Verify the gap chunks and write original_chunks.jsonl")
    ap.add_argument("--topics", default=str(default_topics))
    ap.add_argument("--out", default=str(HERE / "original_chunks.jsonl"))
    args = ap.parse_args()

    topics = {t["topic_id"]: t for t in json.loads(Path(args.topics).read_text())}
    sub_ids = {s["id"] for t in topics.values() for s in t["subconcepts"]}
    tag_ids = {m["id"] for t in topics.values() for m in t["common_misconceptions"]}
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    records, failed = [], 0
    for chunk in GAP_CHUNKS:
        problems = check_ids(chunk, sub_ids, tag_ids) + check_examples(chunk["text"])
        if problems:
            failed += 1
            print(f"FAIL {chunk['topic']} {chunk['title']}")
            for p in problems:
                print("   -", p)
            continue
        records.append(to_record(chunk, topics[chunk["topic"]]["name"], timestamp))
    if failed:
        sys.exit(f"{failed} chunk(s) failed verification; nothing written.")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"OK: {len(records)} gap chunks verified -> {args.out}")


if __name__ == "__main__":
    main()