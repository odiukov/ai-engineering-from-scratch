---
name: check-code
version: 1.0.0
description: >
  Runs the practice tests for a lesson exercise, then — only if they all pass —
  the metric comparison against the reference solution, and finishes with a
  human code review of how the learner wrote it.
  Trigger phrases: "check my code", "run the tests", "проверь код",
  "запусти тесты", "/check-code"
tags: [practice, testing, review, ai-engineering]
---

# Check Code

The learner is working through exercises in `learning-exercises/lesson-NN/`.
They write `exercise.py`; the tests and the reference are already there. This
skill is the one command that tells them where they stand.

Teach and report in the language from `LEARNING.md`'s `Language:` field
(`ru` unless it says otherwise). Keep code, identifiers, test names, and
error strings verbatim.

## Step 0 — Pick the lesson

Argument wins: `/check-code lesson-02` targets that directory. With no
argument, use the highest-numbered `learning-exercises/lesson-*` directory
that exists.

`exercise.py` is deliberately not in git — each learner has their own. If it
is missing, create it by copying `exercise.template.py` and say so, then
continue. If neither exists, stop — there is nothing to check.

## Step 1 — Run the tests

```bash
cd learning-exercises/<lesson> && python3 -m pytest -q --no-header --tb=short
```

Report the count as `N/M`. Then branch on the result.

### If anything failed

Do **not** run the comparison — it is meaningless against broken code, and
seeing performance numbers before correctness distracts from the real
problem.

For each distinct failure (group repeats of the same root cause):

- Name the test and quote the shortest decisive line of its output — the
  `assert 0 == 32` line, not the whole traceback.
- Say what the test was checking, in words. Test names are written to be
  readable; use that.
- Give **one hint that moves them forward without handing over the answer**:
  which concept from the lesson applies, or which specific line to look at.
  Never paste working code for a function they have not solved yet.

Functions still raising `NotImplementedError` are not failures — they are
unstarted. List them separately as "ещё не написаны" and do not analyze them.

Close by naming the single next function to attack, and stop. No comparison,
no review.

### If everything passed

Say so plainly, then continue to Step 2.

## Step 2 — Metric comparison

```bash
python3 learning-exercises/compare.py <lesson>
```

Show the table as-is. Then read it for them, because the numbers alone do
not say what to do:

- Anything flagged red (≥2× slower) — explain the actual cause in their
  code, not the generic advice.
- Yellow (1.15–2×) — mention it only if the cause is something worth
  learning; otherwise say it is noise at this scale.
- Green everywhere — say that too. Do not invent problems.
- Every `ruff` finding gets one line: what the rule means and why it matters
  here.

Timing noise is real. Do not build an argument on a single 1.2× difference.

## Step 3 — Read the code

The comparison catches what is measurable. This step catches what is not.
Read their `exercise.py` in full and comment on **how** it is written:

- an earlier function they already wrote, re-implemented by hand instead of
  called
- two passes over the data where one would do
- work inside a loop that does not change between iterations
- `range(len(x))` where iterating the elements directly reads better
- a manual accumulator where a comprehension is clearer
- quadratic work hidden in a nested loop
- a name that hides what the variable is

Lead with what is genuinely good — be specific, not encouraging-noise. Then
each improvement with its reason: speed, readability, or fewer places to
break. If the code is clean, say it is clean.

Then point at `solution.py` and name the one or two places where it made a
different choice than they did, and why. If their version is better than the
reference, say that outright.

## Step 4 — Close

Two lines: what state they are in (all green / N left), and the single next
action — the next function, the next lesson via `/learn`, or the review of a
specific concept they clearly have not internalized.
