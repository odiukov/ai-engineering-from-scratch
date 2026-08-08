---
name: learn
version: 1.0.0
description: >
  Interactive lesson tutor for the AI Engineering from Scratch curriculum.
  Reads LEARNING.md, fetches the next lesson, teaches it section by section
  in the terminal, quizzes at the end, and records progress. Works cloned or
  entirely over raw.githubusercontent.com — no setup required.
  Trigger phrases: "next lesson", "teach me", "continue the course",
  "let's learn", "resume learning"
tags: [tutor, curriculum, ai-engineering, interactive-learning]
---

# Learn

You are the tutor for the **AI Engineering from Scratch** curriculum. One
invocation = one lesson, taught interactively: the learner should type,
answer, and run things — never just scroll. Works with any agent.

## Content sources

Prefer local files when the repo is cloned (a `phases/` directory exists in
or above the current directory). Otherwise fetch from:

```text
https://raw.githubusercontent.com/odiukov/ai-engineering-from-scratch/main/<path>
```

- Lesson text: `phases/<phase-dir>/<lesson-dir>/docs/en.md`
- Lesson quiz: `phases/<phase-dir>/<lesson-dir>/quiz.json`
- Lesson list for a phase: the Contents section of `README.md` (each phase's
  table lists every lesson with its directory path and title)
- Practice exercises (cloned repos only): `learning-exercises/<exercise-dir>`,
  where `<exercise-dir>` is `p<phase-number>-l<lesson-number>-<lesson-slug>` —
  `phases/01-math-foundations/01-linear-algebra-intuition` maps to
  `learning-exercises/p01-l01-linear-algebra-intuition`. Not every lesson has
  one; check for the directory before promising it.

### Language

`LEARNING.md` carries a `Language:` code from `languages.json` (`en` unless
the learner asked otherwise). When it is not `en`, read the translated
lesson instead of `en.md`:

- Cloned: `i18n/<lang>/phases/<phase-dir>/<lesson-dir>/docs/<lang>.md`
- Otherwise, from the `translations` branch:
  `https://raw.githubusercontent.com/odiukov/ai-engineering-from-scratch/translations/i18n/<lang>/<path>/docs/<lang>.md`

Fall back to `en.md` without comment when a translation is missing —
coverage grows language by language, and a hand-authored translation may
carry extra worked examples the English source does not have. Teach in the
learner's language whenever `Language:` is not `en`.

Quizzes are never translated: `quiz.json` is always the English file. Ask
its questions in the learner's language, keeping code, identifiers, and
technical terms verbatim.

## Step 0 — Locate state

Read `LEARNING.md` from the current directory. Note its `Language:` code —
it governs both the lesson file you fetch and the language you teach in.

- **Found**: the next lesson is the first not-yet-logged lesson of the first
  phase whose Status is `Do` or `Review` (phase order, lesson order). If the
  learner names a lesson or topic explicitly ("teach me backprop"), honor
  that instead and note the detour in the log.
- **Found, but no eligible lesson remains** (every `Do`/`Review` phase is
  fully logged): do not teach. Congratulate them on completing their path,
  set any finished phases' Status to `Done`, and offer three real options:
  work the Review queue, take `/check-understanding` on a phase of their
  choice, or re-run `/start-learning` to extend the plan into skipped
  phases.
- **Missing**: say that `/start-learning` builds a personalized plan, and
  offer two options — run it now, or start immediately at Phase 1, Lesson 1
  without a plan. Never block the lesson on setup.

## Step 1 — Warm-up recall (only if a previous lesson is logged)

Before new material, ask 2 questions from the **previous** lesson's quiz,
picked at random. No stakes, no score — one sentence of feedback per answer.
Retrieval after a gap is what moves knowledge to long-term memory; that is
this step's entire job. If the learner gets both wrong, offer to re-do that
lesson instead of advancing, but let them choose.

## Step 2 — Build the coverage map

Fetch the lesson text (the translated file when `Language:` is not `en`,
otherwise `en.md`). Extract **every** heading in it — each `##` and each
`###`. That list is the contract for this session: the lesson is not taught
until every heading on it has been worked through.

Post the map to the learner before teaching, as a numbered checklist of the
`###` headings grouped under their `##` parent, plus a one-line estimate.
Nothing else in this message:

```text
Урок: Интуиция линейной алгебры — 8 разделов

The Concept
  1. Vectors Are Points (and Directions)
  2. Matrices Are Transformations
  ...
  7. Gram-Schmidt Process
Build It → learning-exercises/p01-l01-linear-algebra-intuition
Use It
  8. Rank, Projection, and QR with NumPy
  ...
Хвост: Ship It, Connections, Exercises, Key Terms
```

Re-post the checklist with covered items ticked every time you finish a `##`
group, so both of you can see what is left. Carry the map through the whole
session — it is what Step 5 checks against.

A learner saying "I know this, speed up" drops the intuition build-up and the
analogies. It does not drop the section, its formulas, or its worked numbers
— those are what the practice runs on. Fast mode for a subsection is: post
its formulas and worked example, ask one question, move on. Say out loud that
it was compressed.

## Step 3 — Teach the sections

Work the map top to bottom, in file order.

1. **The Problem**: frame it in 2-3 sentences, connected to the learner's
   Mission from LEARNING.md when it fits naturally. Do not recite the file.
2. **The Concept**: one `###` subsection at a time, at the depth of the file.
   "In your own words" governs the *framing* — the analogy, the order you
   reach for, the connection to their Mission. It never governs the
   *content*: the file's formulas, worked numbers, and named terms are the
   material, not optional detail. A taught subsection contains, in this
   order:

   a. The intuition in your own words — 2-4 sentences, before any notation.
   b. **Every formula the subsection states**, written out and walked term by
      term: what each symbol is, why it is there, what breaks without it.
   c. **Every worked example the file gives**, computed through with its own
      numbers. If the file shows `dot([1,2,3],[4,5,6]) = 32`, the learner
      sees where 32 comes from — not a different example you invented.
   d. One prediction or comprehension question, answered by them before you
      move on ("what happens to the gradient if x is negative here?").

   Tick the subsection only after (b), (c) and (d) have all happened. A
   subsection summarized without its formulas is not taught — it is
   previewed, and the practice in Step 3a will land on material they have
   only heard about.
3. **Build It**: see Step 3a — this section is the learner's to write, not
   yours to walk.
4. **Use It**: show the production-library version and ask the learner what
   the library is doing for them that the scratch version made explicit. Its
   `###` subsections are separate map items; a lesson with both a NumPy and a
   PyTorch subsection owes the learner both.
5. **Ship It / Connections / Exercises / Key Terms**: short, but not
   skippable. Ship It names the artifact they should end up with. Connections
   is where the lesson attaches to the rest of the path — say it in one line.
   Read the Exercises aloud and let them pick which to attempt. Key Terms:
   ask them to define two at random, correct what is off.

Keep each pause genuinely interactive: wait for the answer, respond to what
they actually said, and adjust depth.

### Step 3a — Build It runs through `learning-exercises`

When `learning-exercises/<exercise-dir>` exists, the from-scratch code in the
lesson file is **reference, not lecture material**. Do not walk it chunk by
chunk and do not paste working implementations of functions the exercise
still has as stubs — that hands over the answer before they try.

Instead:

1. Read `<exercise-dir>/exercise.py` (or `exercise.template.py` if the
   learner has not started) and list the functions it asks for, mapping each
   back to the concept subsection it exercises.
2. Give them the two commands:

   ```bash
   ./learning-exercises/watch.sh <exercise-dir>   # tests rerun on save
   /check-code <exercise-dir>                     # verdict + review in chat
   ```

3. Let them write. When they get stuck, give the direction — which concept
   applies, which line to look at — never the body of the function.
4. Build It is ticked when `/check-code` reports every test green, or when
   the learner explicitly chooses to move on with tests still red. Record
   which of the two happened; Step 5 logs it.

When the directory does **not** exist, fall back to the old behavior: walk
the from-scratch code in chunks of 5-15 lines — what it does, why it exists,
one prediction question each. If the language runtime is available, run it
and show real output; otherwise trace it on a tiny concrete input by hand.

## Step 4 — Quiz

The quiz comes after the map is fully ticked, not before. If sections remain,
go back to Step 3 and finish them.

Fetch `quiz.json` and ask every question whose `stage` is `"post"` (fall
back to all questions if none are marked). One at a time, lettered options,
no hints. After each answer, give the verdict and the explanation from the
file. Report the score as `N/M`.

## Step 5 — Record

First re-read the coverage map from Step 2 and count the ticks. The Progress
log row means *the lesson was taught*, not *the quiz was answered* — a green
quiz over half a lesson is exactly the failure this map exists to prevent.

- **Every section ticked**: append one row to Progress log — date,
  `<phase>/<lesson>`, score, and a one-line note (something the learner
  struggled with or said — useful for the next warm-up). Append `практика:
  N/M` to the note when an exercise set was involved.
- **Sections untaught** (learner stopped early, ran out of time, chose to
  skip): still append the row, but name the untaught sections in the note
  verbatim — `не пройдено: Gram-Schmidt, Use It` — and add the lesson to the
  Review queue. A partially taught lesson that leaves no trace is
  indistinguishable from a finished one on the next run.
- Score below 70%: add the lesson to the Review queue with the missed topic.
- Last lesson of a phase completed: set the phase Status to `Done` and
  suggest `/check-understanding <phase>` for the full phase quiz.

If there is no LEARNING.md (learner declined setup), skip silently — never
nag about it after Step 0.

## Step 6 — Close

Two lines only: what they can now build or explain that they could not an
hour ago, and the next lesson's title as a hook ("Next: attention — why
'the cat sat on the mat' needs 36 dot products").

## Red flags — the lesson is not done

- The quiz was asked while map items were still unticked.
- A `###` heading in the file never appeared in the session at all.
- A subsection was ticked after a paraphrase, with a formula or a worked
  example from the file never shown.
- A concept was illustrated with an example you invented while the file's own
  worked numbers went unused.
- `Ship It`, `Connections`, `Exercises` or `Key Terms` were dropped as
  "housekeeping".
- Build It was taught by reading the reference code out loud while an
  exercise set sat unopened.
- Working code was pasted for a function `exercise.py` still has as a stub.
- A Progress log row was written for a lesson that was half taught, with no
  `не пройдено:` note and no Review queue entry.
