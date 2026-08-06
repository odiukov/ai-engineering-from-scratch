"""T5 span corruption + BART denoising noise functions.

Pure stdlib. Shows how encoder-decoder models turn any input into
a supervised (corrupted_input -> clean_spans) training pair.
"""

import random


# Seeded default so every noise function is reproducible when the caller does
# not pass an `rng` — the round-trip check in Step 2 has to be repeatable.
DEFAULT_SEED = 0


def sentinel(i):
    return f"<extra_id_{i}>"


def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """T5-style span corruption.

    Exactly `round(mask_rate * len(tokens))` tokens end up inside spans, so the
    realized mask fraction matches `mask_rate`.

    Returns (corrupted_source, decoder_target) as lists of tokens (strings).
    """
    if rng is None:
        rng = random.Random(DEFAULT_SEED)
    n = len(tokens)
    n_mask = max(1, int(round(n * mask_rate)))
    n_spans = max(1, int(round(n_mask / mean_span)))
    # Average span length that spends the whole budget in `n_spans` spans.
    mean_len = n_mask / n_spans
    starts = []
    used = [False] * n
    span_lengths = []
    remaining = n_mask
    # Keep placing non-overlapping spans until the budget is spent. `n_spans` is
    # a target, not a cap: a span that gets truncated by an already-masked
    # neighbour is followed by another span, so `remaining` always reaches 0.
    while remaining > 0:
        want = max(1, int(round(rng.gauss(mean_len, 1.0))))
        placed = False
        random_order = list(range(n))
        rng.shuffle(random_order)
        for start in random_order:
            if used[start]:
                continue
            # Longest run of free positions starting at `start`.
            end = start
            while end < n and not used[end]:
                end += 1
            length = min(want, remaining, end - start)
            if length < 1:
                continue
            for i in range(start, start + length):
                used[i] = True
            starts.append(start)
            span_lengths.append(length)
            remaining -= length
            placed = True
            break
        if not placed:
            break  # every position is already masked

    ordered = sorted(zip(starts, span_lengths), key=lambda x: x[0])

    source = []
    target = []
    prev_end = 0
    for idx, (start, length) in enumerate(ordered):
        source.extend(tokens[prev_end:start])
        source.append(sentinel(idx))
        target.append(sentinel(idx))
        target.extend(tokens[start:start + length])
        prev_end = start + length
    source.extend(tokens[prev_end:])
    target.append(sentinel(len(ordered)))  # closing sentinel
    return source, target


def round_trip(source, target):
    """Reconstruct original by replacing sentinels in source with corresponding target spans."""
    # Parse target into sentinel->span map
    spans = {}
    current_key = None
    current_span = []
    for tok in target:
        if tok.startswith("<extra_id_"):
            if current_key is not None:
                spans[current_key] = current_span
            current_key = tok
            current_span = []
        else:
            current_span.append(tok)
    # Flush the final span. Without this, the last sentinel is only recorded
    # when a closing marker happens to follow it.
    if current_key is not None:
        spans[current_key] = current_span
    out = []
    for tok in source:
        if tok.startswith("<extra_id_"):
            out.extend(spans.get(tok, []))
        else:
            out.append(tok)
    return out


def token_mask(tokens, rate=0.15, rng=None, mask_token="<mask>"):
    if rng is None:
        rng = random.Random(DEFAULT_SEED)
    return [mask_token if rng.random() < rate else t for t in tokens]


def token_delete(tokens, rate=0.15, rng=None):
    if rng is None:
        rng = random.Random(DEFAULT_SEED)
    return [t for t in tokens if rng.random() >= rate]


def text_infill(tokens, rate=0.15, mean_span=3.0, rng=None, mask_token="<mask>"):
    """BART text infill: mask spans with a SINGLE mask; decoder infers length.

    Exactly `round(rate * len(tokens))` tokens are swallowed by masks, so the
    realized corruption fraction equals `rate`.
    """
    if rng is None:
        rng = random.Random(DEFAULT_SEED)
    out = []
    i = 0
    n = len(tokens)
    budget = int(round(n * rate))
    while i < n:
        left = n - i
        # Spend the budget proportionally to how much text is left. The ratio
        # rises to 1 as `i` approaches `n`, so the budget is always fully spent.
        if budget > 0 and rng.random() < budget / left:
            span_len = max(1, min(int(round(rng.gauss(mean_span, 1.0))), budget, left))
            out.append(mask_token)
            budget -= span_len
            i += span_len
        else:
            out.append(tokens[i])
            i += 1
    return out


def sentence_permute(sentences, rng=None):
    if rng is None:
        rng = random.Random(DEFAULT_SEED)
    sents = list(sentences)
    rng.shuffle(sents)
    return sents


def document_rotate(tokens, rng=None):
    if rng is None:
        rng = random.Random(DEFAULT_SEED)
    if len(tokens) <= 1:
        return tokens
    pivot = rng.randrange(1, len(tokens))
    return tokens[pivot:] + tokens[:pivot]


def main():
    rng = random.Random(42)

    sentence = (
        "the quick brown fox jumps over the lazy dog a stitch in time saves nine "
        "language models learn statistical patterns subword tokenization helps rare words"
    ).split()

    print("=== T5 span corruption ===")
    source, target = corrupt_spans(sentence, mask_rate=0.20, mean_span=3.0, rng=rng)
    print("corrupted source:")
    print("  " + " ".join(source))
    print()
    print("decoder target:")
    print("  " + " ".join(target))
    print()
    reconstructed = round_trip(source, target)
    print("reconstruction matches original:",
          "YES" if reconstructed == sentence else "NO")
    if reconstructed != sentence:
        print("  reconstructed: " + " ".join(reconstructed))

    print()
    print("=== BART noise functions ===")
    print("original: " + " ".join(sentence[:14]))
    print()
    print("token mask:     " + " ".join(token_mask(sentence[:14], rate=0.2, rng=random.Random(1))))
    print("token delete:   " + " ".join(token_delete(sentence[:14], rate=0.2, rng=random.Random(2))))
    print("text infill:    " + " ".join(text_infill(sentence[:14], rate=0.3, rng=random.Random(3))))

    sentences = [
        ["the", "quick", "brown", "fox"],
        ["a", "stitch", "in", "time"],
        ["language", "models", "learn", "patterns"],
    ]
    perm = sentence_permute(sentences, rng=random.Random(4))
    print("sentence permute:")
    for s in perm:
        print("  " + " ".join(s))

    print()
    print("document rotate: " + " ".join(document_rotate(sentence[:14], rng=random.Random(5))))


if __name__ == "__main__":
    main()
