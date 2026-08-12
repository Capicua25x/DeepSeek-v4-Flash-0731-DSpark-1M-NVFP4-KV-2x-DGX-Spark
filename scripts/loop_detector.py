#!/usr/bin/env python3
"""Tell a reasoning LOOP apart from a heavy TAIL, from the text alone.

Both produce the same user-visible symptom -- `finish_reason=length`, empty
content -- but they are different failures and want different fixes:

  heavy tail : the model is still saying new things, it just says a lot of them.
               Fix by raising max_tokens.
  loop       : the model recycles material it has already emitted, forever.
               Raising max_tokens changes nothing except the bill.

WHAT IT MEASURES

  novelty(window) = fraction of word 8-grams in this window that have not
                    appeared anywhere earlier in the same trace.

  A healthy trace keeps novelty high to the end. A looping trace collapses and
  never recovers.

WHY NOT BLOCK UNIQUENESS

  We first measured "unique 120-character blocks" and it told us the runaways
  were NOT loops. That was wrong, and wrong in a way worth warning about: the
  loop is *templated*, not verbatim -- a small set of stock phrases recombined
  with one varying element each pass. On the same three traces:

      unique 120-char blocks : 22% / 92% / 66%
      unique word 8-grams    :  3.4% / 4.0% / 2.8%

  One of those traces looks almost perfectly novel at block granularity and is
  96% recycled at phrase level. Any fixed-window uniqueness check reports a
  templated loop as fresh text. Count recycled n-grams, or unique lines.

  (Note the contrast needs NON-overlapping blocks; with stride-1 blocks both
  metrics agree at ~3-5%, which is another way to avoid the trap.)

CALIBRATION

  Measured over three captured non-terminating traces of ~100k tokens each:
  novelty collapses at token ~7.9k / ~13.8k / ~29.7k and never recovers --
  afterwards it oscillates between 0 and 0.6%, never above. It does not sit at
  exactly zero, so do not test for `== 0`; the threshold below is 2%.

USAGE

  python3 loop_detector.py trace.txt [trace2.txt ...]
  cat trace.txt | python3 loop_detector.py

  Feed it the REASONING stream (the `reasoning` field, not `content`).
"""
import re
import sys
import zlib

WINDOW = 4000       # characters per window
THRESHOLD = 0.02    # novelty below this counts as exhausted
CONSECUTIVE = 3     # windows in a row before calling it a loop
CHARS_PER_TOKEN = 4.0   # rough; only used to report a token index


def analyse(text: str):
    seen: set = set()
    rows = []
    for pos in range(0, len(text), WINDOW):
        w = text[pos:pos + WINDOW]
        if len(w) < WINDOW // 2:
            break
        words = re.findall(r"\w+", w.lower())
        shingles = {tuple(words[i:i + 8]) for i in range(max(0, len(words) - 7))}
        novelty = len(shingles - seen) / len(shingles) if shingles else 0.0
        seen |= shingles
        compression = len(zlib.compress(w.encode(), 6)) / max(1, len(w.encode()))
        rows.append((pos, novelty, compression))
    return rows


def verdict(rows):
    dry = 0
    for pos, novelty, _ in rows:
        dry = dry + 1 if novelty < THRESHOLD else 0
        if dry >= CONSECUTIVE:
            return pos - WINDOW * (CONSECUTIVE - 1)
    return None


def report(name: str, text: str) -> None:
    rows = analyse(text)
    if not rows:
        print(f"{name}: too short to judge ({len(text)} chars)")
        return
    onset = verdict(rows)
    print(f"\n== {name}  ({len(text):,} chars, ~{int(len(text)/CHARS_PER_TOKEN):,} tokens)")
    print(f"   {'tok~':>9}  {'novelty':>7}  {'compress':>8}")
    step = max(1, len(rows) // 12)
    for pos, novelty, comp in rows[::step]:
        print(f"   {int(pos/CHARS_PER_TOKEN):>9,}  {novelty:>7.2%}  {comp:>8.3f}")
    if onset is None:
        print("   -> HEAVY TAIL: novelty never collapses. The model is still "
              "producing new material; raise max_tokens.")
    else:
        tail = [n for p, n, _ in rows if p >= onset]
        print(f"   -> LOOP from token ~{int(onset/CHARS_PER_TOKEN):,} "
              f"(novelty stays below {max(tail):.2%} thereafter). "
              f"Raising max_tokens will not help.")


def main() -> int:
    args = sys.argv[1:]
    if not args:
        data = sys.stdin.read()
        if not data.strip():
            print(__doc__)
            return 1
        report("<stdin>", data)
        return 0
    for path in args:
        with open(path, encoding="utf-8", errors="replace") as fh:
            report(path, fh.read())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
