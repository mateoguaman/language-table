#!/usr/bin/env python3
"""Test whether curly apostrophes cause divergent or pathological tokenization.

Background: a training run crashed with NaN actions on a single env whose goal
contained a curly (right single quotation mark, U+2019) apostrophe:

  "push the yellow pentagon directly and smoothly toward the green star\u2019s position"

The CLIP regex splits on ASCII 's but NOT on \u2019s, so the curly variant may
produce OOV tokens or wildly different token IDs.
"""

from clip.simple_tokenizer import default_bpe, SimpleTokenizer

from language_table.common import clip_tokenizer

import numpy as np
import tensorflow as tf


# ── helpers ──────────────────────────────────────────────────────────────────

def _simple_tokenize(tokenizer, texts, context_length=77):
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    all_tokens = [
        [sot_token] + tokenizer.encode(text) + [eot_token] for text in texts
    ]
    result = np.zeros((len(all_tokens), context_length), dtype=int)
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            raise RuntimeError(f"Input too long: {texts[i]}")
        result[i, : len(tokens)] = np.asarray(tokens)
    return result


VOCAB_SIZE = 49408  # CLIP vocab size (used for start/end tokens)


class CurlyApostropheTest(tf.test.TestCase):
    """Check that curly apostrophes don't silently produce garbage tokens."""

    def setUp(self):
        super().setUp()
        self.vocab_lookup = clip_tokenizer.create_vocab(bpe_path=default_bpe())
        self.tokenizer = clip_tokenizer.ClipTokenizer(self.vocab_lookup)
        self.simple_tokenizer = SimpleTokenizer(default_bpe())

    # ── 1. exact crash string ────────────────────────────────────────────

    def test_crash_string_curly_apostrophe(self):
        """The exact goal from the NaN crash."""
        curly = "push the yellow pentagon directly and smoothly toward the green star\u2019s position"
        tokens = clip_tokenizer.tokenize_text(curly, self.tokenizer).numpy()

        # Must not be all-zero (which would mean complete tokenization failure)
        self.assertGreater(np.count_nonzero(tokens), 2,
                           "Tokenized to almost nothing")

        # Start token must be present
        self.assertEqual(tokens[0, 0], VOCAB_SIZE - 2,
                         "Missing <|startoftext|>")

        # No token ID should exceed vocab size (OOV bucket = vocab_size)
        self.assertTrue(np.all(tokens <= VOCAB_SIZE),
                        f"Token IDs exceed vocab: max={tokens.max()}")

        print(f"\nCurly apostrophe tokens (non-zero): "
              f"{tokens[0][tokens[0] != 0].tolist()}")

    # ── 2. curly vs straight: TF tokenizer ───────────────────────────────

    def test_curly_vs_straight_tf_tokenizer(self):
        """Compare tokenization of curly vs straight apostrophe."""
        straight = "push the yellow pentagon directly and smoothly toward the green star's position"
        curly = "push the yellow pentagon directly and smoothly toward the green star\u2019s position"

        tok_straight = clip_tokenizer.tokenize_text(straight, self.tokenizer).numpy()
        tok_curly = clip_tokenizer.tokenize_text(curly, self.tokenizer).numpy()

        print(f"\nStraight tokens: {tok_straight[0][tok_straight[0] != 0].tolist()}")
        print(f"Curly    tokens: {tok_curly[0][tok_curly[0] != 0].tolist()}")
        print(f"Tokens match: {np.array_equal(tok_straight, tok_curly)}")

        if not np.array_equal(tok_straight, tok_curly):
            diff_mask = tok_straight[0] != tok_curly[0]
            diff_positions = np.where(diff_mask)[0]
            print(f"Differ at positions: {diff_positions.tolist()}")
            for pos in diff_positions:
                print(f"  pos {pos}: straight={tok_straight[0, pos]}  curly={tok_curly[0, pos]}")

    # ── 3. curly vs straight: CLIP SimpleTokenizer ───────────────────────

    def test_curly_vs_straight_simple_tokenizer(self):
        """Same comparison using the reference CLIP SimpleTokenizer."""
        straight = "push the yellow pentagon directly and smoothly toward the green star's position"
        curly = "push the yellow pentagon directly and smoothly toward the green star\u2019s position"

        tok_straight = _simple_tokenize(self.simple_tokenizer, [straight])
        tok_curly = _simple_tokenize(self.simple_tokenizer, [curly])

        print(f"\n[SimpleTokenizer]")
        print(f"Straight tokens: {tok_straight[0][tok_straight[0] != 0].tolist()}")
        print(f"Curly    tokens: {tok_curly[0][tok_curly[0] != 0].tolist()}")
        print(f"Tokens match: {np.array_equal(tok_straight, tok_curly)}")

    # ── 4. TF vs CLIP divergence on the crash string ─────────────────────

    def test_tf_vs_clip_divergence_on_crash_string(self):
        """Check if TF tokenizer diverges from CLIP on the curly string."""
        curly = "push the yellow pentagon directly and smoothly toward the green star\u2019s position"

        tf_tokens = clip_tokenizer.tokenize_text(curly, self.tokenizer).numpy()
        clip_tokens = _simple_tokenize(self.simple_tokenizer, [curly])

        match = np.array_equal(tf_tokens, clip_tokens)
        print(f"\nTF vs CLIP match on curly string: {match}")
        if not match:
            diff_mask = tf_tokens[0] != clip_tokens[0]
            diff_positions = np.where(diff_mask)[0]
            print(f"Differ at positions: {diff_positions.tolist()}")
            for pos in diff_positions:
                print(f"  pos {pos}: TF={tf_tokens[0, pos]}  CLIP={clip_tokens[0, pos]}")

    # ── 5. OOV check: does the curly apostrophe produce OOV tokens? ──────

    def test_oov_tokens_curly(self):
        """OOV tokens get ID = vocab_size (the OOV bucket). Check for them."""
        curly = "push the yellow pentagon directly and smoothly toward the green star\u2019s position"
        tokens = clip_tokenizer.tokenize_text(curly, self.tokenizer).numpy()

        oov_mask = tokens == VOCAB_SIZE
        n_oov = int(oov_mask.sum())
        print(f"\nOOV token count: {n_oov}")
        if n_oov > 0:
            oov_positions = np.where(oov_mask[0])[0]
            print(f"OOV at positions: {oov_positions.tolist()}")
        self.assertEqual(n_oov, 0, f"Found {n_oov} OOV tokens for curly string")

    # ── 6. batch: mix of curly and straight in same batch ────────────────

    def test_mixed_batch(self):
        """Tokenize a batch mixing curly and straight — check for anomalies."""
        goals = [
            "push the blue cube next to the yellow pentagon",
            "push the yellow pentagon directly and smoothly toward the green star's position",
            "push the yellow pentagon directly and smoothly toward the green star\u2019s position",
            "slide the green star to the red moon",
        ]
        tokens = clip_tokenizer.tokenize_text(goals, self.tokenizer).numpy()

        for i, goal in enumerate(goals):
            non_zero = tokens[i][tokens[i] != 0]
            has_oov = int((tokens[i] == VOCAB_SIZE).sum())
            print(f"  [{i}] len={len(non_zero)} oov={has_oov} : {goal[:60]}...")

        # The curly variant should not have OOV when others don't
        oov_per_row = (tokens == VOCAB_SIZE).sum(axis=1)
        print(f"OOV per row: {oov_per_row.tolist()}")

    # ── 7. more unicode edge cases ───────────────────────────────────────

    def test_various_unicode_apostrophes(self):
        """Test a range of unicode characters that look like apostrophes."""
        base = "the green star{}s position"
        variants = {
            "ASCII '": base.format("'"),           # U+0027
            "curly \u2019": base.format("\u2019"),  # U+2019 RIGHT SINGLE QUOTATION MARK
            "backtick `": base.format("`"),          # U+0060
            "acute \u00b4": base.format("\u00b4"),   # U+00B4 ACUTE ACCENT
            "modifier \u02bc": base.format("\u02bc"),# U+02BC MODIFIER LETTER APOSTROPHE
        }

        baseline = None
        for label, text in variants.items():
            tokens = clip_tokenizer.tokenize_text(text, self.tokenizer).numpy()
            non_zero = tokens[0][tokens[0] != 0]
            has_oov = int((tokens[0] == VOCAB_SIZE).sum())
            print(f"  {label:>16}: len={len(non_zero)} oov={has_oov} tokens={non_zero.tolist()}")
            if baseline is None:
                baseline = tokens
            else:
                if not np.array_equal(tokens, baseline):
                    print(f"    ^ DIFFERS from ASCII baseline")


if __name__ == "__main__":
    tf.test.main()
