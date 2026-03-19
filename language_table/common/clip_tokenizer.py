# coding=utf-8
# Copyright 2024 The Language Tale Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An implementation of an in-graph TF Tokenizer.

This is based on the SimpleTokenizer implementation from CLIP.

Note: while this returns similar results for many strings used in
language_table, there is no guarantee that this returns equivalent tokens for
every text input in general.  The ``tokenize_text`` entry-point applies the
same ``ftfy`` + ``html.unescape`` cleanup that CLIP's reference
SimpleTokenizer uses, so most common Unicode variants (curly quotes,
smart dashes, etc.) are handled.  See test cases for more details.

General usage is:
```
vocab_lookup = create_vocab(bpe_path=default_bpe())
tokenizer = ClipTokenizer(vocab_lookup)
tokens = tokenize_text("example input text", tokenizer)
```
"""

import gzip
import html
import re

import ftfy

from clip.simple_tokenizer import bytes_to_unicode
from clip.simple_tokenizer import default_bpe

import tensorflow as tf
import tensorflow_text as tf_text


class ClipTokenizer(tf_text.TokenizerWithOffsets):
  """An in-graph TF implementation similar to SimpleTokenizer."""

  def __init__(
      self,
      vocab_lookup_table,
      max_bytes_per_word=100,
      max_chars_per_token=None,
      token_out_type=tf.int64,
      unknown_token="[UNK]",
      split_unknown_characters=False,
      lower_case=False,
      keep_whitespace=False,
      normalization_form=None,
      preserve_unused_token=False,
  ):
    super(ClipTokenizer, self).__init__()

    self._re = r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"""  # pylint: disable=line-too-long
    self._wordpiece_tokenizer = tf_text.WordpieceTokenizer(
        vocab_lookup_table,
        suffix_indicator="",
        max_bytes_per_word=max_bytes_per_word,
        max_chars_per_token=max_chars_per_token,
        token_out_type=token_out_type,
        unknown_token=unknown_token,
        split_unknown_characters=split_unknown_characters,
    )

  def tokenize_with_offsets(self, text_input):
    # Do basic cleaning of text inputs.
    # the CLIP simple tokenizer does the following:
    # 1. text = html.unescape(html.unescape(text))
    # (we skip this for tf, which means *escaped HTML inputs will not return
    #  the same tokens as SimpleTokenizer.)
    # 2. text = text.strip()
    text_input = tf.strings.strip(text_input)
    # 3. text = re.sub(r'\s+', ' ', text)
    text_input = tf.strings.regex_replace(text_input, r"\s+", " ")
    # 4. text = text.strip()
    text_input = tf.strings.strip(text_input)
    # 5. text = text.lower()
    text_input = tf.strings.lower(text_input)

    tokens, begin, _ = tf_text.regex_split_with_offsets(
        text_input,
        delim_regex_pattern=self._re + r"|\s",
        keep_delim_regex_pattern=self._re,
    )

    # Add the end </w> character to each token after the regex split.
    num_tokens = tf.shape(tokens.values)[0]
    end_char = tf.tile(["</w>"], [num_tokens])
    tokens_with_end = tf.strings.join([tokens, end_char])

    begin = tf.cast(begin, tf.int64)

    # The wordpiece tokenizer has the same logic as the bpe() routine.
    (
        wordpieces,
        wp_begin,
        wp_end,
    ) = self._wordpiece_tokenizer.tokenize_with_offsets(tokens_with_end)

    begin_expanded = tf.expand_dims(begin, axis=2)
    final_begin = begin_expanded + wp_begin
    final_end = begin_expanded + wp_end

    return wordpieces, final_begin, final_end

  def tokenize(self, text_input):
    tokens, _, _ = self.tokenize_with_offsets(text_input)
    return tokens.flat_values


def create_vocab(*, bpe_path=default_bpe()):
  """Creates the input vocabulary table for the tokenizer."""
  with tf.io.gfile.GFile(bpe_path, "rb") as f:
    merges = gzip.open(f).read().decode("utf-8").split("\n")
  merges = merges[1 : 49152 - 256 - 2 + 1]
  merges = [tuple(merge.split()) for merge in merges]
  vocab = list(bytes_to_unicode().values())
  # add '##' prefix to indicate sub-word
  vocab = vocab + [v + "</w>" for v in vocab]
  for merge in merges:
    vocab.append("".join(merge))
  vocab.extend(["<|startoftext|>", "<|endoftext|>"])
  bpe_lookup = tf.lookup.StaticVocabularyTable(
      num_oov_buckets=1,
      initializer=tf.lookup.KeyValueTensorInitializer(
          keys=vocab, values=tf.range(len(vocab), dtype=tf.int64)
      ),
  )
  return bpe_lookup


def basic_clean(text):
  """Match the preprocessing from CLIP's SimpleTokenizer.

  The reference SimpleTokenizer (clip.simple_tokenizer) applies
  ``ftfy.fix_text`` followed by double ``html.unescape`` before
  tokenizing.  The TF ClipTokenizer historically skipped this step,
  which meant that any Unicode character outside the BPE byte-vocab
  (e.g. curly quotes U+2018/2019, em-dashes U+2014, zero-width
  spaces, etc.) would produce an OOV token (id 49408).  Since the
  CLIP embedding table has only 49408 entries (indices 0-49407), an
  OOV index causes an out-of-bounds lookup that silently returns
  garbage in JAX, leading to NaN actions downstream.

  Applying the same cleanup here closes the parity gap.
  """
  text = ftfy.fix_text(text)
  text = html.unescape(html.unescape(text))
  return text.strip()


def _clean_for_clip(text):
  """Normalize text inputs before TF tokenization.

  Accepts a single string, a list of strings, or a TF string tensor.
  Returns a TF string tensor with the same shape.
  """
  if isinstance(text, (str, bytes)):
    return basic_clean(text if isinstance(text, str)
                       else text.decode("utf-8", errors="replace"))

  if isinstance(text, (list, tuple)):
    return [basic_clean(t) for t in text]

  # TF tensor path — apply a best-effort regex substitution for the
  # most common OOV-producing characters.  This is not as thorough as
  # ftfy but covers the characters that actually appear in practice
  # (LLM-generated goal strings).  A full ftfy pass would require
  # tf.py_function which breaks graph tracing in the training
  # pipeline; the regex approach keeps the op inside the TF graph.
  _UNICODE_REPLACEMENTS = [
      # Curly single quotes / apostrophe variants → ASCII apostrophe
      (r"[\u2018\u2019\u201A\u201B\u02BC\uFF07]", "'"),
      # Curly double quotes → ASCII double quote
      (r"[\u201C\u201D\u201E\u201F\uFF02]", '"'),
      # En-dash / em-dash → hyphen
      (r"[\u2013\u2014]", "-"),
      # Ellipsis → three dots
      (r"\u2026", "..."),
      # Prime / double prime → apostrophe / double quote
      (r"\u2032", "'"),
      (r"\u2033", '"'),
      # Minus sign → hyphen-minus
      (r"\u2212", "-"),
      # Non-breaking space → regular space
      (r"\u00A0", " "),
      # Zero-width characters → empty
      (r"[\u200B\u200C\u200D\uFEFF\u00AD]", ""),
  ]
  for pattern, replacement in _UNICODE_REPLACEMENTS:
    text = tf.strings.regex_replace(text, pattern, replacement)
  return text


def tokenize_text(text, tokenizer, vocab_size=49408):
  """Tokenizes the input text given a tokenizer."""
  text = _clean_for_clip(text)
  tokens, start_idx, end_idx = tokenizer.tokenize_with_offsets(text)
  # flatten sub-word tokenization
  tokens = tokens.merge_dims(-2, -1)
  start_idx = start_idx.merge_dims(-2, -1)
  end_idx = end_idx.merge_dims(-2, -1)
  count = tokens.bounding_shape()[0]
  # pad sparse tokens tensor with start and end tokens
  starts = tf.cast(tf.fill([count, 1], vocab_size - 2), dtype=tf.int64)
  ends = tf.cast(tf.fill([count, 1], vocab_size - 1), dtype=tf.int64)
  tokens = tf.concat([starts, tokens, ends], axis=1)
  # convert sparse tensor to zero padded dense tensor
  tokens = tokens.to_tensor(shape=(count, 77))
  return tokens
