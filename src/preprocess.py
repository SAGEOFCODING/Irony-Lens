"""Social-media text normalisation.

The same function is used at training time and at prediction time, so the
model never sees text that was cleaned differently from what it learned on.

Design rule: remove noise, keep sarcasm signals.
  * Noise   -> URLs, user names, HTML entities, label hashtags (#sarcasm).
  * Signals -> punctuation (!!!, ?!), emojis, capitalisation, elongation
               ("sooo"), hashtag words. These are NOT removed.
"""
import html
import re

import emoji

URL_RE = re.compile(r"https?://\S+|www\.\S+")
USER_RE = re.compile(r"@\w+")
# Hashtags people use to *announce* sarcasm. Many datasets were collected by
# searching for them, so leaving them in would leak the label into the input.
LABEL_TAG_RE = re.compile(r"#(sarcasm|sarcastic|irony|ironic|not)\b", re.IGNORECASE)
HASHTAG_RE = re.compile(r"#(\w+)")
# 4+ repeats of a letter -> 3 ("soooooo" -> "sooo"): keeps the emphasis signal
# while stopping every length from becoming a separate vocabulary entry.
ELONGATION_RE = re.compile(r"([a-zA-Z])\1{3,}")
SPACE_RE = re.compile(r"\s+")


def clean_text(text, demojize=True):
    """Normalise one post.

    demojize=True turns emojis into words (":face_with_rolling_eyes:") so that
    bag-of-words and LSTM models can use them as tokens. Transformers
    pre-trained on tweets already understand raw emojis, so they pass False.
    """
    text = html.unescape(str(text))
    text = URL_RE.sub("http", text)
    text = USER_RE.sub("@user", text)
    text = LABEL_TAG_RE.sub("", text)
    text = HASHTAG_RE.sub(r"\1", text)
    if demojize:
        text = emoji.demojize(text, delimiters=(" :", ": "))
    text = ELONGATION_RE.sub(r"\1\1\1", text)
    return SPACE_RE.sub(" ", text).strip()


def clean_batch(texts):
    """List version of clean_text, used inside the scikit-learn pipeline."""
    return [clean_text(t) for t in texts]
