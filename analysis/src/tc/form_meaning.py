"""Stimuli for the form-vs-meaning separation (experiment 017).

Reuses ``factorial.py``: four binary attributes, three templates, three synonyms
per value.  Meaning is the scene (the attribute values); form is
(template, synonym).  Nothing here depends on a model.
"""

from __future__ import annotations

import numpy as np

from .factorial import ATTRS, build_corpus, prompt


def build_items(attrs=None) -> list[dict]:
    """Return the controlled corpus with the probe appended to every text."""
    attrs = ATTRS if attrs is None else attrs
    items = []
    for c in build_corpus(attrs):
        items.append({
            "text": prompt(c["text"]),
            "scene": tuple(c["values"][a] for a in attrs),
            "template": c["template"],
            "syn": c["syn"],
        })
    return items


def labels(items) -> tuple[np.ndarray, np.ndarray]:
    """Return integer ``(meaning, form)`` labels for the items.

    Meaning is the scene; form is the (template, synonym) variant.
    """
    scenes = sorted({it["scene"] for it in items})
    forms = sorted({(it["template"], it["syn"]) for it in items})
    s_index = {s: i for i, s in enumerate(scenes)}
    f_index = {f: i for i, f in enumerate(forms)}
    y_meaning = np.asarray([s_index[it["scene"]] for it in items], dtype=int)
    y_form = np.asarray([f_index[(it["template"], it["syn"])] for it in items], dtype=int)
    return y_meaning, y_form
