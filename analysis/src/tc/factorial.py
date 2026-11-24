"""Factorial concept corpus (experiment 002, phase 0).

Four binary attributes are rendered into a fixed syntactic frame. A *minimal
pair* for attribute ``a`` is two sentences that differ only in the phrase of
``a`` (same template, same synonym index, same values of the other attributes),
so the contrast isolates a single attribute by construction.

Nothing here depends on a model: pure text generation, testable offline.
"""

from __future__ import annotations

import itertools

ATTRS = ["diet", "sociality", "body", "nest"]

VALUES = {
    "diet": {
        0: ["feeds on nectar and pollen", "collects nectar from flowers", "gathers pollen from blossoms"],
        1: ["hunts other insects", "preys on smaller bugs", "catches other insects to eat"],
    },
    "sociality": {
        0: ["lives in a large colony", "lives in a colony with others", "dwells in a crowded colony"],
        1: ["lives mostly alone", "lives a solitary life", "usually lives on its own"],
    },
    "body": {
        0: ["has a hairy body", "has a fuzzy body", "has a furry body"],
        1: ["has a smooth body", "has a bare body", "has a sleek body"],
    },
    "nest": {
        0: ["builds cells of wax", "makes honeycomb from wax", "builds a nest of wax"],
        1: ["builds nests of paper", "makes nests from chewed wood", "builds a nest of paper"],
    },
    # semantically unrelated to the insect; used as a transfer / unrelated control
    "setting": {
        0: ["near a lake", "by a calm lake", "close to a lake"],
        1: ["near a mountain", "by a tall mountain", "close to a mountain"],
    },
}

ATTRIBUTES_ALL = ATTRS + ["setting"]

TEMPLATES = [
    "This insect {diet}, {sociality}, {body}, and {nest}.",
    "An insect that {diet}, {sociality}, {body}, and {nest}.",
    "Consider an insect: it {diet}, {sociality}, {body}, and {nest}.",
]

N_SYN = 3
PROBE = "It is a"


def render(values: dict, template_id: int, syn_idx: int, attrs=None) -> str:
    attrs = ATTRS if attrs is None else attrs
    phrases = {a: VALUES[a][values[a]][syn_idx] for a in ATTRS}
    text = TEMPLATES[template_id].format(**phrases)
    if "setting" in attrs:
        text += f" It is found {VALUES['setting'][values['setting']][syn_idx]}."
    return text


def _scenes(attrs):
    return [dict(zip(attrs, combo)) for combo in itertools.product([0, 1], repeat=len(attrs))]


def build_corpus(attrs=None):
    """Return list of {text, values, template, syn} over the full factorial."""
    attrs = ATTRS if attrs is None else attrs
    corpus = []
    for tpl in range(len(TEMPLATES)):
        for values in _scenes(attrs):
            for syn in range(N_SYN):
                corpus.append({"text": render(values, tpl, syn, attrs), "values": values,
                               "template": tpl, "syn": syn})
    return corpus


def build_pairs(attrs=None):
    """Return minimal pairs: {attr, base, source, values_base, values_source, template, syn}.

    base has attr=0, source has attr=1; every other attribute is held fixed.
    """
    attrs = ATTRS if attrs is None else attrs
    pairs = []
    for attr in attrs:
        rest = [a for a in attrs if a != attr]
        for tpl in range(len(TEMPLATES)):
            for syn in range(N_SYN):
                for combo in itertools.product([0, 1], repeat=len(rest)):
                    other_vals = dict(zip(rest, combo))
                    vb = dict(other_vals); vb[attr] = 0
                    vs = dict(other_vals); vs[attr] = 1
                    pairs.append({
                        "attr": attr, "base": render(vb, tpl, syn, attrs),
                        "source": render(vs, tpl, syn, attrs),
                        "values_base": vb, "values_source": vs, "template": tpl, "syn": syn,
                    })
    return pairs


def prompt(text: str, probe: str = PROBE) -> str:
    return f"{text}\n\n{probe}"
