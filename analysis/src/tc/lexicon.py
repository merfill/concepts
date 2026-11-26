"""WordNet member lexicons for category centrality (experiment 010).

The member set of a category is the transitive hyponym closure of its WordNet
synset (lemmas), optionally extended with the annotation words.  WordNet import
is lazy so the rest of the package (and unit tests) works without the corpus.
"""

from __future__ import annotations

import functools

CATEGORY_SYNSETS = {
    "mammal": "mammal.n.01",
    "bird": "bird.n.01",
    "fruit": "fruit.n.01",
    "vegetable": "vegetable.n.01",
    "furniture": "furniture.n.01",
    "vehicle": "vehicle.n.01",
    "tool": "tool.n.01",
    "weapon": "weapon.n.01",
    "clothing": "clothing.n.01",
    "musical instrument": "musical_instrument.n.01",
}


def normalize(term: str) -> str:
    """Lower-case and turn spaces into underscores (WordNet lemma form)."""
    return str(term).strip().lower().replace(" ", "_")


@functools.lru_cache(maxsize=None)
def hyponym_lemmas(synset_name: str) -> frozenset[str]:
    """All lemma names in the transitive hyponym closure of a synset."""
    from nltk.corpus import wordnet as wn

    root = wn.synset(synset_name)
    seen = {root}
    stack = [root]
    while stack:
        synset = stack.pop()
        for child in synset.hyponyms():
            if child not in seen:
                seen.add(child)
                stack.append(child)
    lemmas = {normalize(name) for synset in seen for name in synset.lemma_names()}
    return frozenset(lemmas)


def member_lemmas(category: str, extra=()) -> set[str]:
    """Member lexicon for a category: WordNet hyponyms plus extra words."""
    if category not in CATEGORY_SYNSETS:
        raise KeyError(f"no WordNet synset mapped for category {category!r}")
    lemmas = set(hyponym_lemmas(CATEGORY_SYNSETS[category]))
    lemmas.update(normalize(word) for word in extra)
    return lemmas
