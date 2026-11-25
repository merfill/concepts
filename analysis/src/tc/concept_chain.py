"""Related-but-distinct concept chains (trajectory of texts).

Each chain is an ordered list of concepts that are *related* but *distinct*
(bee → bumblebee → wasp → hornet).  For every concept we provide paraphrases
that avoid the concept word (hard paraphrases), so that a next-token readout
cannot exploit lexical overlap.

The behavioural test is purely distributional: concepts on the same chain
should have smaller JS(next-token) between neighbours than between distant
members, while neighbours are still distinct (JS > 0).
"""

from __future__ import annotations

import numpy as np

from .metrics import jaccard_words

CHAINS = {
    "insects": {
        "concepts": ["bee", "bumblebee", "wasp", "hornet"],
        "texts": {
            "bee": [
                "The striped insect gathers nectar and makes honey.",
                "A small winged creature collects sweet fluid from blossoms.",
                "This buzzing pollinator lives in a colony and stores food in wax cells.",
            ],
            "bumblebee": [
                "A big fuzzy black-and-yellow flyer drones over flowers.",
                "This round hairy insect buzzes loudly and carries pollen between blossoms.",
                "A plump warm-bodied pollinator with a thick coat visits spring gardens.",
            ],
            "wasp": [
                "A slim stinging insect hunts other bugs and builds paper nests.",
                "This aggressive flyer has a painful sting and preys on smaller creatures.",
                "A yellow-jacketed hunter that feeds its young on captured insects.",
            ],
            "hornet": [
                "A large aggressive stinger builds a huge paper nest and attacks honey-makers.",
                "This massive predatory insect raids colonies and has a very painful sting.",
                "A big fearsome flyer with a loud buzz that hunts in swarms.",
            ],
        },
    },
    "water_flow": {
        "concepts": ["river", "stream", "creek", "brook"],
        "texts": {
            "river": [
                "A large body of water flows toward the ocean.",
                "A wide natural channel carries freshwater downhill to the sea.",
                "Flowing liquid runs across the land inside a long carved bed.",
            ],
            "stream": [
                "A small narrow current trickles through the countryside.",
                "A little watercourse winds between the hills toward a larger body.",
                "Shallow moving water threads through fields and woods.",
            ],
            "creek": [
                "A tiny watercourse wanders through a meadow.",
                "A narrow shallow flow slips over stones in the woods.",
                "A very small channel of moving water crosses a path.",
            ],
            "brook": [
                "A little rill babbles over pebbles in a shady glen.",
                "A thin thread of water meanders beneath the trees.",
                "A gentle trickle winds through grass and roots.",
            ],
        },
    },
    "frozen_precip": {
        "concepts": ["snow", "sleet", "hail"],
        "texts": {
            "snow": [
                "Frozen flakes fall from clouds in cold weather.",
                "Ice crystals drift down from the sky during winter.",
                "White frozen bits descend when the temperature drops below zero.",
            ],
            "sleet": [
                "Half-melted icy drops splash down on a freezing day.",
                "A mixture of rain and ice pellets falls during a winter storm.",
                "Slushy frozen rain taps on the windows in cold weather.",
            ],
            "hail": [
                "Hard icy pellets pelt the ground during a storm.",
                "Frozen balls of ice crash from thunderclouds in summer.",
                "Chunks of solid ice fall violently from towering clouds.",
            ],
        },
    },
}


def chain_names() -> list[str]:
    return list(CHAINS)


def chain(name: str) -> dict:
    spec = CHAINS[name]
    return {"name": name, "concepts": list(spec["concepts"]),
            "texts": {c: list(ts) for c, ts in spec["texts"].items()}}


def lexical_matrix(name: str) -> np.ndarray:
    """Mean word-Jaccard between paraphrases of every concept pair."""
    spec = CHAINS[name]
    concepts = spec["concepts"]
    n = len(concepts)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            vals = [jaccard_words(a, b) for a in spec["texts"][concepts[i]]
                    for b in spec["texts"][concepts[j]]]
            M[i, j] = M[j, i] = float(np.mean(vals))
    return M


def relatedness_profile(J: np.ndarray) -> dict:
    """Adjacent vs skip-1 vs endpoint JS along a chain."""
    J = np.asarray(J, dtype=np.float64)
    K = J.shape[0]
    adj = [J[i, i + 1] for i in range(K - 1)]
    skip = [J[i, i + 2] for i in range(K - 2)]
    endpoints = J[0, K - 1]
    return {
        "adjacent_mean": float(np.mean(adj)),
        "skip1_mean": float(np.mean(skip)) if skip else float("nan"),
        "endpoints": float(endpoints),
        "adjacent": [float(x) for x in adj],
        "skip1": [float(x) for x in skip],
        "gradient_ok": bool(
            np.mean(adj) > 0 and
            (not skip or np.mean(adj) < np.mean(skip)) and
            np.mean(adj) < endpoints),
    }
