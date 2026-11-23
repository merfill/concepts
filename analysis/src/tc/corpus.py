"""Small handcrafted corpus for H1 validation.

PARAPHRASE_GROUPS: concept -> list of paraphrases describing the same concept.
Used for test-retest reliability and within/between separation. English is used
because the small CPU models (distilgpt2, bert-tiny) are English.

NUMBER_PAIRS: minimal pairs differing only in grammatical number, used as a
weak model-level positive control (a known structural distinction).
"""

from __future__ import annotations

PARAPHRASE_GROUPS = {
    "bee": [
        "A bee is a flying insect that collects nectar from flowers.",
        "Bees live in colonies and make honey in hives.",
        "A small striped insect that gathers nectar and produces honey.",
        "The bee pollinates flowers while collecting nectar for the hive.",
    ],
    "river": [
        "A river is a large natural stream of water flowing to the sea.",
        "Rivers carry freshwater across the land toward lakes or oceans.",
        "A long flowing body of water that runs across the landscape.",
        "Water flows continuously along a river channel toward lower ground.",
    ],
    "volcano": [
        "A volcano is a mountain that erupts lava and ash.",
        "Volcanoes open in the crust of the earth and release molten rock.",
        "A mountain formed by eruptions of hot lava from below.",
        "Molten rock rises through a volcano and flows out as lava.",
    ],
    "democracy": [
        "Democracy is a system where people elect their leaders.",
        "In a democracy citizens vote to choose the government.",
        "A form of government in which the people hold power through elections.",
        "Rule by the people, exercised mainly through voting.",
    ],
    "guitar": [
        "A guitar is a stringed instrument played with fingers or a pick.",
        "The guitar has strings stretched over a body and a neck.",
        "A musical instrument with strings that you pluck to make sound.",
        "Six strings on a wooden body produce notes when strummed.",
    ],
    "snow": [
        "Snow is frozen water that falls from clouds as white flakes.",
        "Snow falls when water vapor freezes into ice crystals in clouds.",
        "White frozen flakes that fall from the sky in cold weather.",
        "Crystals of ice that form in clouds and fall as snow.",
    ],
    "shark": [
        "A shark is a large fish with sharp teeth that lives in the ocean.",
        "Sharks are predatory fish that swim in the sea.",
        "A cartilaginous ocean fish known for its sharp teeth and fins.",
        "Marine predators with rows of teeth that hunt in the ocean.",
    ],
    "library": [
        "A library is a place where books are kept for reading and borrowing.",
        "Libraries store many books that people can read or borrow.",
        "A building full of books available to the public.",
        "A collection of books organized for reading and lending.",
    ],
}

NUMBER_PAIRS = [
    ("The cat sleeps on the warm mat.", "The cats sleep on the warm mat."),
    ("The dog runs in the green park.", "The dogs run in the green park."),
    ("A bird sings in the tall tree.", "Birds sing in the tall tree."),
    ("The child plays with a red ball.", "The children play with a red ball."),
    ("The car stops at the red light.", "The cars stop at the red light."),
]


def flat_paraphrases():
    """Return (texts, group_ids, group_names) for the paraphrase corpus."""
    texts, group_ids, group_names = [], [], []
    for gid, (name, items) in enumerate(PARAPHRASE_GROUPS.items()):
        group_names.append(name)
        for t in items:
            texts.append(t)
            group_ids.append(gid)
    return texts, group_ids, group_names


# "Hard" paraphrases: same concept, deliberately disjoint content words.
# Controlled for lexical overlap so that any surviving signal is semantic.
HARD_GROUPS = {
    "bee": [
        "The striped insect gathers nectar and makes honey.",
        "A small winged creature collects sweet fluid from blossoms and produces golden syrup.",
        "This buzzing pollinator lives in a colony and stores food in wax cells.",
    ],
    "river": [
        "A large stream of water flows toward the ocean.",
        "A wide natural channel carries freshwater downhill to the sea.",
        "Flowing liquid runs across the land inside a long carved bed.",
    ],
    "snow": [
        "Frozen flakes fall from clouds in cold weather.",
        "Ice crystals drift down from the sky during winter.",
        "White frozen bits descend when the temperature drops below zero.",
    ],
    "volcano": [
        "Molten rock erupts from a mountain and flows as lava.",
        "A hill of the crust spews hot melted stone outward.",
        "Burning liquid stone bursts from a peak through a vent.",
    ],
    "democracy": [
        "People choose their leaders by voting in elections.",
        "Citizens select their rulers through popular ballots.",
        "A nation governs itself when adults cast votes for representatives.",
    ],
    "guitar": [
        "A stringed instrument that you pluck to make music.",
        "A wooden device with cords produces notes when strummed.",
        "Musicians play this six-corded object to create melodies.",
    ],
    "shark": [
        "A predatory ocean fish with sharp teeth and fins.",
        "A cartilaginous marine hunter with jagged jaws.",
        "This fearsome sea predator tears prey with rows of pointed blades.",
    ],
    "library": [
        "A place where books are stored for reading and borrowing.",
        "A building that holds texts available to the public.",
        "Shelves of volumes organized for lending to readers.",
    ],
    "clock": [
        "A device that shows the time with hands on a dial.",
        "An instrument displays hours and minutes via moving pointers.",
        "This machine measures passing moments and marks them on a face.",
    ],
    "ocean": [
        "A vast body of salt water covering much of the earth.",
        "An enormous salty expanse that surrounds the continents.",
        "Deep blue marine waters stretch across the globe.",
    ],
}


def flat_hard():
    """Return (texts, group_ids, group_names) for the hard paraphrase corpus."""
    texts, group_ids, group_names = [], [], []
    for gid, (name, items) in enumerate(HARD_GROUPS.items()):
        group_names.append(name)
        for t in items:
            texts.append(t)
            group_ids.append(gid)
    return texts, group_ids, group_names


# Related concept pairs: each concept gets disjoint-word paraphrases.
# Used to test fine-grained discrimination (same concept vs its near neighbour).
RELATED_CONCEPTS = {
    "bee": [
        "The striped insect gathers nectar and makes honey.",
        "A small winged creature collects sweet fluid and produces golden syrup.",
        "This buzzing pollinator lives in a colony and stores food in wax cells.",
    ],
    "wasp": [
        "A slim stinging insect hunts other bugs and builds paper nests.",
        "This aggressive flyer has a painful sting and preys on smaller creatures.",
        "A yellow-jacketed hunter that feeds its young on captured insects.",
    ],
    "river": [
        "A large stream of water flows toward the ocean.",
        "A wide natural channel carries freshwater downhill to the sea.",
        "Flowing liquid runs across the land inside a long carved bed.",
    ],
    "stream": [
        "A small narrow current trickles through the countryside.",
        "A little brook winds between the hills toward a larger body.",
        "Shallow moving water threads through fields and woods.",
    ],
    "snow": [
        "Frozen flakes fall from clouds in cold weather.",
        "Ice crystals drift down from the sky during winter.",
        "White frozen bits descend when the temperature drops below zero.",
    ],
    "hail": [
        "Hard icy pellets pelt the ground during a storm.",
        "Frozen balls of ice crash from thunderclouds in summer.",
        "Chunks of solid ice fall violently from towering clouds.",
    ],
    "volcano": [
        "Molten rock erupts from a mountain and flows as lava.",
        "A hill of the crust spews hot melted stone outward.",
        "Burning liquid stone bursts from a peak through a vent.",
    ],
    "geyser": [
        "Hot water shoots upward from the ground in a tall column.",
        "A spring violently ejects boiling liquid into the air.",
        "Steam and scalding fluid burst periodically from a hole.",
    ],
    "guitar": [
        "A stringed instrument that you pluck to make music.",
        "A wooden device with cords produces notes when strummed.",
        "Musicians play this six-corded object to create melodies.",
    ],
    "violin": [
        "A bowed wooden instrument held under the chin.",
        "This small stringed tool sings when rubbed with a horsehair stick.",
        "A classical instrument with four strings played by drawing a bow.",
    ],
    "shark": [
        "A predatory ocean fish with sharp teeth and fins.",
        "A cartilaginous marine hunter with jagged jaws.",
        "This fearsome sea predator tears prey with rows of pointed blades.",
    ],
    "dolphin": [
        "A clever marine mammal that leaps and clicks underwater.",
        "This friendly oceanic swimmer breathes air and travels in pods.",
        "A playful warm-blooded sea creature known for intelligence.",
    ],
}

RELATED_PAIRS = [
    ("bee", "wasp"),
    ("river", "stream"),
    ("snow", "hail"),
    ("volcano", "geyser"),
    ("guitar", "violin"),
    ("shark", "dolphin"),
]


def flat_related():
    """Return (texts, concept_ids, concept_names, name_to_cid)."""
    texts, cids, names = [], [], []
    for cid, (name, items) in enumerate(RELATED_CONCEPTS.items()):
        names.append(name)
        for t in items:
            texts.append(t)
            cids.append(cid)
    name_to_cid = {n: i for i, n in enumerate(RELATED_CONCEPTS)}
    return texts, cids, names, name_to_cid

