"""Concept centroids from token distributions (experiment 018, step 1).

Each paraphrase is mapped to a next-token distribution ``p`` under a fixed
frame.  We embed ``p`` as ``sqrt(p)`` (Hellinger), so inner products are
``G_ij = <sqrt(p_i), sqrt(p_j)> = 1 - H^2/2``.  All geometry is computed from
the Gram matrix ``G`` (n x n), never from the V-dimensional vectors.

Leave-one-slot-out: one paraphrase per concept is held out, the subspace and
the centroids are built on the rest, and the held-out paraphrase is assigned to
the nearest centroid.  Raw baseline: same PCA coordinates, no Fisher.  TF-IDF
baseline: word vectors without the model.

Protocol: ``../experiment-018-2026-11-28-concept-centroids.md``.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from wordfreq import zipf_frequency

from .controls import partial_spearman, spearman, upper_tri
from .separation import cosine_rdm, fisher_directions, label_rdm, project

FRAME = "You can describe {x} as"
FRAME_PASSIVE = "{x} can be described as"

CONCEPTS = [
    ("dog", ["a dog", "a domestic canine", "a barking quadruped kept as a pet",
             "a loyal household animal", "man's best friend"]),
    ("bear", ["a bear", "a large furry omnivore", "a heavy clawed forest mammal",
              "a hibernating woodland creature", "a big grizzly or black animal"]),
    ("owl", ["an owl", "a nocturnal raptor", "a hooting night bird",
             "a round-eyed feathered hunter", "a wise-looking creature of the dark"]),
    ("eagle", ["an eagle", "a large soaring raptor", "a broad-winged bird of prey",
               "a sharp-eyed winged hunter", "a bald national emblem of flight"]),
    ("apple", ["an apple", "a round red pome", "a crisp orchard fruit",
               "a common tree-grown snack", "a hand-sized sweet crop"]),
    ("banana", ["a banana", "a long yellow tropical produce", "a curved peelable snack",
                "a soft starchy finger fruit", "a potassium-rich bunch item"]),
    ("carrot", ["a carrot", "an orange taproot", "a crunchy root vegetable",
                "a rabbit's favourite crop", "a tapered edible garden plant"]),
    ("onion", ["an onion", "a layered bulb", "a pungent round vegetable",
               "a tear-inducing kitchen staple", "a papery-skinned allium"]),
    ("bed", ["a bed", "a piece of furniture for sleeping", "a mattress on a frame",
             "a bedroom resting place", "somewhere to lie down at night"]),
    ("couch", ["a couch", "a sofa", "a long upholstered seat",
               "a living-room bench for several people", "a padded seating furniture"]),
    ("car", ["a car", "an automobile", "a four-wheeled motor vehicle",
             "a passenger road machine", "a family transport with an engine"]),
    ("boat", ["a boat", "a small vessel", "a watercraft",
              "a floating craft for people", "a hull that travels on water"]),
    ("screwdriver", ["a screwdriver", "a slotted turning implement", "a hand tool with a shaped tip",
                     "a driving device with a handle", "a fastener-twisting gadget"]),
    ("broom", ["a broom", "a sweeping implement", "a brush on a long handle",
               "a floor-cleaning tool", "a bundle of bristles for tidying"]),
    ("spear", ["a spear", "a long pointed weapon", "a throwing lance",
               "a shaft with a sharp tip", "a primitive thrusting polearm"]),
    ("whip", ["a whip", "a lash on a handle", "a leather striking cord",
              "a long flexible thong", "a braided instrument of discipline"]),
    ("jeans", ["jeans", "a pair of jeans", "denim trousers",
               "blue five-pocket pants", "casual cotton twill bottoms"]),
    ("sweater", ["a sweater", "a knitted pullover", "a warm jumper",
                 "a woollen top", "a long-sleeved winter garment"]),
    ("guitar", ["a guitar", "a six-stringed instrument", "a strummed wooden chordophone",
                "a plucked music-maker with a neck", "a fretted box for accompaniment"]),
    ("violin", ["a violin", "a fiddle", "a bowed treble chordophone",
                "a small four-stringed instrument", "a high-pitched orchestral strings member"]),
]


def build_items(frame: str = FRAME) -> list[dict]:
    """Return the 100 items with concept/slot labels and the framed text."""
    items = []
    for ci, (name, paras) in enumerate(CONCEPTS):
        for slot, phr in enumerate(paras):
            items.append({
                "concept": ci,
                "concept_name": name,
                "slot": slot,
                "paraphrase": phr,
                "text": frame.format(x=phr),
            })
    return items


def hellinger_gram(P) -> np.ndarray:
    """``G_ij = <sqrt(p_i), sqrt(p_j)> = 1 - H^2/2``."""
    X = np.sqrt(np.clip(np.asarray(P, dtype=np.float64), 0.0, None))
    return X @ X.T


def centered_gram(G, train_idx) -> np.ndarray:
    """Gram of all vectors centered by the mean of ``train_idx``."""
    G = np.asarray(G, dtype=np.float64)
    train_idx = np.asarray(train_idx, dtype=int)
    m = G[:, train_idx].mean(axis=1)
    mm = G[np.ix_(train_idx, train_idx)].mean()
    return G - m[:, None] - m[None, :] + mm


def train_pca(Gc, train_idx, test_idx, tol: float = 1e-10):
    """Coordinates of train and test in the PCA basis of the train block."""
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    A = Gc[np.ix_(train_idx, train_idx)]
    w, U = np.linalg.eigh(A)
    order = np.argsort(w)[::-1]
    w, U = w[order], U[:, order]
    cut = max(tol * (w.max() if w.size else 0.0), 1e-12)
    keep = w > cut
    w, U = w[keep], U[:, keep]
    Ztr = U * np.sqrt(w)
    cross = Gc[np.ix_(test_idx, train_idx)]
    Zte = cross @ U / np.sqrt(w)
    return Ztr, Zte


def _nearest_centroid(Ztr, ytr, Zte) -> np.ndarray:
    labels = np.unique(ytr)
    cents = np.stack([Ztr[ytr == c].mean(axis=0) for c in labels])
    d = ((Zte[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
    return labels[np.argmin(d, axis=1)]


def cv_fisher(G, y, slots, k: int) -> dict:
    """Leave-one-slot-out accuracy with the Fisher (meaning) projection."""
    y = np.asarray(y)
    slots = np.asarray(slots)
    correct = total = 0
    folds = []
    for s in np.unique(slots):
        test = np.where(slots == s)[0]
        train = np.where(slots != s)[0]
        Gc = centered_gram(G, train)
        Ztr, Zte = train_pca(Gc, train, test)
        dirs = fisher_directions(Ztr, y[train], n_meaning=k, n_form=k)
        Ptr = project(Ztr, dirs["W_meaning"], dirs["mean"])
        Pte = project(Zte, dirs["W_meaning"], dirs["mean"])
        pred = _nearest_centroid(Ptr, y[train], Pte)
        hit = pred == y[test]
        correct += int(hit.sum())
        total += int(hit.size)
        folds.append({"slot": int(s), "accuracy": float(hit.mean()), "n_test": int(hit.size)})
    return {"accuracy": correct / total if total else float("nan"), "folds": folds}


def cv_raw(G, y, slots, k: int = 10) -> dict:
    """Same leave-one-slot-out, but top-``k`` PCA coordinates without Fisher."""
    y = np.asarray(y)
    slots = np.asarray(slots)
    correct = total = 0
    for s in np.unique(slots):
        test = np.where(slots == s)[0]
        train = np.where(slots != s)[0]
        Gc = centered_gram(G, train)
        Ztr, Zte = train_pca(Gc, train, test)
        kk = min(k, Ztr.shape[1])
        pred = _nearest_centroid(Ztr[:, :kk], y[train], Zte[:, :kk])
        hit = pred == y[test]
        correct += int(hit.sum())
        total += int(hit.size)
    return {"accuracy": correct / total if total else float("nan")}


def hellinger_centroid_accuracy(P, y, slots) -> float:
    """Nearest centroid on the raw sqrt(p) vectors (no PCA, no Fisher)."""
    X = np.sqrt(np.clip(np.asarray(P, dtype=np.float64), 0.0, None))
    y = np.asarray(y)
    slots = np.asarray(slots)
    correct = total = 0
    for s in np.unique(slots):
        test = np.where(slots == s)[0]
        train = np.where(slots != s)[0]
        labels = np.unique(y[train])
        C = np.stack([X[train][y[train] == c].mean(axis=0) for c in labels])
        C = normalize(C)
        sim = X[test] @ C.T
        d = np.sqrt(np.clip(1.0 - sim, 0.0, None))
        pred = labels[np.argmin(d, axis=1)]
        hit = pred == y[test]
        correct += int(hit.sum())
        total += int(hit.size)
    return correct / total if total else float("nan")


def tfidf_centroid_accuracy(texts, y, slots) -> float:
    """Nearest centroid on TF-IDF word vectors (no model)."""
    texts = list(texts)
    y = np.asarray(y)
    slots = np.asarray(slots)
    correct = total = 0
    for s in np.unique(slots):
        test = np.where(slots == s)[0]
        train = np.where(slots != s)[0]
        vec = TfidfVectorizer()
        Xtr = vec.fit_transform([texts[i] for i in train])
        Xte = vec.transform([texts[i] for i in test])
        labels = np.unique(y[train])
        C = np.stack([np.asarray(Xtr[y[train] == c].mean(axis=0)).ravel() for c in labels])
        C = normalize(C)
        sim = cosine_similarity(Xte, C)
        pred = labels[np.argmax(sim, axis=1)]
        hit = pred == y[test]
        correct += int(hit.sum())
        total += int(hit.size)
    return correct / total if total else float("nan")


def permutation_null(G, y, slots, k: int, n_perm: int, seed: int = 0) -> dict:
    """Permutation test of the Fisher CV accuracy under shuffled concept labels."""
    y = np.asarray(y)
    obs = cv_fisher(G, y, slots, k)["accuracy"]
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        null[i] = cv_fisher(G, rng.permutation(y), slots, k)["accuracy"]
    p = (float(np.sum(null >= obs)) + 1.0) / (n_perm + 1.0)
    return {"observed": obs, "mean": float(np.mean(null)),
            "p": p, "n_perm": int(n_perm)}


# ---------------------------------------------------------------------------
# Lexical control
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z']+")


def content_words(text) -> set[str]:
    """Lower-case content words: alphabetic, longer than two, not stopwords."""
    return {w for w in _WORD_RE.findall(str(text).lower())
            if len(w) > 2 and w not in ENGLISH_STOP_WORDS}


def anchor_diagnostic(items) -> list[dict]:
    """For each concept, content words shared by all paraphrases and unique to it."""
    by, names = {}, {}
    for it in items:
        by.setdefault(it["concept"], []).append(it["paraphrase"])
        names[it["concept"]] = it["concept_name"]
    all_words = {c: set().union(*(content_words(p) for p in ps)) for c, ps in by.items()}
    out = []
    for c, ps in by.items():
        common = set.intersection(*(content_words(p) for p in ps))
        others = set().union(*(all_words[d] for d in by if d != c))
        out.append({
            "concept": names[c],
            "shared_all": sorted(common),
            "anchors": sorted(common - others),
        })
    return out


def tfidf_fisher_cv(texts, y, slots, k: int) -> float:
    """Same leave-one-slot-out Fisher pipeline, but on TF-IDF word vectors."""
    texts = list(texts)
    y = np.asarray(y)
    slots = np.asarray(slots)
    correct = total = 0
    for s in np.unique(slots):
        test = np.where(slots == s)[0]
        train = np.where(slots != s)[0]
        vec = TfidfVectorizer()
        Xtr = vec.fit_transform([texts[i] for i in train]).toarray()
        Xte = vec.transform([texts[i] for i in test]).toarray()
        dirs = fisher_directions(Xtr, y[train], n_meaning=k, n_form=k)
        Ptr = project(Xtr, dirs["W_meaning"], dirs["mean"])
        Pte = project(Xte, dirs["W_meaning"], dirs["mean"])
        pred = _nearest_centroid(Ptr, y[train], Pte)
        hit = pred == y[test]
        correct += int(hit.sum())
        total += int(hit.size)
    return correct / total if total else float("nan")


def cv_fisher_residualized(G, texts, y, slots, k: int, n_lex: int = 10) -> float:
    """Fisher CV after removing the TF-IDF-predictable part of the coordinates.

    Per fold: PCA coordinates are regressed on the top ``n_lex`` lexical
    components (fit on train only), and the residuals go through the Fisher
    projection.  Conservative: lexical components may also carry meaning.
    """
    texts = list(texts)
    y = np.asarray(y)
    slots = np.asarray(slots)
    correct = total = 0
    for s in np.unique(slots):
        test = np.where(slots == s)[0]
        train = np.where(slots != s)[0]
        Gc = centered_gram(G, train)
        Ztr, Zte = train_pca(Gc, train, test)
        vec = TfidfVectorizer()
        Ltr = vec.fit_transform([texts[i] for i in train])
        Lte = vec.transform([texts[i] for i in test])
        ncomp = int(min(n_lex, max(1, Ltr.shape[1] - 1), max(1, Ltr.shape[0] - 1)))
        svd = TruncatedSVD(n_components=ncomp, random_state=0)
        Str = svd.fit_transform(Ltr)
        Ste = svd.transform(Lte)
        Atr = np.column_stack([np.ones(len(train)), Str])
        beta, *_ = np.linalg.lstsq(Atr, Ztr, rcond=None)
        Ztr_r = Ztr - Atr @ beta
        Ate = np.column_stack([np.ones(len(test)), Ste])
        Zte_r = Zte - Ate @ beta
        dirs = fisher_directions(Ztr_r, y[train], n_meaning=k, n_form=k)
        Ptr = project(Ztr_r, dirs["W_meaning"], dirs["mean"])
        Pte = project(Zte_r, dirs["W_meaning"], dirs["mean"])
        pred = _nearest_centroid(Ptr, y[train], Pte)
        hit = pred == y[test]
        correct += int(hit.sum())
        total += int(hit.size)
    return correct / total if total else float("nan")


def permutation_null_residualized(G, texts, y, slots, k: int, n_perm: int,
                                  n_lex: int = 10, seed: int = 0) -> dict:
    """Permutation test of the residualized Fisher accuracy."""
    y = np.asarray(y)
    obs = cv_fisher_residualized(G, texts, y, slots, k, n_lex)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        null[i] = cv_fisher_residualized(G, texts, rng.permutation(y), slots, k, n_lex)
    p = (float(np.sum(null >= obs)) + 1.0) / (n_perm + 1.0)
    return {"observed": obs, "mean": float(np.mean(null)),
            "p": p, "n_perm": int(n_perm), "n_lex": int(n_lex)}


def frequency_length_covariates(texts, lengths=None) -> np.ndarray:
    """Two covariates per item: token length and mean zipf of content words."""
    texts = list(texts)
    if lengths is None:
        lengths = [len(re.findall(r"\S+", str(t))) for t in texts]
    freqs = []
    for t in texts:
        ws = content_words(t)
        freqs.append(float(np.mean([zipf_frequency(w, "en") for w in ws])) if ws else 0.0)
    return np.column_stack([np.asarray(lengths, dtype=np.float64),
                            np.asarray(freqs, dtype=np.float64)])


def cv_fisher_covariates(G, y, slots, covariates, k: int) -> float:
    """Fisher CV after removing the part of the coordinates predicted by covariates."""
    y = np.asarray(y)
    slots = np.asarray(slots)
    covariates = np.asarray(covariates, dtype=np.float64)
    correct = total = 0
    for s in np.unique(slots):
        test = np.where(slots == s)[0]
        train = np.where(slots != s)[0]
        Gc = centered_gram(G, train)
        Ztr, Zte = train_pca(Gc, train, test)
        Atr = np.column_stack([np.ones(len(train)), covariates[train]])
        beta, *_ = np.linalg.lstsq(Atr, Ztr, rcond=None)
        Ztr_r = Ztr - Atr @ beta
        Ate = np.column_stack([np.ones(len(test)), covariates[test]])
        Zte_r = Zte - Ate @ beta
        dirs = fisher_directions(Ztr_r, y[train], n_meaning=k, n_form=k)
        Ptr = project(Ztr_r, dirs["W_meaning"], dirs["mean"])
        Pte = project(Zte_r, dirs["W_meaning"], dirs["mean"])
        pred = _nearest_centroid(Ptr, y[train], Pte)
        hit = pred == y[test]
        correct += int(hit.sum())
        total += int(hit.size)
    return correct / total if total else float("nan")


def _transfer_one(G, y, train_idx, test_idx, k: int):
    Gc = centered_gram(G, train_idx)
    Ztr, Zte = train_pca(Gc, train_idx, test_idx)
    dirs = fisher_directions(Ztr, y[train_idx], n_meaning=k, n_form=k)
    Ptr = project(Ztr, dirs["W_meaning"], dirs["mean"])
    Pte = project(Zte, dirs["W_meaning"], dirs["mean"])
    labels = np.unique(y[train_idx])
    Ctr = np.stack([Ptr[y[train_idx] == c].mean(axis=0) for c in labels])
    pred = labels[np.argmin(((Pte[:, None, :] - Ctr[None, :, :]) ** 2).sum(axis=2), axis=1)]
    acc = float(np.mean(pred == y[test_idx]))
    yte = y[test_idx]
    Cte = np.stack([Pte[yte == c].mean(axis=0) for c in labels])
    rho = spearman(upper_tri(cosine_rdm(Ctr)), upper_tri(cosine_rdm(Cte)))
    return acc, rho


def cross_frame_transfer(G, y, frames, k: int) -> dict:
    """Fit on one frame, test on another (pooled Gram, train-only PCA)."""
    y = np.asarray(y)
    frames = np.asarray(frames)
    a = np.where(frames == 0)[0]
    b = np.where(frames == 1)[0]
    acc_ab, rho_ab = _transfer_one(G, y, a, b, k)
    acc_ba, rho_ba = _transfer_one(G, y, b, a, k)
    return {
        "accuracy_A_to_B": acc_ab,
        "accuracy_B_to_A": acc_ba,
        "rdm_corr_A_to_B": rho_ab,
        "rdm_corr_B_to_A": rho_ba,
        "chance": float(1.0 / np.unique(y).size),
    }


def _jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def lexical_overlap_stats(items) -> dict:
    """Mean within-concept and between-concept Jaccard of content words."""
    by = {}
    for it in items:
        by.setdefault(it["concept"], []).append(content_words(it["paraphrase"]))
    within = []
    for sets in by.values():
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                within.append(_jaccard(sets[i], sets[j]))
    concepts = list(by)
    between = []
    for a in range(len(concepts)):
        for b in range(a + 1, len(concepts)):
            for sa in by[concepts[a]]:
                for sb in by[concepts[b]]:
                    between.append(_jaccard(sa, sb))
    return {
        "within_mean": float(np.mean(within)) if within else float("nan"),
        "between_mean": float(np.mean(between)) if between else float("nan"),
        "gap": (float(np.mean(within) - np.mean(between))
                if within and between else float("nan")),
    }


def rdm_partial(P, texts, y) -> dict:
    """How much of the model RDM survives after the lexical RDM is removed."""
    X = np.sqrt(np.clip(np.asarray(P, dtype=np.float64), 0.0, None))
    G = X @ X.T
    model = np.sqrt(np.clip(1.0 - G, 0.0, None))
    np.fill_diagonal(model, 0.0)
    vec = TfidfVectorizer()
    T = vec.fit_transform(list(texts))
    lexical = 1.0 - cosine_similarity(T)
    np.fill_diagonal(lexical, 0.0)
    concept = label_rdm(np.asarray(y))
    return {
        "rho_model_concept": spearman(upper_tri(model), upper_tri(concept)),
        "rho_model_lexical": spearman(upper_tri(model), upper_tri(lexical)),
        "partial_model_concept_given_lexical":
            partial_spearman(model, concept, lexical),
    }
