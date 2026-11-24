"""Model loading and per-layer activation extraction (CPU-friendly)."""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


def load(name: str, untrained: bool = False, seed: int = 0):
    """Load a (causal or masked) model and tokenizer.

    untrained=True builds the same architecture with random weights (control).
    """
    torch.use_deterministic_algorithms(True, warn_only=True)
    tokenizer = AutoTokenizer.from_pretrained(name)
    if untrained:
        config = AutoConfig.from_pretrained(name)
        torch.manual_seed(seed)
        model = AutoModel.from_config(config, attn_implementation="eager")
    else:
        model = AutoModel.from_pretrained(name, attn_implementation="eager")
    model.eval()
    return model, tokenizer


def extract(text: str, model, tokenizer, layer: int):
    """Return (V, A) at `layer`.

    V : (n, d) hidden states of position vectors.
    A : (n, n) attention averaged over heads after softmax, or None if the
        architecture does not return attentions (e.g. some masked models).
    """
    enc = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, output_attentions=True)
    V = out.hidden_states[layer][0].double().numpy()
    A = None
    att = getattr(out, "attentions", None)
    # hidden_states has L+1 entries (embeddings + each layer); attentions has L.
    # hidden_states[layer] is the output of attention layer (layer-1).
    if att is not None and layer >= 1 and layer - 1 < len(att) and att[layer - 1] is not None:
        A = att[layer - 1].mean(dim=1)[0].double().numpy()
    return V, A


def last_layer(model) -> int:
    return int(model.config.num_hidden_layers)


def token_count(text: str, tokenizer) -> int:
    return int(len(tokenizer(text).input_ids))


def next_token_probs(model, tokenizer, text: str) -> np.ndarray:
    """Full next-token probability vector at the last position of `text`.

    This is the functional readout the protocol scores (experiment 001): the
    distribution the loss is defined on, not the residual-stream geometry.
    """
    enc = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    return torch.softmax(out.logits[0, -1], dim=-1).double().numpy()
