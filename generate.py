import os
import sys
import json
import ast
import math
import random
import argparse
import logging
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from music21 import stream, note, tempo as m21_tempo

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model definition — must match training exactly
# ─────────────────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    """
    Additive attention over LSTM output sequence.
    Computes a weighted sum of all positions so the model can
    'look back' at any past token when predicting the next one.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out : (batch, seq_len, hidden_size)
        scores  = self.attn(lstm_out).squeeze(-1)              # (batch, seq_len)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (batch, seq_len, 1)
        context = (lstm_out * weights).sum(dim=1)              # (batch, hidden_size)
        return context


class MusicLSTM(nn.Module):
    """
    Stacked LSTM + Attention for symbolic music generation.

    Input  : sequence of token indices  (batch, seq_len)
    Output : logits over vocabulary     (batch, vocab_size)
    """

    def __init__(self, vocab_size: int, embed_dim: int,
                 hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = Attention(hidden_size)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded    = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)
        context     = self.attention(lstm_out)
        return self.fc(self.dropout(context))


# ─────────────────────────────────────────────────────────────────────────────
# Duration & velocity maps (must match preprocessing)
# ─────────────────────────────────────────────────────────────────────────────

DURATION_MAP = {
    "thirty_second":  0.125,
    "sixteenth":      0.25,
    "eighth":         0.5,
    "dotted_eighth":  0.75,
    "quarter":        1.0,
    "dotted_quarter": 1.5,
    "half":           2.0,
    "dotted_half":    3.0,
    "whole":          4.0,
    "long":           4.0,   # clip to whole note
}

VELOCITY_MAP = {
    "ppp":  8,
    "pp":   24,
    "p":    40,
    "mp":   56,
    "mf":   72,
    "f":    88,
    "ff":   104,
    "fff":  120,
}

# ─────────────────────────────────────────────────────────────────────────────
# Soundfont auto-download
# ─────────────────────────────────────────────────────────────────────────────

SOUNDFONT_URL  = (
    "https://github.com/JustEnoughLinuxOS/generaluser-gs/raw/main/"
    "GeneralUser%20GS%20v1.471.sf2"
)
SOUNDFONT_PATH     = "artifacts/soundfont.sf2"
HF_REPO_ID         = "KalineZephyr/music-lstm-midi-codealpha"
DEFAULT_MODEL_PATH = "artifacts/best_model.pt"


def ensure_model(model_path: str) -> str:
    """
    Download best_model.pt from HuggingFace Hub if not present locally.
    Requires: pip install huggingface_hub
    """
    if os.path.exists(model_path):
        logger.info("Model found: %s", model_path)
        return model_path

    logger.info("Model not found locally — downloading from HuggingFace Hub...")
    logger.info("  Repo : %s", HF_REPO_ID)
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="best_model.pt",
            local_dir=os.path.dirname(model_path) or "artifacts",
        )
        logger.info("Model downloaded: %s", downloaded)
        return downloaded
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)
    except Exception as e:
        logger.error("Failed to download model: %s", e)
        logger.error("Download manually from: https://huggingface.co/%s", HF_REPO_ID)
        sys.exit(1)


def ensure_soundfont() -> str:
    """Download GeneralUser GS soundfont if not already present."""
    if os.path.exists(SOUNDFONT_PATH):
        logger.info("Soundfont found: %s", SOUNDFONT_PATH)
        return SOUNDFONT_PATH

    logger.info("Soundfont not found — downloading GeneralUser GS (~30MB)...")
    os.makedirs(os.path.dirname(SOUNDFONT_PATH), exist_ok=True)

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            pct = count * block_size * 100 / total_size
            print(f"\r  Downloading... {pct:.1f}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(SOUNDFONT_URL, SOUNDFONT_PATH, reporthook)
        print()  # newline after progress
        logger.info("Soundfont downloaded: %s", SOUNDFONT_PATH)
        return SOUNDFONT_PATH
    except Exception as e:
        logger.error("Failed to download soundfont: %s", e)
        logger.error("Install manually: place any .sf2 file at %s", SOUNDFONT_PATH)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────────────────────────────────────────

def load_vocab(vocab_path: str) -> tuple[dict, dict, int, int]:
    """Load vocabulary from vocab.json."""
    with open(vocab_path) as f:
        vocab = json.load(f)

    # idx2token : int → (pitch, duration_label, velocity_label)
    idx2token = {int(k): tuple(v) for k, v in vocab["idx2token"].items()}

    # token2idx : (pitch, duration_label, velocity_label) → int
    # Keys were saved as string repr of tuples — use ast.literal_eval
    token2idx = {ast.literal_eval(k): v for k, v in vocab["token2idx"].items()}

    vocab_size   = vocab["vocab_size"]
    sequence_len = vocab["sequence_len"]

    logger.info("Vocab loaded: %d tokens, sequence_len=%d", vocab_size, sequence_len)
    return idx2token, token2idx, vocab_size, sequence_len


def load_model(model_path: str, vocab_size: int,
               device: torch.device) -> MusicLSTM:
    """Load trained MusicLSTM from checkpoint."""
    ckpt = torch.load(model_path, map_location=device)
    cfg  = ckpt["config"]

    model = MusicLSTM(
        vocab_size=vocab_size,
        embed_dim=cfg["embed_dim"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=0.0,   # no dropout at inference
    ).to(device)

    # Handle DataParallel state dict (keys prefixed with "module.")
    state = ckpt["model_state"]
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}

    model.load_state_dict(state)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded from epoch %d (val_loss=%.4f, params=%s)",
                ckpt["epoch"], ckpt["val_loss"], f"{n_params:,}")
    return model


def generate_sequence(
    model: MusicLSTM,
    seed_tokens: list[int],
    n_tokens: int,
    temperature: float,
    sequence_len: int,
    device: torch.device,
) -> list[int]:
    """
    Generate n_tokens new tokens autoregressively.

    Temperature controls creativity:
    - 0.7 : conservative, more repetitive but coherent
    - 0.9 : balanced
    - 1.1 : creative, more surprising

    Same principle as temperature in LLMs (GPT, Claude) or
    guidance_scale in image diffusion models.
    """
    generated   = []
    current_seq = list(seed_tokens[-sequence_len:])

    with torch.no_grad():
        for i in range(n_tokens):
            x      = torch.tensor([current_seq], dtype=torch.long).to(device)
            logits = model(x)[0]                          # (vocab_size,)
            scaled = logits / temperature
            probs  = torch.softmax(scaled, dim=-1)
            next_tok = torch.multinomial(probs, 1).item()

            generated.append(next_tok)
            current_seq = current_seq[1:] + [next_tok]   # slide window

            if (i + 1) % 100 == 0:
                logger.info("  Generated %d / %d tokens...", i + 1, n_tokens)

    return generated


def tokens_to_midi(
    token_indices: list[int],
    idx2token: dict,
    output_path: str,
    bpm: int = 120,
) -> None:
    """
    Decode token indices → music21 notes → .mid file.

    Each token is a (pitch, duration_label, velocity_label) triple.
    We map labels back to numeric values and build a music21 Part.
    """
    part = stream.Part()
    part.append(m21_tempo.MetronomeMark(number=bpm))

    skipped = 0
    for idx in token_indices:
        if idx not in idx2token:
            skipped += 1
            continue

        pitch_midi, dur_label, vel_label = idx2token[idx]
        quarter_len = DURATION_MAP.get(dur_label, 1.0)
        velocity    = VELOCITY_MAP.get(vel_label, 64)

        n = note.Note()
        n.pitch.midi             = pitch_midi
        n.duration.quarterLength = quarter_len
        n.volume.velocity        = velocity
        part.append(n)

    if skipped:
        logger.warning("Skipped %d unknown token indices", skipped)

    score = stream.Score([part])
    score.write("midi", fp=output_path)
    logger.info("MIDI saved: %s (%d notes, %d BPM)",
                output_path, len(token_indices) - skipped, bpm)


def midi_to_wav(midi_path: str, wav_path: str, soundfont_path: str) -> bool:
    """Convert .mid to .wav using FluidSynth via midi2audio."""
    try:
        from midi2audio import FluidSynth
        fs = FluidSynth(sound_font=soundfont_path)
        fs.midi_to_audio(midi_path, wav_path)
        size_mb = os.path.getsize(wav_path) / 1_000_000
        logger.info("WAV saved : %s (%.1f MB)", wav_path, size_mb)
        return True
    except ImportError:
        logger.error("midi2audio not installed. Run: pip install midi2audio")
        logger.error("Also install FluidSynth: sudo apt-get install fluidsynth")
        return False
    except Exception as e:
        logger.error("WAV conversion failed: %s", e)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate MIDI music with trained MusicLSTM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",       default="artifacts/best_model.pt",
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--vocab",       default="artifacts/vocab.json",
                        help="Path to vocabulary file (vocab.json)")
    parser.add_argument("--output",      default="artifacts/generated.mid",
                        help="Output MIDI file path")
    parser.add_argument("--n_tokens",    type=int, default=512,
                        help="Number of tokens to generate (~notes)")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Sampling temperature (0.7=conservative, 1.1=creative)")
    parser.add_argument("--bpm",         type=int, default=120,
                        help="Tempo in beats per minute")
    parser.add_argument("--seed",        type=int, default=None,
                        help="Random seed for reproducibility (default: random)")
    parser.add_argument("--audio",       action="store_true",
                        help="Also convert output MIDI to WAV")
    parser.add_argument("--device",      default="auto",
                        help="Device to use: auto, cpu, cuda (default: auto)")
    parser.add_argument("--soundfont",   default=SOUNDFONT_PATH,
                        help="Path to soundfont .sf2 for WAV conversion")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed = args.seed if args.seed is not None else random.randint(0, 99999)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info("Random seed: %d", seed)

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    # ── Output dir ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # ── Load vocab & model ────────────────────────────────────────────────────
    # Auto-download model from HuggingFace if not present locally
    model_path = ensure_model(args.model)
    idx2token, token2idx, vocab_size, sequence_len = load_vocab(args.vocab)
    model = load_model(model_path, vocab_size, device)

    # ── Random seed sequence ──────────────────────────────────────────────────
    # Pick sequence_len random valid tokens as the seed
    all_indices  = list(idx2token.keys())
    seed_tokens  = random.choices(all_indices, k=sequence_len)
    logger.info("Seed: %d random tokens from vocabulary", sequence_len)

    # ── Generate ──────────────────────────────────────────────────────────────
    logger.info("Generating %d tokens at temperature %.2f...",
                args.n_tokens, args.temperature)
    generated = generate_sequence(
        model, seed_tokens, args.n_tokens,
        args.temperature, sequence_len, device,
    )
    logger.info("Generation complete: %d tokens", len(generated))

    # ── MIDI export ───────────────────────────────────────────────────────────
    tokens_to_midi(generated, idx2token, args.output, bpm=args.bpm)

    # ── WAV conversion (optional) ─────────────────────────────────────────────
    if args.audio:
        soundfont = ensure_soundfont() if args.soundfont == SOUNDFONT_PATH \
                    else args.soundfont
        if soundfont:
            wav_path = str(Path(args.output).with_suffix(".wav"))
            midi_to_wav(args.output, wav_path, soundfont)

    logger.info("Done. Output: %s", args.output)


if __name__ == "__main__":
    main()
