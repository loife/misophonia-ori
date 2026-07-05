from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import librosa
import pandas as pd
import soundfile as sf

import hashlib
from tqdm import tqdm
import numpy as np

COLUMNS = [
    "split", # 'train' | 'valid' | 'test'
    "clip_id",
    "origin_file", 
    "label",
    "start",
    "end",
]

TARGET_SR = 16_000


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in COLUMNS})


def _is_float(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False


def parse_elan_txt(txt_path: Path) -> list[tuple[str, float, float]]:
    out: list[tuple[str, float, float]] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [c.strip() for c in line.rstrip("\n").split("\t")]
            parts = [c for c in parts if c != ""]
            if not parts:
                continue
            label = parts[0]
            nums = [float(c) for c in parts[1:] if _is_float(c)]
            if len(nums) < 2:
                continue 
            start_s, end_s = nums[0], nums[1]
            if end_s < start_s:
                start_s, end_s = end_s, start_s
            out.append((label, start_s, end_s))
    return out


def _clip_event(clip_start: float, clip_end: float,
                ev_start: float, ev_end: float,
                min_overlap: float) -> tuple[float, float] | None:
    lo = max(clip_start, ev_start)
    hi = min(clip_end, ev_end)
    if hi - lo > min_overlap:
        return round(lo - clip_start, 3), round(hi - clip_start, 3)
    return None

def build_from_elan(
    dirs: Iterable[str],
    out_dir: str,
    split: str = "train",
    event_labels: Iterable[str] = ("Chewing",),
    clip_label: str = "Clip",
    target_sr: int = TARGET_SR,
    min_overlap: float = 0.0,
    chunk_size: float = 10.0,
    pad_threshold: float = 7.0,
    ) -> pd.DataFrame:
    out_dir = Path(out_dir)
    event_labels = set(event_labels)

    rows: list[dict] = []

    for dir in dirs:
        print(f"[info] Looking through strongly labeled dir: {dir}")
        dir = Path(dir)    
        for txt_path in tqdm(sorted(dir.glob("*.txt"))):
            stem = txt_path.stem

            src_wav = dir / f"{stem}.wav"

            if not src_wav.is_file():
                print(f"[warn] no .wav found for '{stem}', skipping")
                continue

            stem_hash = hashlib.md5(stem.encode(), usedforsecurity=False).hexdigest()

            src_dur = sf.info(str(src_wav)).duration
            annotations = parse_elan_txt(txt_path)

            clips = sorted((s, e) for (label, s, e) in annotations if label == clip_label)
            events = [(label, s, e) for (label, s, e) in annotations if label in event_labels]

            if not clips:
                print(f"[warn] '{stem}' has no '{clip_label}' markers, skipping")
                continue

            for i, (c_start, c_end) in enumerate(clips):
                clip_id = f"{stem_hash}_clip{i:03d}"
                c_dur = round(c_end - c_start, 3)

                if c_end > src_dur:
                    print(f"[warn] {clip_id}: clip end {c_end:.2f}s exceeds audio length {src_dur:.2f}s, cutting to fit")
                    c_end = src_dur

                clip_dur = c_end - c_start
                n_full = int(clip_dur // chunk_size)
                remainder = round(clip_dur - n_full * chunk_size, 3)
                windows = [(round(c_start + j * chunk_size, 3), chunk_size) for j in range(n_full)]

                if remainder > pad_threshold:
                    windows.append((round(c_start + n_full * chunk_size, 3), remainder))

                if not windows:
                    print(f"[warn] clip {i} of '{stem}' shorter than pad threshold, dropped")
                    continue

                for j, (ch_start, real_len) in enumerate(windows):
                    clip_id = f"{stem_hash}_clip{i:03d}_chunk{j:02d}"
                    ch_end = round(ch_start + real_len, 3)

                    out_wav = out_dir / str(target_sr) / split / f"{clip_id}.wav"
                    out_wav.parent.mkdir(parents=True, exist_ok=True)
                    y = _load_chunk(src_wav, ch_start, real_len, chunk_size, target_sr)
                    sf.write(str(out_wav), y, target_sr, subtype="PCM_16")

                    base = {
                        "split": split, "clip_id": clip_id, "origin_file": stem
                    }

                    kept = []
                    for lab, e_start, e_end in events:
                        lo = max(ch_start, e_start)
                        hi = min(ch_end, e_end) 
                        if hi - lo > min_overlap:
                            rel_start = round(lo - ch_start, 3)
                            rel_end = round(hi - ch_start, 3)
                        else:
                            continue
                        kept.append((lab, rel_start, rel_end))

                    if kept:
                        for lab, s, e in kept:
                            rows.append({**base, "label": lab, "start": s, "end": e})
                    else:
                        rows.append({**base, "label": pd.NA, "start": pd.NA, "end": pd.NA})

    df = pd.DataFrame(rows, columns=COLUMNS)
    return df

def build_weak_as_strong(
        dirs: Iterable[str],
        out_dir: str,
        split: str = "train",
        event_labels: Iterable[str] = ("Chewing",),
        target_sr: int = TARGET_SR,
        chunk_size: float = 10.0,
        pad_threshold: float = 7.0,):
    out_dir = Path(out_dir)
    event_labels = set(event_labels)

    rows: list[dict] = []

    for dir in dirs:
        print(f"[info] Looking through weakly labeled dir, writing as strong: {dir}")
        dir = Path(dir)    
        for wav_path in tqdm(sorted(dir.glob("*.wav"))):
            stem = wav_path.stem
            stem_hash = hashlib.md5(stem.encode(), usedforsecurity=False).hexdigest()
            src_dur = sf.info(str(wav_path)).duration

            n_full = int(src_dur // chunk_size)
            remainder = round(src_dur - n_full * chunk_size, 3)
            windows = [(round(j * chunk_size, 3), chunk_size) for j in range(n_full)]

            if n_full == 0:
                # Keep files that are shorter than 1 chunk
                if src_dur > 0:
                    windows.append((0.0, src_dur))
            elif remainder > pad_threshold:
                windows.append((round(n_full * chunk_size, 3), remainder))

            for j, (ch_start, real_len) in enumerate(windows):
                    clip_id = f"{stem_hash}_chunk{j:02d}"
                    ch_end = round(ch_start + real_len, 3)

                    out_wav = out_dir / str(target_sr) / split / f"{clip_id}.wav"
                    out_wav.parent.mkdir(parents=True, exist_ok=True)
                    y = _load_chunk(wav_path, ch_start, real_len, chunk_size, target_sr)
                    sf.write(str(out_wav), y, target_sr, subtype="PCM_16")

                    base = {
                        "split": split, "clip_id": clip_id, "origin_file": stem
                    }

                    kept = []
                    for lab in event_labels:
                        kept.append((lab, 0, real_len))

                    if kept:
                        for lab, s, e in kept:
                            rows.append({**base, "label": lab, "start": s, "end": e})
                    else:
                        rows.append({**base, "label": pd.NA, "start": pd.NA, "end": pd.NA})

    df = pd.DataFrame(rows, columns=COLUMNS)
    return df


def build_weak(dirs: Iterable[str],
        out_dir: str,
        split: str = "train",
        event_labels: Iterable[str] = ("Chewing",),
        target_sr: int = TARGET_SR,
        chunk_size: float = 10.0,
        pad_threshold: float = 7.0):
    
    rows: list[dict] = []
    out_dir = Path(out_dir)
    
    for dir in dirs:
        print(f"[info] Looking through weakly labeled dir: {dir}")
        dir = Path(dir)    
        for wav_path in tqdm(sorted(dir.glob("*.wav"))):
            stem = wav_path.stem
            stem_hash = hashlib.md5(stem.encode(), usedforsecurity=False).hexdigest()
            src_dur = sf.info(str(wav_path)).duration

            n_full = int(src_dur // chunk_size)
            remainder = round(src_dur - n_full * chunk_size, 3)
            windows = [(round(j * chunk_size, 3), chunk_size) for j in range(n_full)]

            if remainder > pad_threshold:
                windows.append((round(n_full * chunk_size, 3), remainder))

            for j, (ch_start, real_len) in enumerate(windows):
                    clip_id = f"{stem_hash}_chunk{j:02d}"
                    ch_end = round(ch_start + real_len, 3)

                    out_wav = out_dir / str(target_sr) / split / f"{clip_id}.wav"
                    out_wav.parent.mkdir(parents=True, exist_ok=True)
                    y = _load_chunk(wav_path, ch_start, real_len, chunk_size, target_sr)
                    sf.write(str(out_wav), y, target_sr, subtype="PCM_16")

                    base = {
                        "split": split, "clip_id": clip_id, "origin_file": stem
                    }

                    kept = []
                    for lab in event_labels:
                        kept.append(lab)

                    if kept:
                        for lab in kept:
                            rows.append({**base, "label": lab})
                    else:
                        rows.append({**base, "label": pd.NA})
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df

def _load_chunk(src_wav: Path, start_s: float, real_len: float,
                chunk_size: float, target_sr: int) -> np.ndarray:
    y, _ = librosa.load(str(src_wav), sr=target_sr, mono=True,
                        offset=start_s, duration=real_len)
    n = int(round(chunk_size * target_sr))
    return np.pad(y, (0, n - len(y))) if len(y) < n else y[:n]


def write_hear(df: pd.DataFrame, out_dir: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hear = {}
    for (split, clip_id), grp in df.groupby(["split", "clip_id"], sort=True):
        events = []
        for _, row in grp.iterrows():
            if pd.isna(row["label"]):
                continue
            events.append({
                "start": int(round(float(row["start"]) * 1000)),
                "end": int(round(float(row["end"]) * 1000)),
                "label": str(row["label"]),
            })
        hear.setdefault(split, {})[f"{clip_id}.wav"] = events

    for split, mapping in hear.items():
        with open(out_dir / f"{split}.json", "w") as f:
            json.dump(mapping, f, indent=2)

    vocab = sorted(df["label"].dropna().astype(str).unique())
    pd.DataFrame({"label": vocab, "idx": range(len(vocab))}).to_csv(out_dir / "labelvocabulary.csv", index=False)


def write_weak(df: pd.DataFrame,
               out_dir: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for (split, clip_id), grp in df.groupby(["split", "clip_id"], sort=True):
        events = []
        for _, row in grp.iterrows():
            if pd.isna(row["label"]):
                continue
            events.append(str(row["label"]))
        manifest.setdefault(split, {})[f"{clip_id}.wav"] = events

    for split, mapping in manifest.items():
        with open(out_dir / f"{split}.json", "w") as f:
            json.dump(mapping, f, indent=2)

    vocab = sorted(df["label"].dropna().astype(str).unique())
    pd.DataFrame({"label": vocab, "idx": range(len(vocab))}).to_csv(out_dir / "labelvocabulary.csv", index=False)


if __name__ == "__main__":
    # str_label_dirs = ["./videos/labeled/movies", "./videos/labeled/yt", "./videos/mukbang/original"]
    # negative_dirs = ["./videos/negative", "./videos/AudioSet/negative"]
    # str_positive_dirs = ["./videos/synthesis/script_output", "./videos/MATA"]
    # str_negative_dirs = ["./videos/synthesis/used_ambience"]

    # weak_positive_dirs = ["./videos/AudioSet/positive"]
    # weak_negative_dirs = ["./videos/AudioSet/negative"]

    tr_str_label_dirs = ["./videos/labeled/movies/train", "./videos/labeled/yt/train", "./videos/mukbang/train"]
    tr_str_positive_dirs = ["./videos/synthesis/script_output/train", "./videos/MATA/train"]
    tr_str_negative_dirs = ["./videos/synthesis/used_ambience/train"]

    tr_weak_positive_dirs = ["./videos/AudioSet/positive/train"]
    tr_weak_negative_dirs = ["./videos/AudioSet/negative/train"]

    eval_str_label_dirs = ["./videos/labeled/movies/eval", "./videos/labeled/yt/eval", "./videos/mukbang/eval"]
    eval_str_positive_dirs = ["./videos/synthesis/script_output/eval", "./videos/MATA/eval"]
    eval_str_negative_dirs = ["./videos/synthesis/used_ambience/eval"]

    eval_weak_positive_dirs = ["./videos/AudioSet/positive/eval"]
    eval_weak_negative_dirs = ["./videos/AudioSet/negative/eval"]

    # Build train ds
    tr_strong_df = build_from_elan(
        tr_str_label_dirs, "./out/strong", split="train",
        event_labels=["Chewing"], clip_label="Clip"
    )

    tr_str_positives_df = build_weak_as_strong(
        tr_str_positive_dirs, "./out/strong", split="train",
        event_labels=["Chewing"]
    )

    tr_str_negatives_df = build_weak_as_strong(
        tr_str_negative_dirs, "./out/strong", split="train",
        event_labels=[]
    )

    tr_strong_df = pd.concat([tr_strong_df, tr_str_positives_df], ignore_index=True)
    tr_strong_df = pd.concat([tr_strong_df, tr_str_negatives_df], ignore_index=True)
    
    write_hear(tr_strong_df, "./out/strong")

    tr_weak_df = build_weak(tr_weak_positive_dirs, "./out/weak", split="train", event_labels=["Chewing"])
    tr_negative_df = build_weak(tr_weak_negative_dirs, "./out/weak", split="train", event_labels=[])

    tr_weak_df = pd.concat([tr_weak_df, tr_negative_df], ignore_index=True)

    write_weak(tr_weak_df, "./out/weak")

    # Build eval ds
    eval_strong_df = build_from_elan(
        eval_str_label_dirs, "./out/strong", split="valid",
        event_labels=["Chewing"], clip_label="Clip"
    )

    eval_str_positives_df = build_weak_as_strong(
        eval_str_positive_dirs, "./out/strong", split="valid",
        event_labels=["Chewing"]
    )

    eval_str_negatives_df = build_weak_as_strong(
        eval_str_negative_dirs, "./out/strong", split="valid",
        event_labels=[]
    )

    eval_strong_df = pd.concat([eval_strong_df, eval_str_positives_df], ignore_index=True)
    eval_strong_df = pd.concat([eval_strong_df, eval_str_negatives_df], ignore_index=True)
    
    write_hear(eval_strong_df, "./out/strong")

    eval_weak_df = build_weak(eval_weak_positive_dirs, "./out/weak", split="valid", event_labels=["Chewing"])
    eval_negative_df = build_weak(eval_weak_negative_dirs, "./out/weak", split="valid", event_labels=[])

    eval_weak_df = pd.concat([eval_weak_df, eval_negative_df], ignore_index=True)

    write_weak(eval_weak_df, "./out/weak")


    # n_clips = strong_df["clip_id"].nunique()
    # n_events = int(strong_df["label"].notna().sum())
    # n_neg = int(strong_df.groupby("clip_id")["label"].apply(lambda s: s.notna().sum() == 0).sum())
    # print(f"\n{n_clips} clips | {n_events} events | {n_neg} negative clips")