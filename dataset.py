import csv
import warnings
from pathlib import Path

import torch
import torchaudio

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio._backend.utils")
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def _load_audio(path: str, target_sr: int, max_duration_seconds: int) -> torch.Tensor:
    """Load audio file, convert to mono, resample, then repeat-pad or truncate."""
    waveform = None
    try:
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        waveform = waveform.squeeze(0)
    except Exception:
        import librosa
        import numpy as np
        y, _ = librosa.load(path, sr=target_sr, mono=True)
        waveform = torch.from_numpy(y)

    max_samples = target_sr * max_duration_seconds
    if waveform.shape[0] >= max_samples:
        waveform = waveform[:max_samples]
    else:
        repeats = (max_samples // waveform.shape[0]) + 1
        waveform = waveform.repeat(repeats)[:max_samples]

    return waveform


class ASVspoof2019Dataset(Dataset):
    """ASVspoof 2019 LA dataset.

    Protocol format (whitespace-separated, 5 columns):
        SPEAKER_ID FILENAME - ATTACK_ID LABEL
    """

    def __init__(
        self,
        root_dir: str,
        protocol_path: str,
        subset: str = "all",
        target_sample_rate: int = 16000,
        max_duration_seconds: int = 5,
    ):
        self.root_dir = Path(root_dir)
        self.target_sample_rate = target_sample_rate
        self.max_duration_seconds = max_duration_seconds
        self.samples = []

        with open(protocol_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                # Actual format: FILEPATH - LABEL DATASET SPEAKER LANG
                # parts[0] is a (possibly relative) path; extract just the filename
                filename = Path(parts[0]).name
                label_str = parts[2]
                label = 1 if label_str == "bonafide" else 0
                if subset == "bonafide" and label != 1:
                    continue
                if subset == "spoof" and label != 0:
                    continue
                self.samples.append((filename, label))

        print(f"ASVspoof2019 ({subset}): {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        path = self.root_dir / filename
        waveform = _load_audio(str(path), self.target_sample_rate, self.max_duration_seconds)
        return waveform, label, Path(filename).stem


class ASVspoof5Dataset(Dataset):
    """ASVspoof 5 dataset.

    Train/dev (6 columns):  SPEAKER_ID FILENAME GENDER CODEC ATTACK_ID LABEL
    Eval TSV   (10 columns): SPEAKER_ID FILENAME GENDER CODEC N REF AC1 ATTACK_ID LABEL -
    Label column is auto-detected by row width.
    """

    def __init__(
        self,
        root_dir: str,
        protocol_path: str,
        subset: str = "all",
        target_sample_rate: int = 16000,
        max_duration_seconds: int = 5,
    ):
        self.root_dir = Path(root_dir)
        self.target_sample_rate = target_sample_rate
        self.max_duration_seconds = max_duration_seconds
        self.samples = []

        with open(protocol_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                filename = parts[1]
                # 10-column eval TSV has label at index 8; 6-column train/dev at index 5
                label_str = parts[8] if len(parts) >= 9 else parts[5]
                if not filename.endswith(".flac"):
                    filename = filename + ".flac"
                label = 1 if label_str == "bonafide" else 0
                if subset == "bonafide" and label != 1:
                    continue
                if subset == "spoof" and label != 0:
                    continue
                self.samples.append((filename, label))

        print(f"ASVspoof5 ({subset}): {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        path = self.root_dir / filename
        waveform = _load_audio(str(path), self.target_sample_rate, self.max_duration_seconds)
        return waveform, label, filename


class InTheWildDataset(Dataset):
    """In-The-Wild dataset.

    Protocol is a headerless CSV: file, speaker, label
    Labels: bonafide / bona-fide (normalized to bonafide)
    """

    def __init__(
        self,
        root_dir: str,
        protocol_path: str,
        subset: str = "all",
        target_sample_rate: int = 16000,
        max_duration_seconds: int = 5,
    ):
        self.root_dir = Path(root_dir)
        self.target_sample_rate = target_sample_rate
        self.max_duration_seconds = max_duration_seconds
        self.samples = []

        with open(protocol_path, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3:
                    continue
                filename = row[0].strip()
                label_str = row[2].strip().replace("bona-fide", "bonafide")
                label = 1 if label_str == "bonafide" else 0
                if subset == "bonafide" and label != 1:
                    continue
                if subset == "spoof" and label != 0:
                    continue
                self.samples.append((filename, label))

        print(f"InTheWild ({subset}): {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        path = self.root_dir / filename
        waveform = _load_audio(str(path), self.target_sample_rate, self.max_duration_seconds)
        return waveform, label, filename


class FakeXposeDataset(Dataset):
    """FakeXpose dataset — no protocol file, directory-based labels.

    root_dir/ElevenLabs/ (or 11labs/) -> spoof (label=0)
    root_dir/Original/   (or original/) -> bonafide (label=1)
    """

    _SPOOF_NAMES = {"elevenlabs", "11labs", "11 labs", "11_labs"}
    _BONAFIDE_NAMES = {"original"}
    _EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a"}

    def __init__(
        self,
        root_dir: str,
        subset: str = "all",
        target_sample_rate: int = 16000,
        max_duration_seconds: int = 5,
    ):
        self.target_sample_rate = target_sample_rate
        self.max_duration_seconds = max_duration_seconds
        self.samples = []

        root = Path(root_dir)
        for top_dir in root.iterdir():
            if not top_dir.is_dir():
                continue
            name_lower = top_dir.name.lower()
            if name_lower in self._SPOOF_NAMES:
                label = 0
            elif name_lower in self._BONAFIDE_NAMES:
                label = 1
            else:
                continue
            if subset == "bonafide" and label != 1:
                continue
            if subset == "spoof" and label != 0:
                continue
            for p in top_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in self._EXTENSIONS:
                    self.samples.append((str(p), label))

        self.samples.sort(key=lambda x: x[0])
        print(f"FakeXpose ({subset}): {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        audio_name = Path(path).name
        waveform = _load_audio(path, self.target_sample_rate, self.max_duration_seconds)
        return waveform, label, audio_name


class MLAADDataset(Dataset):
    """MLAAD dataset.

    Protocol format (whitespace-separated, 5 columns):
        rel_path lang corpus source_or_attack label
    """

    def __init__(
        self,
        root_dir: str,
        protocol_path: str,
        subset: str = "all",
        target_sample_rate: int = 16000,
        max_duration_seconds: int = 5,
    ):
        self.root_dir = Path(root_dir)
        self.target_sample_rate = target_sample_rate
        self.max_duration_seconds = max_duration_seconds
        self.samples = []

        with open(protocol_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                rel_path, label_str = parts[0], parts[4]
                audio_name = Path(rel_path).name
                if audio_name.startswith("._"):
                    continue
                label = 1 if label_str == "bonafide" else 0
                if subset == "bonafide" and label != 1:
                    continue
                if subset == "spoof" and label != 0:
                    continue
                self.samples.append((rel_path, label, audio_name))

        print(f"MLAAD ({subset}): {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label, audio_name = self.samples[idx]
        path = self.root_dir / rel_path
        waveform = _load_audio(str(path), self.target_sample_rate, self.max_duration_seconds)
        return waveform, label, audio_name


class FamousFiguresDataset(Dataset):
    """Famous Figures dataset.

    Protocol is a TSV with header: AudioName Speaker Source Label AudioPath
    AudioPath may be absolute or relative to root_dir.
    """

    def __init__(
        self,
        root_dir: str,
        protocol_path: str,
        subset: str = "all",
        target_sample_rate: int = 16000,
        max_duration_seconds: int = 5,
    ):
        self.root_dir = Path(root_dir)
        self.target_sample_rate = target_sample_rate
        self.max_duration_seconds = max_duration_seconds
        self.samples = []

        with open(protocol_path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                audio_name = row["AudioName"].strip()
                label_str = row["Label"].strip().replace("bona-fide", "bonafide")
                audio_path_str = row["AudioPath"].strip()
                label = 1 if label_str == "bonafide" else 0
                if subset == "bonafide" and label != 1:
                    continue
                if subset == "spoof" and label != 0:
                    continue
                p = Path(audio_path_str)
                if not p.is_absolute():
                    p = self.root_dir / p
                self.samples.append((audio_name, label, str(p)))

        print(f"FamousFigures ({subset}): {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_name, label, path = self.samples[idx]
        waveform = _load_audio(path, self.target_sample_rate, self.max_duration_seconds)
        return waveform, label, audio_name


class ATADDDataset(Dataset):
    """AT-ADD Track 1 dataset.

    Protocol: CSV with header  name,label,generator
    Labels: "real" -> 1 (bonafide), "fake" -> 0 (spoof)
    Audio files sit directly in root_dir.
    """

    def __init__(
        self,
        root_dir: str,
        protocol_path: str,
        subset: str = "all",
        target_sample_rate: int = 16000,
        max_duration_seconds: int = 5,
    ):
        self.root_dir = Path(root_dir)
        self.target_sample_rate = target_sample_rate
        self.max_duration_seconds = max_duration_seconds
        self.samples = []

        with open(protocol_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row["name"].strip()
                label = 1 if row["label"].strip() == "real" else 0
                if subset == "bonafide" and label != 1:
                    continue
                if subset == "spoof" and label != 0:
                    continue
                self.samples.append((filename, label))

        print(f"ATADD ({subset}): {len(self.samples)} samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        path = self.root_dir / filename
        waveform = _load_audio(str(path), self.target_sample_rate, self.max_duration_seconds)
        return waveform, label, Path(filename).stem


def collate_fn(batch):
    """Pad waveforms to longest in batch.

    Returns:
        waveforms: [B, T]
        labels:    [B]
        audio_names: list[str]
    """
    waveforms, labels, audio_names = zip(*batch)
    padded = pad_sequence(list(waveforms), batch_first=True, padding_value=0.0)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return padded, labels_tensor, list(audio_names)


# --- smoke test ---
if __name__ == "__main__":
    from torch.utils.data import Dataset, DataLoader

    class _DummyDataset(Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, idx):
            return torch.randn(8000 + idx * 1000), idx % 2, f"dummy_{idx}"

    ds = _DummyDataset()
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
    waveforms, labels, names = next(iter(loader))
    assert waveforms.shape == (4, 11000), f"Expected (4, 11000), got {waveforms.shape}"
    assert labels.shape == (4,)
    assert len(names) == 4
    print(f"collate_fn smoke test passed: waveforms={waveforms.shape}, labels={labels.shape}")
