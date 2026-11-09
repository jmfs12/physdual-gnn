"""Extração offline e cache de embeddings proteicos do ESM-2 (B1 e B4).

Implementa:
- Isolamento de cache por modelo (`cache/esm_embeddings/{model_name}/{hash}.pt`) e tarball
  por modelo (`esm_embeddings_{model_name}.tar`).
- Armazenamento em disco em `torch.float16` com metadados primitivos (`model_name`, `dim`,
  `seq_len`, `windowed`) carregáveis com `torch.load(..., weights_only=True)` e conversão
  para `torch.float32` na leitura com validação de `expected_dim`.
- Janela deslizante com sobreposição (`max_len=1022`, `overlap=256`) para proteínas com
  mais de 1022 resíduos, removendo `<cls>` e `<eos>` em cada janela e calculando a média
  exata nas regiões sobrepostas para garantir shape final `[len(sequence), esm_dim]`.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import tarfile
from pathlib import Path
from typing import Any, Iterator

from physdual.data.datasets import validate_protein_sequence

MODEL_REPR_LAYERS: dict[str, int] = {
    "esm2_t33_650M_UR50D": 33,
    "esm2_t30_150M_UR50D": 30,
    "esm2_t12_35M_UR50D": 12,
    "esm2_t6_8M_UR50D": 6,
}


@contextlib.contextmanager
def _scoped_esm_weights_load() -> Iterator[None]:
    """Contorno estritamente escopado para `weights_only=True` no `fair-esm` (ADR #5 em `docs/DECISOES.md`)."""
    import torch

    orig_load = torch.load

    def _patched_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return orig_load(*args, **kwargs)

    safe_ctx = (
        torch.serialization.safe_globals([argparse.Namespace])
        if hasattr(torch.serialization, "safe_globals")
        else contextlib.nullcontext()
    )
    with safe_ctx:
        torch.load = _patched_load  # type: ignore[assignment]
        try:
            yield
        finally:
            torch.load = orig_load  # type: ignore[assignment]


def compute_seq_hash(sequence: str) -> str:
    """Calcula o hash SHA-256 de 16 caracteres da sequência proteica sanitizada."""
    clean_seq = validate_protein_sequence(sequence)
    return hashlib.sha256(clean_seq.encode("utf-8")).hexdigest()[:16]


def generate_sliding_windows(
    seq_len: int,
    max_len: int = 1022,
    overlap: int = 256,
) -> list[tuple[int, int]]:
    """Gera intervalos `[start, end)` de janela deslizante cobrindo `0..seq_len`."""
    if seq_len <= 0:
        raise ValueError(f"seq_len deve ser positivo, recebido: {seq_len}")
    if max_len <= 0 or overlap < 0 or overlap >= max_len:
        raise ValueError(
            f"Parâmetros de janela inválidos: max_len={max_len}, overlap={overlap} (exigido 0 <= overlap < max_len)."
        )

    if seq_len <= max_len:
        return [(0, seq_len)]

    stride = max_len - overlap
    windows: list[tuple[int, int]] = []
    start = 0
    while start < seq_len:
        end = min(start + max_len, seq_len)
        if end == seq_len and (end - start) < max_len:
            start = max(0, seq_len - max_len)
        if not windows or windows[-1] != (start, end):
            windows.append((start, end))
        if end == seq_len:
            break
        start += stride
    return windows


def get_embedding_cache_path(sequence: str, model_name: str, cache_dir: Path) -> Path:
    """Devolve o caminho padronizado `cache_dir / model_name / f'{hash}.pt'`."""
    seq_hash = compute_seq_hash(sequence)
    return Path(cache_dir) / str(model_name).strip() / f"{seq_hash}.pt"


def save_protein_embedding(
    sequence: str,
    embedding: Any,
    model_name: str,
    cache_dir: Path,
    windowed: bool = False,
) -> Path:
    """Salva o embedding por resíduo em `fp16` no diretório isolado por `model_name`.

    Payload salvo (100% compatível com `weights_only=True`):
    `{"embedding": Tensor(fp16), "model_name": str, "dim": int, "seq_len": int, "seq_hash": str, "windowed": bool}`
    """
    import torch

    clean_seq = validate_protein_sequence(sequence)
    seq_hash = compute_seq_hash(clean_seq)
    out_path = Path(cache_dir) / str(model_name).strip() / f"{seq_hash}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if embedding.ndim != 2 or int(embedding.size(0)) != len(clean_seq):
        raise ValueError(
            f"Shape inválido para embedding da sequência (len={len(clean_seq)}): {tuple(embedding.shape)}"
        )

    emb_fp16 = embedding.detach().to(device="cpu", dtype=torch.float16).contiguous()
    payload = {
        "embedding": emb_fp16,
        "model_name": str(model_name).strip(),
        "dim": int(emb_fp16.size(1)),
        "seq_len": len(clean_seq),
        "seq_hash": seq_hash,
        "windowed": bool(windowed),
    }
    torch.save(payload, out_path)
    return out_path


def load_cached_protein_embedding(
    sequence: str,
    model_name: str,
    cache_dir: Path,
    expected_dim: int | None = None,
) -> Any:
    """Carrega o embedding do disco com `weights_only=True`, valida metadados e converte para `fp32`."""
    import torch

    clean_seq = validate_protein_sequence(sequence)
    pt_path = get_embedding_cache_path(clean_seq, model_name=model_name, cache_dir=cache_dir)
    if not pt_path.exists():
        raise FileNotFoundError(
            f"Embedding não encontrado no cache para model_name='{model_name}': {pt_path}"
        )

    payload = torch.load(pt_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "embedding" not in payload:
        raise ValueError(f"Formato inválido no arquivo de cache de embedding: {pt_path}")

    saved_model = str(payload.get("model_name", ""))
    if saved_model != str(model_name).strip():
        raise ValueError(
            f"Colisão/divergência de model_name em {pt_path}: salvo='{saved_model}', solicitado='{model_name}'"
        )

    emb = payload["embedding"]
    saved_dim = int(payload.get("dim", emb.size(1)))
    saved_len = int(payload.get("seq_len", emb.size(0)))

    if saved_len != len(clean_seq) or int(emb.size(0)) != len(clean_seq):
        raise ValueError(
            f"Comprimento do embedding ({emb.size(0)}) difere de len(sequence) ({len(clean_seq)}) em {pt_path}"
        )

    if expected_dim is not None and (saved_dim != int(expected_dim) or int(emb.size(1)) != int(expected_dim)):
        raise ValueError(
            f"Dimensão do embedding salvo (dim={saved_dim}) difere de expected_dim={expected_dim} para '{model_name}'"
        )

    return emb.to(dtype=torch.float32)


def _infer_repr_layer(model: Any, model_name: str) -> int:
    """Obtém o índice da última camada de representação para o modelo ESM-2."""
    if model_name in MODEL_REPR_LAYERS:
        return MODEL_REPR_LAYERS[model_name]
    if hasattr(model, "num_layers"):
        return int(model.num_layers)
    return 33


def _extract_single_window_batch(
    model: Any,
    batch_converter: Any,
    items: list[tuple[str, str]],
    repr_layer: int,
    device: str,
    use_fp16: bool,
) -> list[Any]:
    """Executa forward pass do ESM-2 sobre um lote de janelas (cada uma com comprimento <= max_len)."""
    import torch

    _, _, batch_tokens = batch_converter(items)
    batch_tokens = batch_tokens.to(device)

    amp_enabled = use_fp16 and str(device).startswith("cuda")
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if amp_enabled
        else contextlib.nullcontext()
    )

    with torch.no_grad(), autocast_ctx:
        results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
        token_reprs = results["representations"][repr_layer]

    outputs: list[Any] = []
    for i, (_, sub_seq) in enumerate(items):
        win_len = len(sub_seq)
        # Remove <cls> (índice 0) e <eos> (índice win_len + 1), mantendo exatamente [1 : win_len + 1]
        sub_emb = token_reprs[i, 1 : win_len + 1].detach().to(device="cpu", dtype=torch.float32)
        outputs.append(sub_emb)
    return outputs


def extract_esm_embeddings_batch(
    sequences: list[tuple[str, str]],
    model_name: str,
    cache_dir: Path,
    batch_tokens: int = 2048,
    max_len: int = 1022,
    overlap: int = 256,
    device: str = "cuda",
    force_recompute: bool = False,
) -> tuple[list[Path], dict[str, Any]]:
    """Extrai embeddings por resíduo do ESM-2 com janela deslizante para `len(seq) > max_len`.

    Args:
        sequences: Lista de tuplas `(target_id, sequence)`.
        model_name: Nome do modelo `fair-esm` (ex.: `"esm2_t33_650M_UR50D"`).
        cache_dir: Diretório raiz do cache (`cache/esm_embeddings`).
        batch_tokens: Orçamento máximo de tokens por sub-lote para sequências curtas.
        max_len: Comprimento máximo por janela sem contar `<cls>` e `<eos>` (padrão 1022).
        overlap: Sobreposição entre janelas consecutivas em resíduos (padrão 256).
        device: `"cuda"` ou `"cpu"`.
        force_recompute: Se `False`, reutiliza arquivos `.pt` já existentes no cache.

    Returns:
        Tupla `(saved_paths, report)` onde `report` contém `model_name`, `dim`, `total_sequences`,
        `num_cached_reused`, `num_extracted`, `num_windowed_sequences` e `windowed_target_ids`.
    """
    import esm  # type: ignore[import-untyped]
    import torch

    cache_root = Path(cache_dir)
    clean_entries: list[tuple[str, str]] = []
    seen_hashes: set[str] = set()

    for t_id, raw_seq in sequences:
        c_seq = validate_protein_sequence(raw_seq)
        h = compute_seq_hash(c_seq)
        if h not in seen_hashes:
            seen_hashes.add(h)
            clean_entries.append((str(t_id), c_seq))

    saved_paths: list[Path] = []
    to_compute: list[tuple[str, str]] = []
    windowed_target_ids: list[str] = []
    num_cached_reused = 0

    for t_id, c_seq in clean_entries:
        if len(c_seq) > max_len:
            windowed_target_ids.append(t_id)
        pt_path = get_embedding_cache_path(c_seq, model_name=model_name, cache_dir=cache_root)
        if pt_path.exists() and not force_recompute:
            saved_paths.append(pt_path)
            num_cached_reused += 1
        else:
            to_compute.append((t_id, c_seq))

    inferred_dim: int | None = None
    if to_compute:
        if not hasattr(esm.pretrained, model_name):
            raise ValueError(f"Modelo ESM-2 desconhecido em esm.pretrained: '{model_name}'")

        loader_fn = getattr(esm.pretrained, model_name)
        with _scoped_esm_weights_load():
            model, alphabet = loader_fn()

        repr_layer = _infer_repr_layer(model, model_name)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        target_device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        model = model.to(target_device)
        batch_converter = alphabet.get_batch_converter()

        # Separa sequências curtas (<= max_len) de sequências longas (> max_len)
        short_seqs = [(t_id, s) for t_id, s in to_compute if len(s) <= max_len]
        long_seqs = [(t_id, s) for t_id, s in to_compute if len(s) > max_len]

        # 1. Processa sequências curtas em lotes respeitando batch_tokens
        short_Seqs_sorted = sorted(short_seqs, key=lambda x: len(x[1]))
        current_batch: list[tuple[str, str]] = []
        current_tokens = 0

        def _flush_short_batch(batch_items: list[tuple[str, str]]) -> None:
            nonlocal inferred_dim
            if not batch_items:
                return
            embs = _extract_single_window_batch(
                model=model,
                batch_converter=batch_converter,
                items=batch_items,
                repr_layer=repr_layer,
                device=target_device,
                use_fp16=True,
            )
            for (_, seq_str), emb_tensor in zip(batch_items, embs):
                inferred_dim = int(emb_tensor.size(1))
                p = save_protein_embedding(
                    seq_str,
                    emb_tensor,
                    model_name=model_name,
                    cache_dir=cache_root,
                    windowed=False,
                )
                saved_paths.append(p)

        for item in short_Seqs_sorted:
            tokens_needed = len(item[1]) + 2
            if current_batch and (current_tokens + tokens_needed > batch_tokens):
                _flush_short_batch(current_batch)
                current_batch = []
                current_tokens = 0
            current_batch.append(item)
            current_tokens += tokens_needed
        _flush_short_batch(current_batch)

        # 2. Processa sequências longas (> max_len) com janela deslizante e média nas sobreposições
        for t_id, l_seq in long_seqs:
            windows = generate_sliding_windows(len(l_seq), max_len=max_len, overlap=overlap)
            win_items = [(f"{t_id}_w{idx}", l_seq[w_start:w_end]) for idx, (w_start, w_end) in enumerate(windows)]
            win_embs = _extract_single_window_batch(
                model=model,
                batch_converter=batch_converter,
                items=win_items,
                repr_layer=repr_layer,
                device=target_device,
                use_fp16=True,
            )
            emb_dim = int(win_embs[0].size(1))
            inferred_dim = emb_dim
            sum_emb = torch.zeros((len(l_seq), emb_dim), dtype=torch.float32)
            counts = torch.zeros((len(l_seq), 1), dtype=torch.float32)

            for (w_start, w_end), w_emb in zip(windows, win_embs):
                sum_emb[w_start:w_end] += w_emb
                counts[w_start:w_end] += 1.0

            merged_emb = sum_emb / torch.clamp(counts, min=1.0)
            p = save_protein_embedding(
                l_seq,
                merged_emb,
                model_name=model_name,
                cache_dir=cache_root,
                windowed=True,
            )
            saved_paths.append(p)

    if inferred_dim is None and saved_paths:
        sample_payload = torch.load(saved_paths[0], map_location="cpu", weights_only=True)
        inferred_dim = int(sample_payload.get("dim", sample_payload["embedding"].size(1)))

    report: dict[str, Any] = {
        "model_name": str(model_name).strip(),
        "dim": inferred_dim,
        "total_sequences": len(clean_entries),
        "num_cached_reused": num_cached_reused,
        "num_extracted": len(to_compute),
        "num_windowed_sequences": len(windowed_target_ids),
        "windowed_target_ids": windowed_target_ids,
        "max_len": max_len,
        "overlap": overlap,
    }
    return saved_paths, report


def create_cache_tarball(
    cache_dir: Path,
    model_name: str,
    output_tar: Path | None = None,
) -> Path:
    """Empacota o subdiretório `cache_dir / model_name` em `esm_embeddings_{model_name}.tar`."""
    cache_root = Path(cache_dir)
    model_subdir = cache_root / str(model_name).strip()
    if not model_subdir.exists():
        raise FileNotFoundError(
            f"Diretório de cache do modelo '{model_name}' não encontrado: {model_subdir}"
        )

    tar_path = (
        Path(output_tar)
        if output_tar is not None
        else (cache_root / f"esm_embeddings_{str(model_name).strip()}.tar")
    )
    tar_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "w") as tar:
        tar.add(model_subdir, arcname=model_subdir.name)

    return tar_path
