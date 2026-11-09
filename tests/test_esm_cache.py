"""Testes unitários do motor de extração e cache de embeddings do ESM-2 (Task 4)."""

from __future__ import annotations

import pytest

from physdual.data.esm_cache import compute_seq_hash, generate_sliding_windows


def test_seq_hash_deterministic():
    seq = "MKTVRQERLKSIVRILERSKEPVS"
    h1 = compute_seq_hash(seq)
    h2 = compute_seq_hash(seq)
    assert h1 == h2
    assert len(h1) == 16


def test_sliding_windows_coverage_and_overlap():
    # Para L <= max_len: janela única [0, L)
    w_short = generate_sliding_windows(seq_len=500, max_len=1022, overlap=256)
    assert w_short == [(0, 500)]

    # Para L = 1500, max_len = 1022, overlap = 256 (stride = 766):
    # janela 0: [0, 1022), janela 1: [766, 1500) -> sobreposição em [766, 1022) (256 resíduos)
    w_long = generate_sliding_windows(seq_len=1500, max_len=1022, overlap=256)
    assert len(w_long) == 2
    assert w_long[0] == (0, 1022)
    assert w_long[1][1] == 1500
    assert w_long[0][1] - w_long[1][0] >= 256


def test_save_fp16_and_load_fp32_with_model_isolation(tmp_path):
    torch = pytest.importorskip("torch")
    from physdual.data.esm_cache import (
        load_cached_protein_embedding,
        save_protein_embedding,
    )

    seq = "MKTVRQERLKSIV"
    emb_650m = torch.randn(len(seq), 1280, dtype=torch.float32)
    emb_150m = torch.randn(len(seq), 640, dtype=torch.float32)

    p1 = save_protein_embedding(seq, emb_650m, model_name="esm2_t33_650M_UR50D", cache_dir=tmp_path)
    p2 = save_protein_embedding(seq, emb_150m, model_name="esm2_t30_150M_UR50D", cache_dir=tmp_path)

    assert p1 != p2
    assert p1.parent.name == "esm2_t33_650M_UR50D"
    assert p2.parent.name == "esm2_t30_150M_UR50D"

    raw_payload = torch.load(p1, weights_only=True)
    assert raw_payload["embedding"].dtype == torch.float16
    assert raw_payload["dim"] == 1280
    assert raw_payload["seq_len"] == len(seq)

    loaded_650m = load_cached_protein_embedding(
        seq, model_name="esm2_t33_650M_UR50D", cache_dir=tmp_path, expected_dim=1280
    )
    assert loaded_650m.dtype == torch.float32
    assert loaded_650m.shape == (len(seq), 1280)
    assert torch.allclose(emb_650m, loaded_650m, atol=1e-2)

    with pytest.raises(ValueError, match="dim|1280"):
        load_cached_protein_embedding(
            seq, model_name="esm2_t33_650M_UR50D", cache_dir=tmp_path, expected_dim=640
        )


def test_tarball_per_model(tmp_path):
    torch = pytest.importorskip("torch")
    from physdual.data.esm_cache import create_cache_tarball, save_protein_embedding

    seq = "MKTVRQ"
    emb = torch.randn(len(seq), 320, dtype=torch.float32)
    save_protein_embedding(seq, emb, model_name="esm2_t6_8M_UR50D", cache_dir=tmp_path)
    tar_path = create_cache_tarball(tmp_path, model_name="esm2_t6_8M_UR50D")
    assert tar_path.exists()
    assert tar_path.name == "esm_embeddings_esm2_t6_8M_UR50D.tar"


def test_long_sequence_sliding_window_1500aa(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("esm")
    from physdual.data.esm_cache import (
        extract_esm_embeddings_batch,
        load_cached_protein_embedding,
    )

    long_seq = ("MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG" * 24)[:1500]
    assert len(long_seq) == 1500

    try:
        paths, report = extract_esm_embeddings_batch(
            sequences=[("T_LONG", long_seq)],
            model_name="esm2_t6_8M_UR50D",
            cache_dir=tmp_path,
            max_len=1022,
            overlap=256,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    except Exception as exc:
        if "URLError" in str(type(exc)) or "Connection" in str(exc) or "HTTPError" in str(exc):
            pytest.skip(f"Sem conexão ou pesos locais para baixar esm2_t6_8M_UR50D: {exc}")
        raise

    assert len(paths) == 1
    assert report["num_windowed_sequences"] == 1
    emb = load_cached_protein_embedding(
        long_seq, model_name="esm2_t6_8M_UR50D", cache_dir=tmp_path, expected_dim=320
    )
    assert emb.shape == (1500, 320)
    assert emb.dtype == torch.float32
    assert not torch.isnan(emb).any()
    assert not torch.isinf(emb).any()
