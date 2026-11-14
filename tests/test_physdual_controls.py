"""Testes unitários e de smoke para os controles diagnósticos da Fase 3.

Cobre:
- `--shuffle-esm`: mapeamento fixo sequência -> embedding de outra sequência (zero pontos fixos)
  consistente entre train/val/test e distinto para permutações `0, 1, 2`.
- `--train-seed-offset`: mantém exatamente as mesmas amostras nos splits enquanto altera a semente de treino
  e gera arquivo isolado `_tsoff<k>` sem sobrescrever a run primária.
- `physdual_ecfp_esm_mlp` (`ecfp_mlp`): `ESMTargetEncoder` e `fusion_mlp` idênticos ao `PhysDualPhase3`,
  substituindo apenas o ramo molecular (`GIN` -> `ECFP-MLP`), com execução end-to-end smoke.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("rdkit")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from physdual.data.datamodule import build_shuffled_esm_map, get_dta_dataloaders  # noqa: E402
from physdual.models.physdual import PhysDualECFPPhase3, PhysDualPhase3, build_physdual_phase3_model  # noqa: E402
from test_verify_phase3 import _create_synthetic_phase3_workspace  # noqa: E402


def test_build_shuffled_esm_map_is_deterministic_derangement():
    raw_cache = {f"SEQ_{i}": torch.full((16,), float(i)) for i in range(6)}

    m0 = build_shuffled_esm_map(raw_cache, perm_seed=0)
    m0_repeat = build_shuffled_esm_map(raw_cache, perm_seed=0)
    m1 = build_shuffled_esm_map(raw_cache, perm_seed=1)
    m2 = build_shuffled_esm_map(raw_cache, perm_seed=2)

    assert set(m0.keys()) == set(raw_cache.keys())
    for k in raw_cache:
        assert not torch.equal(m0[k], raw_cache[k]), f"Sequência {k} manteve o próprio embedding em seed=0"
        assert not torch.equal(m1[k], raw_cache[k]), f"Sequência {k} manteve o próprio embedding em seed=1"
        assert not torch.equal(m2[k], raw_cache[k]), f"Sequência {k} manteve o próprio embedding em seed=2"
        assert torch.equal(m0[k], m0_repeat[k])

    assert any(not torch.equal(m0[k], m1[k]) for k in raw_cache)
    assert any(not torch.equal(m1[k], m2[k]) for k in raw_cache)


def test_dataloaders_shuffle_esm_consistent_across_splits(tmp_path: Path):
    cfg, esm_dir = _create_synthetic_phase3_workspace(tmp_path)

    tr_true, _, te_true = get_dta_dataloaders(
        dataset_name="davis",
        split_type="cold_target_seqid30",
        config=cfg,
        root=tmp_path,
        protocol="kfold",
        fold=0,
        use_esm=True,
        esm_cache_dir=esm_dir,
        shuffle_esm=False,
    )
    tr_s0, val_s0, te_s0 = get_dta_dataloaders(
        dataset_name="davis",
        split_type="cold_target_seqid30",
        config=cfg,
        root=tmp_path,
        protocol="kfold",
        fold=0,
        use_esm=True,
        esm_cache_dir=esm_dir,
        shuffle_esm=True,
        shuffle_esm_seed=0,
    )
    # Em cold_target cada partição tem sequências próprias; verifica que:
    # (a) nenhuma sequência recebeu o próprio vetor verdadeiro
    # (b) a bijeção global dos embeddings é preservada entre partições e dobras (fold=0 vs fold=1)
    _, _, te_s0_fold1 = get_dta_dataloaders(
        dataset_name="davis",
        split_type="cold_target_seqid30",
        config=cfg,
        root=tmp_path,
        protocol="kfold",
        fold=1,
        use_esm=True,
        esm_cache_dir=esm_dir,
        shuffle_esm=True,
        shuffle_esm_seed=0,
    )

    true_vecs = {}
    true_vecs.update(tr_true.dataset.unique_esm_vecs)
    true_vecs.update(te_true.dataset.unique_esm_vecs)

    shuf_vecs_f0 = {}
    shuf_vecs_f0.update(tr_s0.dataset.unique_esm_vecs)
    shuf_vecs_f0.update(val_s0.dataset.unique_esm_vecs)
    shuf_vecs_f0.update(te_s0.dataset.unique_esm_vecs)

    for seq, vec in shuf_vecs_f0.items():
        if seq in true_vecs:
            assert not torch.equal(vec, true_vecs[seq])

    for seq, vec in te_s0_fold1.dataset.unique_esm_vecs.items():
        assert torch.equal(vec, shuf_vecs_f0[seq])


def test_ecfp_mlp_shares_identical_target_encoder_and_fusion_mlp(tmp_path: Path, monkeypatch):
    from run_phase3_physdual import aggregate_phase3_kfold_runs, run_single_unit

    monkeypatch.setenv("PHYSDUAL_ROOT", str(tmp_path))
    cfg, esm_dir = _create_synthetic_phase3_workspace(tmp_path)

    m_gin = build_physdual_phase3_model(cfg, model_name="physdual_gin_esm_mlp")
    m_ecfp = build_physdual_phase3_model(cfg, model_name="ecfp_mlp")

    assert isinstance(m_gin, PhysDualPhase3)
    assert isinstance(m_ecfp, PhysDualECFPPhase3)
    assert m_ecfp.expects_ecfp is True

    # ESMTargetEncoder e fusion_mlp idênticos em arquitetura e contagem de parâmetros
    assert sum(p.numel() for p in m_gin.target_encoder.parameters()) == sum(
        p.numel() for p in m_ecfp.target_encoder.parameters()
    )
    assert sum(p.numel() for p in m_gin.fusion_mlp.parameters()) == sum(
        p.numel() for p in m_ecfp.fusion_mlp.parameters()
    )
    assert [tuple(p.shape) for p in m_gin.fusion_mlp.parameters()] == [
        tuple(p.shape) for p in m_ecfp.fusion_mlp.parameters()
    ]

    for fold in (0, 1):
        res = run_single_unit(
            dataset_name="davis",
            split_type="cold_target_seqid30",
            protocol="kfold",
            unit_id=fold,
            config=cfg,
            root=tmp_path,
            esm_cache_dir=esm_dir,
            force=True,
            device="cpu",
            smoke=True,
            model_name="ecfp_mlp",
        )
        assert res["model"] == "physdual_ecfp_esm_mlp"
        assert res["base_model"] == "physdual_ecfp_esm_mlp"
        assert res["test_isolation_verified"] is True
        assert res["diagnostic_control"]["is_control"] is True

    agg = aggregate_phase3_kfold_runs(
        dataset="davis",
        split="cold_target_seqid30",
        folds=[0, 1],
        config=cfg,
        root=tmp_path,
        model="physdual_ecfp_esm_mlp",
    )
    assert agg["model"] == "physdual_ecfp_esm_mlp"
    assert (tmp_path / "outputs" / "results" / "phase3_physdual_ecfp_esm_mlp_davis_cold_target_seqid30_kfold.json").exists()
    assert not (tmp_path / "outputs" / "results" / "phase3_physdual_gin_esm_mlp_davis_cold_target_seqid30_kfold.json").exists()


def test_train_seed_offset_preserves_split_and_isolates_outputs(tmp_path: Path, monkeypatch):
    from run_phase3_physdual import run_single_unit

    monkeypatch.setenv("PHYSDUAL_ROOT", str(tmp_path))
    cfg, esm_dir = _create_synthetic_phase3_workspace(tmp_path)

    res0 = run_single_unit(
        dataset_name="davis",
        split_type="cold_target_seqid30",
        protocol="kfold",
        unit_id=0,
        config=cfg,
        root=tmp_path,
        esm_cache_dir=esm_dir,
        force=True,
        device="cpu",
        smoke=True,
        train_seed_offset=0,
    )
    res1 = run_single_unit(
        dataset_name="davis",
        split_type="cold_target_seqid30",
        protocol="kfold",
        unit_id=0,
        config=cfg,
        root=tmp_path,
        esm_cache_dir=esm_dir,
        force=True,
        device="cpu",
        smoke=True,
        train_seed_offset=1000,
    )

    assert res0["model"] == "physdual_gin_esm_mlp"
    assert res1["model"] == "physdual_gin_esm_mlp_tsoff1000"
    assert res1["run_seed"] == res0["run_seed"] + 1000
    assert res1["diagnostic_control"]["train_seed_offset"] == 1000

    pairs0 = [(r["drug_id"], r["target_id"], r["y_true"]) for r in res0["oof_predictions"]]
    pairs1 = [(r["drug_id"], r["target_id"], r["y_true"]) for r in res1["oof_predictions"]]
    assert pairs0 == pairs1

    preds0 = [r["y_pred"] for r in res0["oof_predictions"]]
    preds1 = [r["y_pred"] for r in res1["oof_predictions"]]
    assert preds0 != preds1

    runs_dir = tmp_path / "outputs" / "results" / "runs"
    assert (runs_dir / "physdual_gin_esm_mlp_davis_cold_target_seqid30_fold0.json").exists()
    assert (runs_dir / "physdual_gin_esm_mlp_tsoff1000_davis_cold_target_seqid30_fold0.json").exists()
    assert json.loads((runs_dir / "physdual_gin_esm_mlp_davis_cold_target_seqid30_fold0.json").read_text())["model"] == "physdual_gin_esm_mlp"


def test_aggregate_only_supports_diagnostic_models_and_kiba_kfold_splits(tmp_path: Path, monkeypatch):
    from run_phase3_physdual import _resolve_requested_splits, main as p3_main, run_single_unit

    assert _resolve_requested_splits("kiba", "all", "kfold") == ["cold_target_seqid30", "cold_target_seqid40"]
    assert _resolve_requested_splits("davis", "all", "kfold") == [
        "cold_drug",
        "cold_target",
        "cold_target_seqid30",
        "cold_target_seqid40",
    ]
    assert _resolve_requested_splits("kiba", "all", "seeds") == ["random", "cold_drug", "cold_target"]

    monkeypatch.setenv("PHYSDUAL_ROOT", str(tmp_path))
    cfg, esm_dir = _create_synthetic_phase3_workspace(tmp_path)
    for fold in (0, 1):
        run_single_unit(
            dataset_name="davis",
            split_type="cold_target_seqid30",
            protocol="kfold",
            unit_id=fold,
            config=cfg,
            root=tmp_path,
            esm_cache_dir=esm_dir,
            force=True,
            device="cpu",
            smoke=True,
            model_name="ecfp_mlp",
            train_seed_offset=1000,
        )

    agg_path = (
        tmp_path
        / "outputs"
        / "results"
        / "phase3_physdual_ecfp_esm_mlp_tsoff1000_davis_cold_target_seqid30_kfold.json"
    )
    assert not agg_path.exists()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_phase3_physdual.py",
            "--aggregate-only",
            "--model",
            "ecfp_mlp",
            "--train-seed-offset",
            "1000",
            "--root",
            str(tmp_path),
        ],
    )
    p3_main()
    assert agg_path.exists()
    agg_data = json.loads(agg_path.read_text(encoding="utf-8"))
    assert agg_data["model"] == "physdual_ecfp_esm_mlp_tsoff1000"
    assert agg_data["folds"] == [0, 1]
