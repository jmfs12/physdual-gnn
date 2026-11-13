#!/usr/bin/env python3
"""Verificador objetivo do Gate da Fase 3 (`scripts/verify_phase3.py`).

Modos:
1. `--smoke`: Validação ágil de contratos (`physdual_phase3` no YAML, forward pass do
   `PhysDualPhase3`, `DTADataset` com `esm_cache`, roundtrip `weights_only=True`).
2. `--full`: Auditoria dos artefatos produzidos em `outputs/results/`:
   - Gate 1: Presença e integridade dos runs individuais (`history`, `best_epoch`, `NaN/Inf=0`).
   - Gate 2: Presença dos agregados (`phase3_physdual_gin_esm_mlp_*.json`).
   - Gate 3: Ausência de arquivos temporários `.tmp_*`.
   - Gate 4: Isolamento estrito de teste (`test_isolation_verified: true`).
   - Gate 5: Diagnóstico anti-memorização (`best_epoch >= 15` em `cold_target`).
   - Gate 6: Superioridade pareada contra baselines (`drug_mean`, `knn_kmer`, `ridge_esm`, `graphdta_cnn3`),
             com opção `--no-require-paired-win` para inspeção diagnóstica.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from physdual.utils.config import load_config  # noqa: E402
from physdual.utils.paths import get_project_root  # noqa: E402


def run_phase3_smoke_verification(
    config_path: str | Path | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Executa validação smoke de contratos arquiteturais da Fase 3."""
    checks: dict[str, dict[str, Any]] = {
        "1_imports_and_symbols": {"passed": True, "errors": []},
        "2_config_phase3_contract": {"passed": True, "errors": []},
        "3_model_forward_2d_and_3d": {"passed": True, "errors": []},
        "4_checkpoint_roundtrip_weights_only": {"passed": True, "errors": []},
    }

    try:
        from physdual.data.datamodule import DTADataset, load_esm_mean_pool_dict, resolve_esm_cache_dir
        from physdual.models.physdual import ESMTargetEncoder, PhysDualPhase3, build_physdual_phase3_model

        assert callable(load_esm_mean_pool_dict)
        assert callable(resolve_esm_cache_dir)
        assert issubclass(ESMTargetEncoder, torch.nn.Module)
        assert issubclass(PhysDualPhase3, torch.nn.Module)
        assert callable(build_physdual_phase3_model)
        assert DTADataset is not None
    except Exception as exc:
        checks["1_imports_and_symbols"]["passed"] = False
        checks["1_imports_and_symbols"]["errors"].append(str(exc))

    try:
        cfg = load_config(config_path=config_path, root=root)
        p3 = cfg.get("physdual_phase3", {})
        if not p3:
            raise KeyError("Seção 'physdual_phase3' ausente em configs/base.yaml")
        if int(p3.get("drug_in_dim", -1)) != 78:
            raise ValueError(f"drug_in_dim esperado 78, obtido {p3.get('drug_in_dim')}")
        if int(p3.get("esm_dim", -1)) != 1280:
            raise ValueError(f"esm_dim esperado 1280, obtido {p3.get('esm_dim')}")
    except Exception as exc:
        checks["2_config_phase3_contract"]["passed"] = False
        checks["2_config_phase3_contract"]["errors"].append(str(exc))

    try:
        from torch_geometric.data import Batch, Data
        from physdual.models.physdual import build_physdual_phase3_model

        cfg = load_config(config_path=config_path, root=root)
        model = build_physdual_phase3_model(cfg)
        model.eval()
        g1 = Data(x=torch.randn(4, 78), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long))
        g2 = Data(x=torch.randn(3, 78), edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long))
        drug_batch = Batch.from_data_list([g1, g2])
        esm_2d = torch.randn(2, 1280)
        esm_3d = torch.randn(2, 5, 1280)
        mask_3d = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
        with torch.no_grad():
            o1 = model(drug_batch, esm_2d)
            o2 = model(drug_batch, esm_3d, target_esm_mask=mask_3d)
        if o1.shape != (2,) or o2.shape != (2,) or not torch.isfinite(o1).all() or not torch.isfinite(o2).all():
            raise ValueError(f"Shapes ou finitude inválidos no forward: {o1.shape}, {o2.shape}")
    except Exception as exc:
        checks["3_model_forward_2d_and_3d"]["passed"] = False
        checks["3_model_forward_2d_and_3d"]["errors"].append(str(exc))

    try:
        from physdual.training.checkpoint import load_checkpoint, save_checkpoint

        with tempfile.TemporaryDirectory() as tmp_dir:
            p = Path(tmp_dir) / "p3_ckpt.pt"
            save_checkpoint({"epoch": 7, "w": torch.ones(4)}, p)
            loaded = load_checkpoint(p)
            if int(loaded["epoch"]) != 7:
                raise ValueError("Checkpoint restaurado divergente.")
    except Exception as exc:
        checks["4_checkpoint_roundtrip_weights_only"]["passed"] = False
        checks["4_checkpoint_roundtrip_weights_only"]["errors"].append(str(exc))

    all_passed = all(v["passed"] for v in checks.values())
    return {"mode": "smoke", "all_checks_passed": all_passed, "checks": checks}


def run_phase3_full_verification(
    config_path: str | Path | None = None,
    root: str | Path | None = None,
    require_paired_win: bool = True,
) -> dict[str, Any]:
    """Audita todos os gates da Fase 3 contra os artefatos em disco."""
    cfg = load_config(config_path=config_path, root=root)
    base_root = Path(root) if root is not None else Path(cfg.get("project_root", get_project_root()))
    results_dir = base_root / cfg["paths"]["results_dir"]
    runs_dir = results_dir / "runs"

    gates: dict[str, dict[str, Any]] = {
        "gate1_runs_integrity": {"passed": True, "errors": [], "num_runs": 0},
        "gate2_aggregates_present": {"passed": True, "errors": [], "num_aggregates": 0},
        "gate3_no_orphan_tmp_files": {"passed": True, "errors": []},
        "gate4_strict_test_isolation": {"passed": True, "errors": []},
        "gate5_anti_memorization_warnings": {"passed": True, "warnings": []},
        "gate6_paired_superiority": {"passed": True, "errors": [], "warnings": []},
    }

    run_files = sorted(runs_dir.glob("physdual_gin_esm_mlp_*.json")) if runs_dir.exists() else []
    gates["gate1_runs_integrity"]["num_runs"] = len(run_files)
    if not run_files:
        gates["gate1_runs_integrity"]["passed"] = False
        gates["gate1_runs_integrity"]["errors"].append(f"Nenhum run da Fase 3 encontrado em {runs_dir}")

    for rf in run_files:
        data = json.loads(rf.read_text(encoding="utf-8"))
        for req in ("test_metrics_full", "val_metrics_at_best", "history", "best_epoch", "marginal_baselines", "knn_baselines"):
            if req not in data or not data[req]:
                gates["gate1_runs_integrity"]["passed"] = False
                gates["gate1_runs_integrity"]["errors"].append(f"{rf.name}: campo '{req}' ausente ou vazio.")
        if not bool(data.get("test_isolation_verified", False)):
            gates["gate4_strict_test_isolation"]["passed"] = False
            gates["gate4_strict_test_isolation"]["errors"].append(f"{rf.name}: test_isolation_verified != True")
        if "cold_target" in str(data.get("split", "")).lower() and int(data.get("best_epoch", 0)) < 15:
            gates["gate5_anti_memorization_warnings"]["warnings"].append(
                f"{rf.name}: best_epoch={data.get('best_epoch')} (< 15) indica memorização precoce."
            )

    agg_files = sorted(results_dir.glob("phase3_physdual_gin_esm_mlp_*.json")) if results_dir.exists() else []
    gates["gate2_aggregates_present"]["num_aggregates"] = len(agg_files)
    if not agg_files:
        gates["gate2_aggregates_present"]["passed"] = False
        gates["gate2_aggregates_present"]["errors"].append(
            f"Nenhum agregado 'phase3_physdual_gin_esm_mlp_*.json' em {results_dir}"
        )

    tmp_files = list(base_root.glob("**/.tmp_*")) if base_root.exists() else []
    if tmp_files:
        gates["gate3_no_orphan_tmp_files"]["passed"] = False
        gates["gate3_no_orphan_tmp_files"]["errors"] = [str(p) for p in tmp_files[:10]]

    req_baselines = ("drug_mean_predictor", "knn_kmer_target", "knn_esm_target", "ridge_esm", "graphdta_cnn3")
    for af in agg_files:
        agg = json.loads(af.read_text(encoding="utf-8"))
        if not bool(agg.get("test_isolation_verified", False)):
            gates["gate4_strict_test_isolation"]["passed"] = False
            gates["gate4_strict_test_isolation"]["errors"].append(f"{af.name}: test_isolation_verified != True")
        paired = agg.get("paired_vs_baselines", {})
        if "cold_target" in str(agg.get("split", "")).lower():
            for b in req_baselines:
                if b not in paired:
                    continue
                d_mse = float(paired[b]["delta_mse_mean"])
                d_ci = float(paired[b]["delta_ci_mean"])
                if d_mse >= 0.0 or d_ci <= 0.0:
                    msg = f"{af.name} vs {b}: ΔMSE={d_mse:+.4f}, ΔCI={d_ci:+.4f}"
                    if require_paired_win:
                        gates["gate6_paired_superiority"]["passed"] = False
                        gates["gate6_paired_superiority"]["errors"].append(msg)
                    else:
                        gates["gate6_paired_superiority"]["warnings"].append(msg)

    all_passed = all(g["passed"] for g in gates.values())
    return {"mode": "full", "all_gates_passed": all_passed, "gates": gates}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verificador oficial do Gate da Fase 3.")
    mode_grp = parser.add_mutually_exclusive_group(required=True)
    mode_grp.add_argument("--smoke", action="store_true")
    mode_grp.add_argument("--full", action="store_true")
    parser.add_argument("--no-require-paired-win", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--root", type=str, default=None)
    args = parser.parse_args()

    if args.smoke:
        report = run_phase3_smoke_verification(config_path=args.config, root=args.root)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        sys.exit(0 if report["all_checks_passed"] else 1)

    report = run_phase3_full_verification(
        config_path=args.config,
        root=args.root,
        require_paired_win=not args.no_require_paired_win,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    sys.exit(0 if report["all_gates_passed"] else 1)


if __name__ == "__main__":
    main()
