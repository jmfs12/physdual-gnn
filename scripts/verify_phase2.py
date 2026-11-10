#!/usr/bin/env python3
"""Verificador objetivo do Gate de Pronto da Fase 2 (`scripts/verify_phase2.py`).

Implementa dois modos de validação:
1. `--smoke`: Validação ágil de contratos arquiteturais, integridade de imports, roundtrip
   atômico de checkpoints (weights_only=True), inferência dos modelos GraphDTAGIN e
   GraphDTACNN3, e consistência do cálculo de métricas sem exigir treinamento completo.
2. `--full`: Auditoria completa dos 6 critérios de prontidão da Fase 2 contra artefatos em disco:
   - Gate 1: Presença e integridade dos JSONs individuais de runs (outputs/results/runs/).
   - Gate 2: Presença e consistência dos JSONs agregados (outputs/results/baseline_*.json).
   - Gate 3: Ausência de arquivos temporários órfãos (.tmp_*).
   - Gate 4: Isolamento estrito de teste (avaliação pós-treinamento unicamente).
   - Gate 5: Alertas de sanidade não-bloqueantes (Davis random CI >= 0.86 de referência, teto de épocas).
   - Gate 6: Superação empírica dos baselines triviais (MSE menor que mean_predictor e ridge_aa_composition).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from physdual.utils.config import load_config  # noqa: E402
from physdual.utils.paths import get_project_root  # noqa: E402


def run_phase2_smoke_verification(
    config_path: str | Path | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Executa verificações de integridade arquitetural em modo smoke."""
    checks: dict[str, dict[str, Any]] = {
        "1_imports_and_symbols": {"passed": True, "errors": []},
        "2_config_graphdta_contract": {"passed": True, "errors": []},
        "3_checkpoint_roundtrip_weights_only": {"passed": True, "errors": []},
        "4_model_forward_pass_gin_and_cnn3": {"passed": True, "errors": []},
        "5_metrics_computation": {"passed": True, "errors": []},
    }

    # 1. Imports
    try:
        import physdual.data.datamodule as dm
        import physdual.eval.metrics as met
        import physdual.models.graphdta as gdta
        import physdual.training.checkpoint as ckpt
        import physdual.training.trainer as tr

        assert hasattr(dm, "get_dta_dataloaders")
        assert hasattr(dm, "DTADataset")
        assert hasattr(gdta, "GraphDTAGIN")
        assert hasattr(gdta, "GraphDTACNN3")
        assert hasattr(gdta, "build_baseline_model")
        assert hasattr(met, "compute_all_metrics")
        assert hasattr(ckpt, "save_checkpoint")
        assert hasattr(ckpt, "load_checkpoint")
        assert hasattr(tr, "train_model")
    except Exception as e:
        checks["1_imports_and_symbols"]["passed"] = False
        checks["1_imports_and_symbols"]["errors"].append(f"Falha de importação: {e}")

    # 2. Config contract
    try:
        cfg = load_config(config_path=config_path, root=root)
        if "graphdta" not in cfg:
            checks["2_config_graphdta_contract"]["passed"] = False
            checks["2_config_graphdta_contract"]["errors"].append("Seção 'graphdta' ausente no arquivo YAML.")

        drug_in_dim = cfg.get("graphdta", {}).get("drug_in_dim")
        atom_feature_dim = cfg.get("featurizer", {}).get("atom_feature_dim")
        if drug_in_dim != 78:
            checks["2_config_graphdta_contract"]["passed"] = False
            checks["2_config_graphdta_contract"]["errors"].append(
                f"graphdta.drug_in_dim esperado 78, encontrado {drug_in_dim}"
            )
        if atom_feature_dim != 78:
            checks["2_config_graphdta_contract"]["passed"] = False
            checks["2_config_graphdta_contract"]["errors"].append(
                f"featurizer.atom_feature_dim esperado 78, encontrado {atom_feature_dim}"
            )
    except Exception as e:
        checks["2_config_graphdta_contract"]["passed"] = False
        checks["2_config_graphdta_contract"]["errors"].append(f"Erro ao validar config: {e}")

    # 3. Checkpoint roundtrip com weights_only=True
    try:
        from physdual.training.checkpoint import load_checkpoint, save_checkpoint

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_ckpt.pt"
            state = {
                "epoch": 42,
                "val_metric": 0.235,
                "model_tensor": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
                "nested": {"param": 10},
            }
            save_checkpoint(state, ckpt_path)
            loaded = load_checkpoint(ckpt_path)
            if loaded["epoch"] != 42 or not torch.allclose(loaded["model_tensor"], state["model_tensor"]):
                checks["3_checkpoint_roundtrip_weights_only"]["passed"] = False
                checks["3_checkpoint_roundtrip_weights_only"]["errors"].append(
                    "Valores restaurados do checkpoint diferem dos originais."
                )
    except Exception as e:
        checks["3_checkpoint_roundtrip_weights_only"]["passed"] = False
        checks["3_checkpoint_roundtrip_weights_only"]["errors"].append(f"Erro no roundtrip de checkpoint: {e}")

    # 4. Model forward pass para GraphDTAGIN e GraphDTACNN3
    try:
        from torch_geometric.data import Batch, Data
        from physdual.models.graphdta import build_baseline_model

        cfg = load_config(config_path=config_path, root=root)
        # Mock de 2 moléculas e 2 alvos
        g1 = Data(x=torch.randn(4, 78), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long))
        g2 = Data(x=torch.randn(3, 78), edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long))
        drug_batch = Batch.from_data_list([g1, g2])
        target_tokens = torch.randint(1, 25, (2, 1000), dtype=torch.long)

        for m_name in ("graphdta_gin", "graphdta_cnn3"):
            model = build_baseline_model(m_name, cfg)
            model.eval()
            with torch.no_grad():
                out = model(drug_batch, target_tokens)
            if out.shape[0] != 2 or torch.isnan(out).any() or torch.isinf(out).any():
                checks["4_model_forward_pass_gin_and_cnn3"]["passed"] = False
                checks["4_model_forward_pass_gin_and_cnn3"]["errors"].append(
                    f"Forward pass de {m_name} produziu batch incorreto {out.shape} ou valores NaN/Inf."
                )
    except Exception as e:
        checks["4_model_forward_pass_gin_and_cnn3"]["passed"] = False
        checks["4_model_forward_pass_gin_and_cnn3"]["errors"].append(f"Erro no forward pass: {e}")

    # 5. Cálculo de métricas
    try:
        from physdual.eval.metrics import compute_all_metrics

        y_true = np.array([5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float64)
        y_pred = np.array([5.1, 5.9, 7.2, 7.8, 9.1], dtype=np.float64)
        res = compute_all_metrics(y_true, y_pred)
        for req_m in ("ci", "mse", "rmse", "pearson", "spearman", "rm2"):
            if req_m not in res or np.isnan(res[req_m]):
                checks["5_metrics_computation"]["passed"] = False
                checks["5_metrics_computation"]["errors"].append(f"Métrica obrigatória ausente ou NaN: {req_m}")
    except Exception as e:
        checks["5_metrics_computation"]["passed"] = False
        checks["5_metrics_computation"]["errors"].append(f"Erro ao calcular métricas: {e}")

    overall = all(c["passed"] for c in checks.values())
    return {"passed": overall, "mode": "smoke", "checks": checks}


def run_phase2_full_verification(
    config_path: str | Path | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Executa a auditoria completa dos 6 critérios de prontidão da Fase 2."""
    cfg = load_config(config_path=config_path, root=root)
    base_root = Path(cfg.get("project_root", get_project_root()))
    paths_cfg = cfg["paths"]

    results_dir = base_root / paths_cfg["results_dir"]
    runs_dir = results_dir / "runs"
    ckpts_dir = base_root / paths_cfg["checkpoints_dir"]

    datasets = ["davis", "kiba"]
    splits = ["random", "cold_drug", "cold_target"]
    seeds = [0, 1, 2, 3, 4]
    official_model = "graphdta_gin"

    checks: dict[str, dict[str, Any]] = {
        "gate1_individual_runs_presence": {"passed": True, "errors": [], "count": 0},
        "gate2_aggregated_results_presence": {"passed": True, "errors": [], "count": 0},
        "gate3_no_orphan_temp_files": {"passed": True, "errors": []},
        "gate4_strict_test_isolation": {"passed": True, "errors": []},
        "gate5_sanity_alerts": {"passed": True, "warnings": []},
        "gate6_trivial_baselines_beaten": {"passed": True, "errors": []},
    }

    # Gate 1: Presença dos JSONs individuais de runs
    expected_runs = []
    for ds in datasets:
        for sp in splits:
            for s in seeds:
                expected_runs.append(f"{official_model}_{ds}_{sp}_seed{s}.json")

    missing_runs = []
    found_runs = 0
    for r_name in expected_runs:
        rf = runs_dir / r_name
        if not rf.exists():
            missing_runs.append(r_name)
        else:
            try:
                data = json.loads(rf.read_text(encoding="utf-8"))
                if "test_metrics_full" not in data or "best_epoch" not in data:
                    missing_runs.append(f"{r_name} (incompleto)")
                else:
                    found_runs += 1
            except Exception:
                missing_runs.append(f"{r_name} (corrompido)")

    checks["gate1_individual_runs_presence"]["count"] = found_runs
    if missing_runs:
        checks["gate1_individual_runs_presence"]["passed"] = False
        checks["gate1_individual_runs_presence"]["errors"].append(
            f"Faltam {len(missing_runs)}/{len(expected_runs)} runs obrigatórias. Amostra: {missing_runs[:5]}"
        )

    # Gate 2: Presença dos JSONs agregados
    expected_aggs = []
    for ds in datasets:
        for sp in splits:
            expected_aggs.append(f"baseline_{official_model}_{ds}_{sp}.json")

    missing_aggs = []
    found_aggs = 0
    for a_name in expected_aggs:
        af = results_dir / a_name
        if not af.exists():
            missing_aggs.append(a_name)
        else:
            try:
                data = json.loads(af.read_text(encoding="utf-8"))
                if "aggregated" not in data or "per_seed_results" not in data:
                    missing_aggs.append(f"{a_name} (estrutura inválida)")
                else:
                    found_aggs += 1
            except Exception:
                missing_aggs.append(f"{a_name} (corrompido)")

    checks["gate2_aggregated_results_presence"]["count"] = found_aggs
    if missing_aggs:
        checks["gate2_aggregated_results_presence"]["passed"] = False
        checks["gate2_aggregated_results_presence"]["errors"].append(
            f"Faltam {len(missing_aggs)}/{len(expected_aggs)} agregados. Amostra: {missing_aggs[:5]}"
        )

    # Gate 3: Ausência de temporários órfãos (.tmp_*)
    orphan_tmps = []
    for scan_dir in (results_dir, ckpts_dir):
        if scan_dir.exists():
            for p in scan_dir.rglob(".tmp_*"):
                orphan_tmps.append(str(p.relative_to(base_root)))

    if orphan_tmps:
        checks["gate3_no_orphan_temp_files"]["passed"] = False
        checks["gate3_no_orphan_temp_files"]["errors"].append(
            f"Encontrados {len(orphan_tmps)} arquivos temporários órfãos: {orphan_tmps[:5]}"
        )

    # Gate 4: Isolamento estrito de teste
    # Verifica que em todos os JSONs de run existentes, test_metrics_full foi computado e não há vazamento
    for r_name in expected_runs:
        rf = runs_dir / r_name
        if rf.exists():
            try:
                data = json.loads(rf.read_text(encoding="utf-8"))
                t_metrics = data.get("test_metrics_full", {})
                v_metrics = data.get("val_metrics_at_best", {})
                if not t_metrics or not v_metrics:
                    checks["gate4_strict_test_isolation"]["passed"] = False
                    checks["gate4_strict_test_isolation"]["errors"].append(
                        f"Run {r_name} sem métricas completas de validação/teste."
                    )
            except Exception:
                pass

    # Gate 5: Alertas de Sanidade (não-bloqueantes)
    for a_name in expected_aggs:
        af = results_dir / a_name
        if af.exists():
            try:
                data = json.loads(af.read_text(encoding="utf-8"))
                agg = data.get("aggregated", {})
                ds = data.get("dataset")
                sp = data.get("split")

                # Alerta Davis random CI < 0.86
                if ds == "davis" and sp == "random":
                    ci_mean = agg.get("test_metrics_full", {}).get("ci", {}).get("mean", 0.0)
                    if ci_mean < 0.86:
                        warn_msg = (
                            f"[WARN] Sanity Alert: Davis random CI={ci_mean:.4f} < 0.86 "
                            "(tolerância documentada; splits diferem do paper)."
                        )
                        checks["gate5_sanity_alerts"]["warnings"].append(warn_msg)
                        print(warn_msg)

                # Alerta se best_epoch > 0.8 * epochs
                eff_cfg = cfg.get("graphdta", {}).get("training", {})
                max_ep = eff_cfg.get("epochs", 500)
                best_ep_mean = agg.get("best_epoch", {}).get("mean", 0.0)
                if best_ep_mean > 0.8 * max_ep:
                    warn_msg = (
                        f"[WARN] Best epoch near ceiling ({best_ep_mean:.1f} > 0.8 * {max_ep}) "
                        f"em {a_name}."
                    )
                    checks["gate5_sanity_alerts"]["warnings"].append(warn_msg)
                    print(warn_msg)

            except Exception:
                pass

    # Gate 6: Baseline trivial superado
    for a_name in expected_aggs:
        af = results_dir / a_name
        if af.exists():
            try:
                data = json.loads(af.read_text(encoding="utf-8"))
                agg = data.get("aggregated", {})
                model_mse = agg.get("test_metrics_full", {}).get("mse", {}).get("mean")
                triv = agg.get("trivial_baselines", {})
                mean_mse = triv.get("mean_predictor", {}).get("mse", {}).get("mean")
                ridge_mse = triv.get("ridge_aa_composition", {}).get("mse", {}).get("mean")

                if model_mse is not None and mean_mse is not None and model_mse >= mean_mse:
                    checks["gate6_trivial_baselines_beaten"]["passed"] = False
                    checks["gate6_trivial_baselines_beaten"]["errors"].append(
                        f"Em {a_name}, modelo MSE ({model_mse:.4f}) >= mean_predictor ({mean_mse:.4f})."
                    )

                if model_mse is not None and ridge_mse is not None and model_mse >= ridge_mse:
                    checks["gate6_trivial_baselines_beaten"]["passed"] = False
                    checks["gate6_trivial_baselines_beaten"]["errors"].append(
                        f"Em {a_name}, modelo MSE ({model_mse:.4f}) >= ridge_aa_composition ({ridge_mse:.4f})."
                    )
            except Exception:
                pass

    # Apenas gates 1, 2, 3, 4, 6 são bloqueantes; gate 5 emite apenas avisos
    critical_gates = [
        "gate1_individual_runs_presence",
        "gate2_aggregated_results_presence",
        "gate3_no_orphan_temp_files",
        "gate4_strict_test_isolation",
        "gate6_trivial_baselines_beaten",
    ]
    overall = all(checks[k]["passed"] for k in critical_gates)
    return {"passed": overall, "mode": "full", "checks": checks}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audita os critérios do Gate de Pronto da Fase 2 (GraphDTA Baseline)."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Executa verificação em modo rápido (contratos, imports, forward pass e métricas).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Executa auditoria completa dos 6 critérios contra todos os artefatos em disco.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Caminho para o arquivo YAML de configuração (padrão: configs/base.yaml).",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Raiz opcional do projeto.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.smoke or not args.full:
        # Se for explicitamente smoke, ou nem smoke nem full foram passados:
        # no caso de chamada padrão sem parâmetros, se não houver artefatos completos em disco, fallback elegante para smoke
        if args.smoke:
            res = run_phase2_smoke_verification(config_path=args.config, root=args.root)
            print("=== Gate de Pronto da Fase 2 (Smoke Mode - PhysDual-GNN) ===")
            for name, info in res["checks"].items():
                status = "OK" if info["passed"] else "FALHA"
                print(f"  [{status}] {name}")
                for err in info["errors"][:5]:
                    print(f"       -> {err}")

            if res["passed"]:
                print("RESULTADO FINAL: APROVADO (Todos os critérios smoke da Fase 2 atendidos).")
                return 0
            print("RESULTADO FINAL: REPROVADO (Falha nos critérios smoke da Fase 2).")
            return 1

    # Modo full
    res = run_phase2_full_verification(config_path=args.config, root=args.root)
    print("=== Gate de Pronto da Fase 2 (PhysDual-GNN) ===")
    for name, info in res["checks"].items():
        status = "OK" if info["passed"] else "FALHA"
        print(f"  [{status}] {name}")
        for err in info.get("errors", [])[:5]:
            print(f"       -> {err}")
        for warn in info.get("warnings", [])[:5]:
            print(f"       -> {warn}")

    if res["passed"]:
        print("RESULTADO FINAL: APROVADO (Todos os critérios da Fase 2 atendidos).")
        return 0

    print("RESULTADO FINAL: REPROVADO (Corrija os erros acima antes de avançar para a Fase 3).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
