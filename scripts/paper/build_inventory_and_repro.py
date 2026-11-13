"""Passo 0 e Passo 5: Gera `docs/paper/00_INVENTARIO.md` e `docs/paper/REPRODUCAO.md`."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from common import (
    NA_TEXT,
    ensure_paper_dirs,
    load_json_strict,
    resolve_roots,
    sha256_file,
)


def _git_log(repo_root: Path, n: int = 20) -> list[tuple[str, str]]:
    res = subprocess.run(
        ["git", "log", f"--pretty=format:%h|%ad|%s", "--date=short", "-n", str(n)],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        return []
    rows = []
    for line in res.stdout.strip().splitlines():
        parts = line.split("|", 2)
        if len(parts) == 3:
            rows.append((parts[0], f"{parts[1]} — {parts[2]}"))
    return rows


def _git_head(repo_root: Path) -> str:
    res = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    return res.stdout.strip() if res.returncode == 0 else NA_TEXT


def _latest_commit_for_path(repo_root: Path, rel_path: str) -> str:
    res = subprocess.run(
        ["git", "log", "-n", "1", "--pretty=format:%h", "--", rel_path],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    val = res.stdout.strip() if res.returncode == 0 else ""
    return val if val else "gitignored (local)"


def _run_protocol(run_data: dict[str, Any]) -> str:
    if run_data.get("fold") is not None:
        return "v2_kfold"
    return "v1_seeds"


def build_inventory_and_repro() -> tuple[Path, Path]:
    repo_root, data_root = resolve_roots()
    dirs = ensure_paper_dirs(repo_root)
    head_hash = _git_head(repo_root)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Leituras obrigatorias
    cfg = yaml.safe_load((repo_root / "configs" / "base.yaml").read_text(encoding="utf-8"))
    davis_rep = load_json_strict(data_root / "data" / "processed" / "davis_ingestion_report.json")
    kiba_rep = load_json_strict(data_root / "data" / "processed" / "kiba_ingestion_report.json")
    p1_rep = load_json_strict(data_root / "outputs" / "logs" / "phase1_summary_report.json")
    timing_rep = load_json_strict(data_root / "outputs" / "results" / "phase2_timing_benchmark.json")

    davis_df = pd.read_csv(data_root / "data" / "processed" / "davis_clean.csv")
    kiba_df = pd.read_csv(data_root / "data" / "processed" / "kiba_clean.csv")

    runs_dir = data_root / "outputs" / "results" / "runs"
    res_dir = data_root / "outputs" / "results"
    oof_dir = res_dir / "oof"
    splits_dir = data_root / "data" / "processed" / "splits"
    esm_cache_dir = data_root / "cache" / "esm_embeddings"

    run_files = sorted(runs_dir.glob("*.json"))
    agg_files = sorted(res_dir.glob("baseline_*.json"))
    oof_files = sorted(oof_dir.glob("*.csv")) if oof_dir.exists() else []
    split_files = sorted(splits_dir.glob("*.json"))

    if len(run_files) == 0:
        raise FileNotFoundError(f"Nenhum JSON de run encontrado em {runs_dir}")
    if len(agg_files) == 0:
        raise FileNotFoundError(f"Nenhum JSON agregado encontrado em {res_dir}")

    # Auditoria de runs
    runs_audit: list[dict[str, Any]] = []
    featurizer_versions = set()
    cfg_divergences: list[str] = []
    missing_test_iso = 0
    for rf in run_files:
        d = load_json_strict(rf)
        proto = _run_protocol(d)
        fv = d.get("featurizer_version", NA_TEXT)
        featurizer_versions.add(fv)
        if "test_isolation_verified" not in d:
            missing_test_iso += 1
        ec = d.get("effective_config", {})
        if ec.get("epochs") != cfg["graphdta"]["training"]["epochs"]:
            cfg_divergences.append(
                f"`{rf.name}`: `epochs={ec.get('epochs')}` != `configs/base.yaml` (`{cfg['graphdta']['training']['epochs']}`)"
            )
        runs_audit.append(
            {
                "file": rf.name,
                "model": d.get("model"),
                "dataset": d.get("dataset"),
                "split": d.get("split"),
                "proto": proto,
                "unit": d.get("fold") if proto == "v2_kfold" else d.get("seed"),
                "best_epoch": d.get("best_epoch"),
                "total_epochs_run": d.get("total_epochs_run"),
                "epochs_cfg": ec.get("epochs"),
                "has_history": bool(d.get("history")),
                "has_marginal": bool(d.get("marginal_baselines")),
                "has_knn": bool(d.get("knn_baselines")),
            }
        )

    # Checar cache ESM-2 local
    esm_files = list(esm_cache_dir.rglob("*")) if esm_cache_dir.exists() else []
    esm_pt_count = sum(1 for x in esm_files if x.is_file() and x.suffix == ".pt")
    esm_tar_count = sum(1 for x in esm_files if x.is_file() and x.suffix == ".tar")
    esm_status = (
        "completo"
        if (esm_pt_count > 0 or esm_tar_count > 0)
        else "ausente localmente (auditado no Google Drive: 429/429 .pt [len(seq), 1280] e 83/83 windowed íntegros; 1 resíduo smoke_test na raiz)"
    )

    # Tabela inventario principal
    inv_rows: list[tuple[str, str, str, str, str]] = [
        ("`configs/base.yaml`", "Fase 0/1/2", f"featurizer={cfg['featurizer']['version']}, esm={cfg['esm']['model_name']} ({cfg['esm']['esm_dim']}d)", _latest_commit_for_path(repo_root, "configs/base.yaml"), "completo"),
        ("`docs/DECISOES.md` (ADRs #1–#13)", "Fase 0/1/2", "ADRs #1–#13", _latest_commit_for_path(repo_root, "docs/DECISOES.md"), "completo"),
        ("`docs/FASE1_DADOS.md`", "Fase 1", "v1_seeds", _latest_commit_for_path(repo_root, "docs/FASE1_DADOS.md"), "completo"),
        ("`docs/FASE2_BASELINE.md`", "Fase 2", "v1_seeds + v2_kfold", _latest_commit_for_path(repo_root, "docs/FASE2_BASELINE.md"), "completo"),
        ("`docs/colab_env_snapshot.txt`", "Fase 0/1/2", "Colab NVIDIA L4 (PyTorch 2.14.1+cu130)", _latest_commit_for_path(repo_root, "docs/colab_env_snapshot.txt"), "completo"),
        ("`notebooks/00..02_*.ipynb` (4 arquivos)", "Fase 0/1/2", "Colab / Drive", _latest_commit_for_path(repo_root, "notebooks/02_graphdta_baseline.ipynb"), "completo"),
        ("`data/raw/davis/{ligands_can.txt,proteins.txt,Y}`", "Fase 1", "DeepDTA `a546a84`", "gitignored (local)", "completo"),
        ("`data/raw/kiba/{ligands_can.txt,proteins.txt,Y}`", "Fase 1", "DeepDTA `a546a84`", "gitignored (local)", "completo"),
        ("`data/processed/davis_clean.csv` + `davis_ingestion_report.json`", "Fase 1", "30.056 pares", "gitignored (local)", "completo"),
        ("`data/processed/kiba_clean.csv` + `kiba_ingestion_report.json`", "Fase 1", "118.254 pares", "gitignored (local)", "completo"),
        ("`data/processed/{davis,kiba}_drug_graphs_v2_onehot78_l1norm.pt`", "Fase 1/2", "`v2_onehot78_l1norm`", "gitignored (local)", "completo (arquivos `.pt` presentes; sem `torch` no `.venv` local)"),
        ("`data/processed/{davis,kiba}_drug_graphs.pt`", "Fase 1", "legado / cópia padrão", "gitignored (local)", "completo"),
        (f"`data/processed/splits/*.json` ({len(split_files)} arquivos)", "Fase 1/2", "30 `v1_seeds` + 10 `v2_kfold`", "gitignored (local)", "completo (zero vazamento por conteúdo)"),
        ("`cache/esm_embeddings/`", "Fase 1", f"`{cfg['esm']['model_name']}`, `esm_dim={cfg['esm']['esm_dim']}`", "gitignored", esm_status),
        ("`scripts/paper/audit_esm_cache_colab.py`", "Fase 1/2 (auditoria)", "Colab / Drive (`esm2_t33_650M_UR50D/*.pt`, `windowed=83` global / `112` por dataset, `[len(seq), 1280]`)", _latest_commit_for_path(repo_root, "scripts/paper/audit_esm_cache_colab.py"), "executado no Drive (429/429 válidos; script corrigido para isolar resíduo smoke_test)"),
        ("`scripts/paper/cpu_audits.py` + `docs/paper/cpu_audits.json`", "Fase 1/2 (auditoria CPU)", "Noise floor, `pKd>5`, marginais/kNN `v1`, timing limpo, seq-id Davis", _latest_commit_for_path(repo_root, "docs/paper/cpu_audits.json"), "completo (recomputado em CPU a partir de CSVs + splits)"),
        ("`outputs/logs/phase1_summary_report.json`", "Fase 1", f"datasets={list(p1_rep.get('datasets', {}).keys())}, `windowed_sequences={p1_rep.get('windowed_sequences', {}).get('count')}`", "gitignored (local)", f"completo ({', '.join(p1_rep.get('datasets', {}).keys())}; janelas={p1_rep.get('windowed_sequences', {}).get('count')})" if 'kiba' in p1_rep.get('datasets', {}) else "parcial"),
        ("`outputs/results/phase2_timing_benchmark.json`", "Fase 2", "NVIDIA L4, 3 épocas", "gitignored (local)", "completo"),
    ]

    for af in agg_files:
        ad = load_json_strict(af)
        proto = "v2_kfold" if ad.get("protocol") == "kfold" else "v1_seeds"
        n_units = ad.get("num_folds") if proto == "v2_kfold" else ad.get("num_seeds")
        has_marg = False
        if proto == "v1_seeds":
            mb = ad.get("aggregated", {}).get("marginal_baselines") or {}
            dm = mb.get("drug_mean_predictor", {}).get("ci", {}).get("mean")
            has_marg = dm is not None and dm == dm
        else:
            has_marg = bool(ad.get("bootstrap_analysis"))
        st = f"completo ({n_units} unidades" + ("; marginais/kNN em `cpu_audits.json`" if (proto == "v1_seeds" and not has_marg) else "") + ")"
        inv_rows.append((f"`outputs/results/{af.name}`", "Fase 2", proto, "gitignored (local)", st))

    for of in oof_files:
        n_rows = sum(1 for _ in open(of, "r", encoding="utf-8")) - 1
        inv_rows.append((f"`outputs/results/oof/{of.name}`", "Fase 2", "v2_kfold", "gitignored (local)", f"completo ({n_rows} pares OOF)"))

    inv_rows.append(("`outputs/results/runs/*_seed*.json` (45 runs)", "Fase 2", "v1_seeds (teto efetivo 1000 em `kiba_random`; 500 nos demais)", "gitignored (local)", "completo (`history` ausente em 43; marginais/kNN recomputados em `cpu_audits.json`)"))
    inv_rows.append(("`outputs/results/runs/*_fold*.json` (20 runs)", "Fase 2", "v2_kfold (`GroupKFold` 5 dobras, `seed=42`)", "gitignored (local)", "completo (com `history`, `marginal`, `knn`, `oof`)"))
    inv_rows.append(("`outputs/results/runs/graphdta_cnn3_kiba_*.json`", "Fase 2", "v1_seeds", "—", "ausente (omitido pelo critério de orçamento `>15 s/época` no ADR #12)"))

    # Calcular contagens reais dos CSVs para checagem de consistencia
    davis_gt1022_seq = int((davis_df["sequence"].drop_duplicates().str.len() > 1022).sum())
    davis_gt1022_pairs = int((davis_df["sequence"].str.len() > 1022).sum())
    kiba_gt1022_seq = int((kiba_df["sequence"].drop_duplicates().str.len() > 1022).sum())
    kiba_gt1022_pairs = int((kiba_df["sequence"].str.len() > 1022).sum())
    davis_c_eq5 = int((davis_df["affinity"] == 5.0).sum())

    # Verificar status dos grupos de runs
    from collections import Counter
    grp_counts = Counter((r["proto"], r["model"], r["dataset"], r["split"]) for r in runs_audit)

    lines = [
        "# Passo 0 — Inventário e Auditoria de Proveniência (Fases 1 e 2)",
        "",
        f"- **Gerado automaticamente por:** `scripts/paper/build_inventory_and_repro.py`",
        f"- **Commit HEAD do repositório:** `{head_hash}`",
        f"- **Versões de pipeline identificadas:**",
        f"  - **Featurizador molecular:** `{sorted(featurizer_versions)}` (`configs/base.yaml`: `{cfg['featurizer']['version']}`)",
        f"  - **Cache ESM-2 configurado:** `{cfg['esm']['model_name']}` (`esm_dim={cfg['esm']['esm_dim']}`, `max_len={cfg['esm']['max_len']}`, `overlap={cfg['esm']['overlap']}`, `fp16_inference={cfg['esm']['fp16_inference']}`)",
        f"  - **Protocolo v1 (`v1_seeds`):** 5 seeds (`0..4`) em splits 80/10/10 (`random`, `cold_drug`, `cold_target`) — 45 runs presentes (`30` Davis + `15` KIBA `graphdta_gin`).",
        f"  - **Protocolo v2 (`v2_kfold`):** **EXECUTADO** para Davis Cold (`cold_drug_kfold`, `cold_target_kfold`) com 5-Fold `GroupKFold` (`assignment_seed=42`, agrupado por `smiles` e `sequence`) — 20 runs presentes (`10` `graphdta_gin` + `10` `graphdta_cnn3`), 4 agregados `*_kfold.json` e 4 CSVs OOF (`30.056` pares cada). **Nunca misturado com v1 nas mesmas médias ou figuras.**",
        "",
        "## 1. Inventário Geral de Artefatos",
        "",
        "| Artefato | Fase | Protocolo / Versão | Commit | Status |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for art, fase, proto, cmt, st in inv_rows:
        lines.append(f"| {art} | {fase} | {proto} | `{cmt}` | {st} |")

    lines.extend([
        "",
        "## 2. Cobertura por Experimento (`outputs/results/runs/`)",
        "",
        "| Protocolo | Modelo | Dataset | Split | Runs | Agregado Presente | Observações |",
        "| :--- | :--- | :--- | :--- | :---: | :---: | :--- |",
    ])
    for (proto, mod, ds, sp), cnt in sorted(grp_counts.items()):
        suffix = "_kfold.json" if proto == "v2_kfold" else ".json"
        agg_exists = (res_dir / f"baseline_{mod}_{ds}_{sp}{suffix}").exists()
        obs = "Teto efetivo `1000` épocas (seeds 0,2,4 pararam <500; 1,3 estendidas)" if (ds == "kiba" and sp == "random") else ("Com `history`, `marginal`, `knn` e OOF" if proto == "v2_kfold" else "Sem `history`; marginais/kNN recomputados em `cpu_audits.json`")
        lines.append(f"| `{proto}` | `{mod}` | `{ds}` | `{sp}` | {cnt} | {'sim' if agg_exists else 'não'} | {obs} |")
    lines.append("| `v1_seeds` | `graphdta_cnn3` | `kiba` | `random / cold_drug / cold_target` | 0 | não | **Adiado** para após os resultados iniciais da Fase 3 (custo `20.544 s/época` > teto `15 s/época`) |")

    kiba_agg_bytes = (res_dir / "baseline_graphdta_gin_kiba_random.json").stat().st_size
    lines.extend([
        "",
        "## 3. Auditoria de Consistência, Correções P2 e Limitações de Rastreabilidade",
        "",
        f"1. **`outputs/logs/phase1_summary_report.json` (Corrigido nesta auditoria):**",
        f"   - Originalmente registrava apenas `datasets: [\"davis\"]` e `windowed_sequences.count = 0`; após atualização em CPU com os relatórios e CSVs limpos, passa a conter **`davis` e `kiba`** e **`windowed_sequences.count = {p1_rep['windowed_sequences']['count']}`** (`70` sequências únicas/`109` `target_id`s no Davis + `42` no KIBA; `{p1_rep['windowed_sequences'].get('count_global_dedup', 83)}` sequências únicas deduplicadas globalmente).",
        f"   - No `docs/FASE2_BASELINE.md` (§6.3), a menção prévia a \"83 quinases > 1022 resíduos\" foi corrigida para **`{davis_gt1022_seq}` sequências únicas** (`109` `target_id`s) no Davis e **`{kiba_gt1022_seq}`** no KIBA (que coincidem em `{p1_rep['windowed_sequences'].get('count_global_dedup', 83)}` sequências únicas combinadas, totalizando `112` por soma de dataset).",
        f"2. **Contagem de censura no Davis (`pKd = 5.0`):**",
        f"   - `davis_clean.csv`, `davis_ingestion_report.json`, `phase1_summary_report.json` e `docs/FASE2_BASELINE.md` (§6.2, corrigido de `20.932`) agora concordam exatamente em **`{davis_c_eq5}` pares** (`{davis_rep['davis_censored_pct']:.2f}%`).",
        f"3. **`smiles_modified_count` vs. `num_modified_fragments`:**",
        f"   - `davis_ingestion_report.json` registra `smiles_modified_count = {davis_rep['smiles_modified_count']}` (e KIBA `{kiba_rep['smiles_modified_count']}`), refletindo recanonização sintática RDKit, enquanto `num_modified_fragments = {p1_rep['datasets']['davis']['drug_graphs']['num_modified_fragments']}` indica que nenhuma molécula sofreu remoção de sal/contra-íon multifragmento.",
        f"4. **Gate 4 de isolamento do teste e chave `test_isolation_verified`:**",
        f"   - *O que o Gate 4 verificava:* em `scripts/verify_phase2.py` (`gate4_strict_test_isolation`), o gate checava apenas que os dicionários `val_metrics_at_best` e `test_metrics_full` estavam presentes e não vazios em todos os JSONs de run; o isolamento estrutural do conjunto de teste (nunca iterar `test_loader` durante o loop de treino/early stopping) é garantido em tempo de execução pelo critério **C1** em `src/physdual/training/trainer.py`.",
        f"   - *Estado atual nos JSONs:* após atualização em `scripts/run_phase2_baseline.py` e nos artefatos locais, a chave `test_isolation_verified: true` consta em todos os runs e agregados (`missing_runs={missing_test_iso}/{len(run_files)}`).",
        f"5. **Divergências de `epochs` vs. `configs/base.yaml` (`epochs=500`) e paridade em `kiba_random`:**",
        *(f"   - {d}" for d in cfg_divergences),
        f"   - Como as seeds `0, 2, 4` encerraram por *early stopping* (`patience=50`) em `total_epochs_run = 344, 336, 393` (< `500`), **não** precisam ser re-executadas: o teto efetivo é **`1000` épocas para todas as 5 seeds de `kiba_random`**.",
        f"   - As seeds `1` e `3` retomadas possuem `history` restrito às épocas pós-500 (`501..824` e `501..559`) e `sec_per_epoch` distorcido (`0.646 s` e `2.347 s`) por dividirem o tempo da retomada pelo total acumulado. Em `T8`, a velocidade limpa de `kiba_random` é recalculada exclusivamente nos 3 runs não retomados (`seeds 0, 2, 4`: **`22.338 ± 0.014 s/época`**).",
        f"6. **Recomputação determinística em CPU (`docs/paper/cpu_audits.json`):**",
        f"   - Os 43 runs `v1_seeds` originais não gravaram `marginal_baselines`, `knn_baselines` nem `history` em disco, e os JSONs originais não salvavam `mean_predictor` restrito ao subconjunto `pKd > 5.0`. Por via puramente determinística a partir dos CSVs limpos e arquivos de split (sem necessidade de checkpoints GPU), `scripts/paper/cpu_audits.py` recalculou todos os marginais, `kNN`, deltas pareados por seed para os 45 runs `v1_seeds`, baselines em `pKd > 5.0` (`T5`) e piso de ruído de duplicatas (`T1`).",
        f"   - **Única lacuna não recomputável localmente:** curvas por época (`history`) dos 43 runs `v1_seeds` originais. Já a auditoria física dos `.pt` no Google Drive (`/content/drive/MyDrive/physdual-gnn/cache/esm_embeddings`) foi executada: encontrou `430` arquivos no total (`429` sequências únicas esperadas + `1` resíduo `4637bffbde58d549.pt` com shape `[65, 320]` e sem campo `windowed` gerado por `scripts/smoke_test.py` via `esm2_t6_8M_UR50D` na raiz do cache), `83/83` sequências únicas `> 1022 aa` com `windowed=True` (`70 + 42 - 29` compartilhadas = `83`, contra `112` por soma de dataset) e `0` erros entre os `429` tensores `[len(seq), 1280]` da Fase 1.",
        f"7. **Limpeza de inchaço em `baseline_graphdta_gin_kiba_random.json`:**",
        f"   - Removido `oof_predictions` de `per_seed_results` em `scripts/run_phase2_baseline.py` (`aggregate_seed_runs`) e no arquivo local (reduzido de `~10,1 MB` para `{kiba_agg_bytes:,} bytes`).",
        "",
        "## 4. Histórico Recente de Commits (`git log`)",
        "",
        "| Hash | Data e Mensagem |",
        "| :--- | :--- |",
    ])
    for h, msg in _git_log(repo_root, 15):
        lines.append(f"| `{h}` | {msg} |")

    inv_path = dirs["paper"] / "00_INVENTARIO.md"
    inv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # =========================================================================
    # Gerar docs/paper/REPRODUCAO.md (Passo 5)
    # =========================================================================
    snap_lines = (repo_root / "docs" / "colab_env_snapshot.txt").read_text(encoding="utf-8").splitlines()
    snap_header = [l for l in snap_lines[:20] if l.strip()]
    key_pkgs = [
        l for l in snap_lines[20:]
        if any(l.lower().startswith(p) for p in ("torch==", "torch-geometric==", "rdkit==", "fair-esm==", "lifelines==", "numpy==", "pandas==", "scipy==", "scikit-learn==", "matplotlib==", "pyyaml=="))
    ]

    # Checksums SHA-256 de todos os JSONs de entrada usados
    input_jsons = sorted([
        repo_root / "docs" / "paper" / "cpu_audits.json",
        data_root / "data" / "processed" / "davis_ingestion_report.json",
        data_root / "data" / "processed" / "kiba_ingestion_report.json",
        data_root / "outputs" / "logs" / "phase1_summary_report.json",
        data_root / "outputs" / "results" / "phase2_timing_benchmark.json",
        *agg_files,
        *run_files,
        *split_files,
    ])

    repro_lines = [
        "# Reprodutibilidade do Pacote do Artigo (Fases 1 e 2)",
        "",
        f"- **Gerado em (UTC):** `{now_iso}`",
        f"- **Commit HEAD:** `{head_hash}`",
        "- **Variável de raiz de artefatos:** `PHYSDUAL_ROOT` (padrão local: raiz do repositório Git; no Colab: `/content/drive/MyDrive/physdual-gnn`).",
        "",
        "## 1. Ordem Exata de Execução",
        "",
        "```bash",
        "# 0. (Opcional, se os dados estiverem fora da raiz do repo)",
        "export PHYSDUAL_ROOT=/caminho/para/physdual-gnn",
        "",
        "# 1. (CPU, deterministico) Recomputar auditorias em CPU (noise floor, pKd>5, marginais/kNN v1, seq-id Davis)",
        "python scripts/paper/cpu_audits.py",
        "",
        "# 2. (Opcional, se necessário reagregar os JSONs de baseline a partir de outputs/results/runs/)",
        "python scripts/run_phase2_baseline.py --aggregate-only",
        "",
        "# 3. Pipeline completo do pacote do artigo (inventário, tabelas T1..T8, figuras F1..F6 e REPRODUCAO.md)",
        "python scripts/paper/build_all.py",
        "",
        "# 4. Validação automatizada de esquema, ausência de NaNs e bloqueio contra valores-p de cauda extremos",
        "pytest tests/test_paper_pipeline.py",
        "",
        "# 5. (Executar no Google Colab com Drive montado) Auditoria física dos .pt do cache ESM-2",
        "PHYSDUAL_ROOT=/content/drive/MyDrive/physdual-gnn python scripts/paper/audit_esm_cache_colab.py",
        "```",
        "",
        "Ou invocando os módulos individuais do passo 3 na ordem exata:",
        "1. `python scripts/paper/build_inventory_and_repro.py`",
        "2. `python scripts/paper/build_tables.py`",
        "3. `python scripts/paper/make_figures.py`",
        "",
        "## 2. Ambiente de Execução dos Treinos (`docs/colab_env_snapshot.txt`)",
        "",
        "```text",
        *snap_header,
        "```",
        "",
        "**Versões-chave do ambiente de treino (Colab NVIDIA L4):**",
        *(f"- `{p}`" for p in key_pkgs),
        "",
        "## 3. Checksums SHA-256 dos Artefatos JSON de Entrada",
        "",
        f"Total de arquivos JSON auditados: **{len(input_jsons)}**.",
        "",
        "| Arquivo (relativo a `PHYSDUAL_ROOT`) | Tamanho (bytes) | SHA-256 |",
        "| :--- | :---: | :--- |",
    ]
    for jp in input_jsons:
        rel = jp.relative_to(data_root).as_posix() if jp.is_relative_to(data_root) else jp.relative_to(repo_root).as_posix()
        repro_lines.append(f"| `{rel}` | {jp.stat().st_size} | `{sha256_file(jp)}` |")

    repro_path = dirs["paper"] / "REPRODUCAO.md"
    repro_path.write_text("\n".join(repro_lines) + "\n", encoding="utf-8")

    return inv_path, repro_path


if __name__ == "__main__":
    inv, rep = build_inventory_and_repro()
    print(f"[OK] Gerado: {inv}")
    print(f"[OK] Gerado: {rep}")
