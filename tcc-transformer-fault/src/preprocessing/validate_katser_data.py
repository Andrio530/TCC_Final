"""
Validação da base Katser et al. (2022) para o módulo LSTM (Estágio 1).

Roda checagens estruturais e de plausibilidade e salva um relatório em
reports/metrics/katser_validation_report.json. Deve ser reexecutado sempre
que a base for atualizada ou antes de qualquer treino sério.
"""
import glob
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "external" / "katser_npp_transformers"
OUT_PATH = Path(__file__).resolve().parent.parent.parent / "reports" / "metrics" / "katser_validation_report.json"

GASES = ["H2", "CO", "C2H4", "C2H2"]


def checar_vazamento(fdd_train, fdd_test):
    overlap = set(fdd_train["id"]) & set(fdd_test["id"])
    return {"ids_em_comum_treino_teste": len(overlap)}


def checar_consistencia_labels(fdd_train, fdd_test, rul_train, rul_test):
    return {
        "fdd_rul_train_mesmos_ids": set(fdd_train["id"]) == set(rul_train["id"]),
        "fdd_rul_test_mesmos_ids": set(fdd_test["id"]) == set(rul_test["id"]),
    }


def checar_arquivos_vs_labels(fdd_train, fdd_test):
    files_train = set(p.name for p in (BASE_DIR / "data_train").glob("*.csv"))
    files_test = set(p.name for p in (BASE_DIR / "data_test").glob("*.csv"))
    return {
        "treino_arquivos_sem_label": len(files_train - set(fdd_train["id"])),
        "treino_labels_sem_arquivo": len(set(fdd_train["id"]) - files_train),
        "teste_arquivos_sem_label": len(files_test - set(fdd_test["id"])),
        "teste_labels_sem_arquivo": len(set(fdd_test["id"]) - files_test),
    }


def checar_duplicatas_exatas(all_files):
    hashes = {}
    dupes = 0
    for f in all_files:
        with open(f, "rb") as fh:
            h = hashlib.md5(fh.read()).hexdigest()
        if h in hashes:
            dupes += 1
        else:
            hashes[h] = f
    return {"duplicatas_exatas": dupes, "total_arquivos": len(all_files)}


def checar_quase_duplicatas(all_files, n_amostra=300, seed=42, limiar=0.999):
    random.seed(seed)
    sample_files = random.sample(all_files, min(n_amostra, len(all_files)))
    series_h2 = {f: pd.read_csv(f)["H2"].values for f in sample_files}
    keys = list(series_h2.keys())
    checados, altos = 0, 0
    for i in range(len(keys)):
        for j in range(i + 1, min(i + 30, len(keys))):
            c = np.corrcoef(series_h2[keys[i]], series_h2[keys[j]])[0, 1]
            checados += 1
            if c > limiar:
                altos += 1
    return {"pares_checados": checados, "pares_alta_correlacao": altos, "limiar": limiar}


def checar_separacao_classes(fdd_train):
    rows = []
    for _, row in fdd_train.iterrows():
        df = pd.read_csv(BASE_DIR / "data_train" / row["id"])
        inicio, fim = df.iloc[:10].mean(), df.iloc[-10:].mean()
        rows.append({"classe": row["category"], **{f"{g}_delta": fim[g] - inicio[g] for g in GASES}})
    res = pd.DataFrame(rows)
    return res.groupby("classe").mean(numeric_only=True).to_dict()


def main():
    fdd_train = pd.read_csv(BASE_DIR / "labels_fdd_train.csv")
    fdd_test = pd.read_csv(BASE_DIR / "labels_fdd_test.csv")
    rul_train = pd.read_csv(BASE_DIR / "labels_rul_train.csv")
    rul_test = pd.read_csv(BASE_DIR / "labels_rul_test.csv")
    all_files = [str(p) for p in (BASE_DIR / "data_train").glob("*.csv")] + \
                [str(p) for p in (BASE_DIR / "data_test").glob("*.csv")]

    report = {
        "vazamento_treino_teste": checar_vazamento(fdd_train, fdd_test),
        "consistencia_labels": checar_consistencia_labels(fdd_train, fdd_test, rul_train, rul_test),
        "arquivos_vs_labels": checar_arquivos_vs_labels(fdd_train, fdd_test),
        "duplicatas_exatas": checar_duplicatas_exatas(all_files),
        "quase_duplicatas": checar_quase_duplicatas(all_files),
        "separacao_classes_delta_gases": checar_separacao_classes(fdd_train),
        "distribuicao_classes_treino": fdd_train["category"].value_counts().to_dict(),
        "distribuicao_classes_teste": fdd_test["category"].value_counts().to_dict(),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)

    print(json.dumps(report, indent=2, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
