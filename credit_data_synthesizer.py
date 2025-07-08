"""Synthetic credit data generator for the **creditlab** project.

Principais recursos
-------------------
* **GroupProfile** parametriza cada grupo homogêneo (GH1…GHN) de risco.
* `default_group_profiles(n)` gera *n* perfis automaticamente, do mais arriscado (GH1)
  ao menos arriscado (GHn), com nomes consistentes.
* **CreditDataSynthesizer** cria:
  1. **snapshot** (safra 0) com N contratos por grupo.
  2. **panel** (evolução mensal) com lógica de atraso, cura, refinanciamento e
     renegociação.
  3. **trace** com o rastro de renegociações (id_antigo → id_novo).
* Gera +10 features relevantes além das já existentes.
* Inclui kernel‑stratificação (quantis) e possibilita choque macroeconômico.

Exemplo de uso com buckets personalizados::

    synth = CreditDataSynthesizer(
        group_profiles=default_group_profiles(4),
        buckets=[0,15,30,60,90,120,180,240,360],
    )

    # também é possível definir uma matriz BASE personalizada
    custom_base = np.eye(5)
    synth = CreditDataSynthesizer(
        group_profiles=default_group_profiles(4),
        base_matrix=custom_base,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd

DEFAULT_BUCKETS = [0, 15, 30, 60, 90]

# matriz base padrao 5x5 para transicoes (sev=0)
BASE_5x5 = np.array(
    [
        [0.6, 0.4, 0.0, 0.0, 0.0],
        [0.0925, 0.6275, 0.28, 0.0, 0.0],
        [0.0, 0.085, 0.655, 0.26, 0.0],
        [0.0, 0.0, 0.0775, 0.6825, 0.24],
        [0.0, 0.0, 0.0, 0.35, 0.65],
    ],
    dtype=float,
)

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

# =============================================================================
# GroupProfile
# =============================================================================

@dataclass
class GroupProfile:
    """Representa um grupo homogêneo (GH) com parâmetros de risco."""

    name: str
    pd_base: float  # PD 12m baseline (pior grupo ≈ 0.12, melhor ≈ 0.02)
    refin_prob: float  # probabilidade de aceitar refinanciamento quando elegível
    reneg_prob_exog: float  # prob mensal de renegociação exógena (fora trigger 90d)
    transition_matrix: np.ndarray | None = None  # matriz de transição

    def __post_init__(self) -> None:
        if self.transition_matrix is None:
            sev = (self.pd_base - 0.02) / (0.12 - 0.02)
            self.transition_matrix = generate_transition_matrix(
                DEFAULT_BUCKETS,
                sev,
                base_mat=BASE_5x5,
            )

    # distribuições de variáveis demográficas/financeiras serão geradas on‑the‑fly

    def _delay_probs(self, buckets: List[int]) -> np.ndarray:
        """Return probability vector for initial delay buckets."""
        base = [1.0, 0.4, 0.25, 0.1, 0.05]
        if len(buckets) > 5:
            base.extend([0.05] * (len(buckets) - 5))
        base = np.array(base[: len(buckets)], dtype=float)
        weight = (self.pd_base - 0.02) / (0.12 - 0.02)
        scale = 1 + weight
        scaled = base.copy()
        scaled[0] /= scale
        for i in range(2, len(buckets)):
            scaled[i] *= scale * (1.3 if i == 3 else 1.6 if i >= 4 else 1.0)
        return scaled / scaled.sum()

    # ---------------------------------------------------------------------
    # Amostragem de contratos
    # ---------------------------------------------------------------------

    def sample_contracts(
        self,
        n: int,
        rng: np.random.Generator,
        *,
        id_offset: int = 0,
        buckets: List[int] | None = None,
    ) -> pd.DataFrame:
        """Gera *n* contratos sintéticos para este grupo."""

        df = pd.DataFrame(index=np.arange(n))

        # Identificadores
        df["id_contrato"] = id_offset + np.arange(n, dtype="int64")
        df["id_cliente"] = rng.integers(1e9, 2e9, size=n, dtype="int64")
        df["grupo_homogeneo"] = self.name

        # Datas
        df["data_inicio_contrato"] = pd.Timestamp("today").normalize() - pd.to_timedelta(
            rng.integers(0, 365, n), unit="D"
        )
        df["data_ref"] = pd.Timestamp("today").normalize()
        df["safra"] = df["data_ref"].dt.strftime("%Y%m")

        # Variáveis principais
        if buckets is None:
            buckets = DEFAULT_BUCKETS
        df["dias_atraso"] = rng.choice(
            buckets, size=n, p=self._delay_probs(buckets)
        ).astype("int16")
        df["nivel_refinanciamento"] = np.zeros(n, dtype="int8")

        # -----------------------------------------------------------------
        # Demográficas (estáticas)
        # -----------------------------------------------------------------
        df["idade_cliente"] = rng.integers(18, 75, size=n, dtype="int8")
        df["sexo"] = rng.choice(["M", "F", "N"], size=n, p=[0.48, 0.48, 0.04])
        df["renda_mensal"] = (
            rng.normal(4000 * (1 + 0.3 * (1 - self.pd_base / 0.12)), 1500, size=n)
            .clip(500, 50_000)
            .astype("float32")
        )
        df["tempo_no_endereco_anual"] = rng.exponential(scale=5, size=n).clip(0, 40)
        df["tempo_no_emprego_anual"] = rng.exponential(scale=4, size=n).clip(0, 40)
        df["num_dependentes"] = rng.integers(0, 5, size=n, dtype="int8")
        df["tipo_residencia"] = rng.choice(
            ["propria", "alugada", "financiada", "familiar"], size=n
        )

        # Bureau / comportamento de crédito
        df["score_bureau_externo"] = rng.integers(300, 1000, size=n, dtype="int16")
        df["qtd_consultas_bureau_3m"] = rng.poisson(
            lam=2 + 6 * (self.pd_base / 0.12), size=n
        ).astype("int8")

        # Variáveis contratuais adicionais
        df["dia_vencimento_parcela"] = rng.integers(1, 29, size=n, dtype="int8")
        df["tipo_garantia"] = rng.choice(
            ["sem", "veiculo", "imovel", "avalista"], p=[0.7, 0.15, 0.1, 0.05], size=n
        )
        df["ltv_inicial_pct"] = rng.uniform(40, 100, size=n).astype("float32")
        df["renda_liquida_disp_pct"] = rng.uniform(0.1, 0.8, size=n).astype("float32")

        # -----------------------------------------------------------------
        # Variáveis financeiras adicionais para completar 30+ colunas
        # -----------------------------------------------------------------
        df["prazo_meses"] = rng.integers(6, 60, size=n, dtype="int16")
        df["taxa_juros_anual_pct"] = rng.uniform(10, 40, size=n).astype("float32")
        df["valor_liberado"] = rng.uniform(2000, 50_000, size=n).astype("float32")
        df["saldo_devedor"] = df["valor_liberado"].astype("float32")
        df["valor_parcela"] = (
            df["saldo_devedor"] * (1 + df["taxa_juros_anual_pct"] / 100) / df["prazo_meses"]
        ).astype("float32")
        df["num_parcelas_pagas"] = rng.integers(0, 3, size=n, dtype="int16")
        df["data_ult_pgt"] = (
            df["data_inicio_contrato"] + pd.to_timedelta(df["num_parcelas_pagas"] * 30, unit="D")
        )
        df["num_parcelas_pagas_consecutivas"] = df["num_parcelas_pagas"].astype("int16")
        df["score_latente"] = rng.normal(0, 1, size=n).astype("float32")
        df["qtd_renegociacoes"] = np.zeros(n, dtype="int16")
        df["streak_90"] = np.where(df["dias_atraso"] == 90, 1, 0).astype("int8")
        df["ever90m12"] = np.zeros(n, dtype="int8")
        df["over90m12"] = np.zeros(n, dtype="int8")
        df["ever360m18"] = np.zeros(n, dtype="int8")
        df["flag_cura"] = np.zeros(n, dtype="int8")
        df["write_off"] = np.zeros(n, dtype="int8")

        return df

# =============================================================================
# Helpers
# =============================================================================

def bucket_index(delay: int, buckets: List[int]) -> int:
    """Return index of largest bucket <= delay."""
    idx = np.searchsorted(buckets, delay, side="right") - 1
    if idx < 0:
        return 0
    if idx >= len(buckets):
        return len(buckets) - 1
    return int(idx)


def _risk_score(mat: np.ndarray, buckets: List[int]) -> float:
    """Return probability de sair do bucket 30 para atrasos >=60."""
    try:
        start = buckets.index(30)
    except ValueError:
        start = 0
    try:
        thr = next(i for i, b in enumerate(buckets) if b >= 60)
    except StopIteration:
        return 0.0
    return float(mat[start, thr:].sum())


def generate_transition_matrix(
    buckets: List[int],
    sev: float,
    *,
    base_mat: np.ndarray | None = None,
    alpha: float = 0.6,
    beta: float = 0.5,
) -> np.ndarray:
    """Return transition matrix ajustada pela severidade."""

    n = len(buckets)
    if n < 5:
        raise ValueError("n_buckets must be >= 5")

    if base_mat is not None:
        mat = np.asarray(base_mat, dtype=float).copy()
        if mat.shape != (n, n):
            raise ValueError("base_mat shape must match number of buckets")
    else:
        if n == 5:
            mat = BASE_5x5.copy()
        else:
            # gera base generica para qualquer tamanho
            stay = np.array([0.6 + 0.05 * i / (n - 1) for i in range(n)], dtype=float)
            up = np.array([0.3 - 0.02 * i for i in range(n)], dtype=float)
            down = np.array([0.1 - 0.03 * i / (n - 1) for i in range(n)], dtype=float)
            mat = np.zeros((n, n), dtype=float)
            for i in range(n):
                if i == 0:
                    mat[i, i] = stay[i]
                    mat[i, i + 1] = 1 - mat[i, i]
                elif i == n - 1:
                    mat[i, i] = stay[i]
                    mat[i, i - 1] = 1 - mat[i, i]
                else:
                    mat[i, i] = stay[i]
                    mat[i, i + 1] = up[i]
                    mat[i, i - 1] = down[i]
                    s = mat[i].sum()
                    if s != 1:
                        mat[i, i] += 1 - s

    # aplica severidade --------------------------------------------------
    # linha 0: reduz probabilidade de permanencia
    mat[0, 0] *= max(0.0, 1 - beta * sev)
    mat[0] /= mat[0].sum()

    for i in range(1, n):
        diag = mat[i, i]
        off = mat[i].copy()
        off[i] = 0
        off *= 1 + alpha * sev
        mat[i] = off
        mat[i, i] = diag
        mat[i] /= mat[i].sum()

    # garantias extras ---------------------------------------------------
    for i in range(n):
        if mat[i, i] < 0.01:
            mat[i, i] = 0.01
        mat[i] /= mat[i].sum()

    # prob max de inadimplencia direta para ultimo bucket
    last_idx = n - 1
    if mat[0, last_idx] > 0.05:
        excess = mat[0, last_idx] - 0.05
        mat[0, last_idx] = 0.05
        other = mat[0, :last_idx]
        mat[0, :last_idx] = other / other.sum() * (1 - 0.05)

    # normalizacao final
    mat /= mat.sum(axis=1, keepdims=True)
    return mat

def default_group_profiles(n_groups: int) -> List[GroupProfile]:
    """Cria *n_groups* perfis padrão numerados GH1…GHn.

    GH1 é o mais arriscado (PD ~ 12 %), GHn o menos arriscado (PD ~ 2 %).
    """
    worst_pd, best_pd = 0.12, 0.02
    profiles: List[GroupProfile] = []
    for i in range(n_groups):
        interp = i / (n_groups - 1) if n_groups > 1 else 0  # 0…1
        pd_base = worst_pd - interp * (worst_pd - best_pd)
        refin_prob = 0.15 + interp * 0.15  # bons clientes refinanciam mais
        reneg_prob = 0.15 - interp * 0.10  # piores renegociam mais
        profiles.append(
            GroupProfile(
                name=f"GH{i+1}",
                pd_base=pd_base,
                refin_prob=refin_prob,
                reneg_prob_exog=reneg_prob,
            )
        )
    return profiles

# =============================================================================
# CreditDataSynthesizer
# =============================================================================

class CreditDataSynthesizer:
    """Gera snapshot, painel e rastro de renegociação.

    Parameters
    ----------
    buckets : list[int], optional
        Lista global de buckets de atraso. Se ``None``, usa ``[0,15,30,60,90]``.
    writeoff_bucket : int, optional
        Se definido, marca ``write_off`` quando ``dias_atraso`` ≥ esse valor.
    base_matrix : np.ndarray, optional
        Matriz BASE para gerar transições dos grupos. Deve ter shape
        ``(len(buckets), len(buckets))``.
    """

    def __init__(
        self,
        *,
        group_profiles: List[GroupProfile],
        contracts_per_group: int = 10_000,
        n_safras: int = 24,
        random_seed: int = 42,
        kernel_trick: bool = True,
        start_safra: str | pd.Timestamp | int | None = None,
        buckets: List[int] | None = None,
        writeoff_bucket: int | None = None,
        base_matrix: np.ndarray | None = None,
    ) -> None:
        self.group_profiles = group_profiles
        self.contracts_per_group = contracts_per_group
        self.n_safras = n_safras
        self.kernel_trick = kernel_trick
        self.rng = np.random.default_rng(random_seed)
        self.buckets = sorted(buckets if buckets is not None else DEFAULT_BUCKETS)
        self.writeoff_bucket = writeoff_bucket
        self.base_matrix = base_matrix

        for gp in self.group_profiles:
            if gp.transition_matrix is None or gp.transition_matrix.shape[0] != len(self.buckets):
                sev = (gp.pd_base - 0.02) / (0.12 - 0.02)
                gp.transition_matrix = generate_transition_matrix(
                    self.buckets,
                    sev,
                    base_mat=self.base_matrix,
                )

        if start_safra is None:
            self.start_safra = pd.Timestamp("today").normalize().replace(day=1)
        else:
            if isinstance(start_safra, int):
                start_safra = str(start_safra)
            if isinstance(start_safra, str):
                self.start_safra = pd.to_datetime(start_safra, format="%Y%m")
            elif isinstance(start_safra, pd.Timestamp):
                self.start_safra = start_safra
            else:
                raise TypeError("start_safra must be str, int, Timestamp or None")
            self.start_safra = self.start_safra.normalize().replace(day=1)

        # DataFrames de saída
        self._snapshot: pd.DataFrame | None = None
        self._panel: pd.DataFrame | None = None
        self._trace: pd.DataFrame | None = None

    # ---------------------------------------------------------------------
    # API pública
    # ---------------------------------------------------------------------
    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Gera snapshot inicial, painel evolutivo e rastro de renegociação.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            DataFrames de snapshot, painel e trace, respectivamente.
        """

        self._generate_snapshot()
        self._generate_panel()
        self._compute_targets()
        return self._snapshot, self._panel, self._trace

    # ------------------------------------------------------------------
    # Passo 1: snapshot
    # ------------------------------------------------------------------
    def _generate_snapshot(self) -> None:
        records: List[pd.DataFrame] = []
        start_date = self.start_safra
        for g_idx, gp in enumerate(self.group_profiles):
            offset = g_idx * self.contracts_per_group
            df = gp.sample_contracts(
                self.contracts_per_group,
                rng=self.rng,
                id_offset=offset,
                buckets=self.buckets,
            )
            if self.writeoff_bucket is not None:
                df["write_off"] = 0
            df["data_ref"] = start_date
            df["safra"] = start_date.strftime("%Y%m")

            # Kernel‑trick: bucketing por quantis de renda × idade → subclusters
            if self.kernel_trick and {"renda_mensal", "idade_cliente"} <= set(df.columns):
                q_renda = pd.qcut(df["renda_mensal"], q=4, labels=False, duplicates="drop")
                q_idade = pd.qcut(df["idade_cliente"], q=4, labels=False, duplicates="drop")
                df["subcluster"] = (
                    (q_renda.astype("int8").to_numpy() * 4) + q_idade.astype("int8").to_numpy()
                ).astype("int8")

            records.append(df)

        self._snapshot = pd.concat(records, ignore_index=True, copy=False)

    # ------------------------------------------------------------------
    # Passo 2: evolução mensal
    # ------------------------------------------------------------------
    def _generate_panel(self) -> None:
        assert self._snapshot is not None

        current = self._snapshot.copy()
        start_date = current["data_ref"].iloc[0]
        self._id_counter = int(current["id_contrato"].max()) + 1
        traces: List[dict] = []
        records = [current.copy()]

        for m in range(1, self.n_safras):
            ref_date = start_date + pd.DateOffset(months=m)
            current = self._evolve_one_month(current, ref_date, traces)
            records.append(current.copy())

        self._panel = pd.concat(records, ignore_index=True)
        self._trace = pd.DataFrame(traces, columns=["id_antigo", "id_novo", "data_evento"])

    # ------------------------------------------------------------------
    def _evolve_one_month(
        self,
        df: pd.DataFrame,
        ref_date: pd.Timestamp,
        trace_records: List[dict],
    ) -> pd.DataFrame:
        df = df.copy()
        buckets = np.array(self.buckets, dtype=np.int16)
        n_buckets = len(buckets)
        for gp in self.group_profiles:
            mask = df["grupo_homogeneo"] == gp.name
            if not mask.any():
                continue
            sub = df.loc[mask]

            prev_delay = sub["dias_atraso"].to_numpy()
            delay_idx = np.array([bucket_index(d, list(buckets)) for d in prev_delay], dtype=np.int16)
            mat = gp.transition_matrix
            new_idx = np.fromiter(
                (self.rng.choice(n_buckets, p=mat[i]) for i in delay_idx), dtype=np.int16
            )
            new_delay = buckets[new_idx]

            # triggers
            prev_streak90 = sub["streak_90"].to_numpy()
            enter_60 = (prev_delay < 60) & (new_delay >= 60)
            trigger1 = enter_60 & (self.rng.random(len(sub)) < gp.reneg_prob_exog)
            streak_90 = np.where(new_delay == 90, prev_streak90 + 1, 0)
            trigger2 = (streak_90 >= 3) & (new_delay == 90)
            reneg_mask = trigger1 | trigger2

            # atualiza atraso e contadores
            sub["dias_atraso"] = new_delay
            sub["streak_90"] = streak_90
            if self.writeoff_bucket is not None:
                sub.loc[new_delay >= self.writeoff_bucket, "write_off"] = 1

            # pagamentos e contagem consecutiva
            on_time = new_delay == 0
            sub.loc[on_time, "saldo_devedor"] = (
                sub.loc[on_time, "saldo_devedor"] - sub.loc[on_time, "valor_parcela"]
            ).clip(lower=0)
            sub.loc[on_time, "num_parcelas_pagas"] += 1
            sub.loc[on_time, "data_ult_pgt"] = ref_date
            sub.loc[on_time, "num_parcelas_pagas_consecutivas"] += 1
            sub.loc[~on_time, "num_parcelas_pagas_consecutivas"] = 0

            # Refinanciamento
            refin_mask = (
                on_time
                & (sub["num_parcelas_pagas_consecutivas"] >= 4)
                & (self.rng.random(len(sub)) < gp.refin_prob)
            )
            sub.loc[refin_mask, "nivel_refinanciamento"] += 1
            sub.loc[refin_mask, "saldo_devedor"] *= 1.1
            sub.loc[refin_mask, "num_parcelas_pagas_consecutivas"] = 0

            # Renegociação
            for idx in sub.index[reneg_mask]:
                old_id = int(sub.at[idx, "id_contrato"])
                new_id = int(self._id_counter)
                self._id_counter += 1
                trace_records.append(
                    {"id_antigo": old_id, "id_novo": new_id, "data_evento": ref_date}
                )
                sub.at[idx, "id_contrato"] = new_id
                sub.at[idx, "nivel_refinanciamento"] = 0
                sub.at[idx, "dias_atraso"] = 0
                sub.at[idx, "streak_90"] = 0
                sub.at[idx, "data_inicio_contrato"] = ref_date
                sub.at[idx, "safra"] = ref_date.strftime("%Y%m")
                sub.at[idx, "num_parcelas_pagas"] = 0
                sub.at[idx, "data_ult_pgt"] = ref_date
                sub.at[idx, "num_parcelas_pagas_consecutivas"] = 0
                sub.at[idx, "qtd_renegociacoes"] += 1

            df.loc[mask] = sub

        df["data_ref"] = ref_date
        df["safra"] = ref_date.strftime("%Y%m")
        return df

    # ------------------------------------------------------------------
    def _compute_targets(self) -> None:
        assert self._panel is not None

        self._panel = self._panel.sort_values(["id_contrato", "data_ref"])  # type: ignore[assignment]
        ever = np.zeros(len(self._panel), dtype="int8")
        over = np.zeros(len(self._panel), dtype="int8")
        ever360 = np.zeros(len(self._panel), dtype="int8")
        cura = np.zeros(len(self._panel), dtype="int8")

        trace_map = {row.id_antigo: pd.Timestamp(row.data_evento) for row in self._trace.itertuples()}

        for cid, grp in self._panel.groupby("id_contrato"):
            delays = grp["dias_atraso"].to_numpy()
            dates = pd.to_datetime(grp["data_ref"]).to_numpy()
            idx = grp.index.to_numpy()
            event_date = trace_map.get(cid)
            for i in range(len(delays)):
                start = pd.Timestamp(dates[i])
                horizon_end_12 = start + pd.DateOffset(months=12)
                horizon_end_18 = start + pd.DateOffset(months=18)
                future_delays_12 = delays[i : i + 13]
                future_delays_18 = delays[i : i + 19]
                if (future_delays_12 >= 90).any() or (
                    event_date is not None and start < event_date <= horizon_end_12
                ):
                    ever[idx[i]] = 1

                idx90 = np.where(future_delays_12 >= 90)[0]
                if len(idx90) > 0:
                    first = idx90[0]
                    if (future_delays_12[first:] >= 30).all():
                        over[idx[i]] = 1

                if (future_delays_18 >= 360).any():
                    ever360[idx[i]] = 1

                if i > 0 and delays[i] < 90 and delays[i - 1] >= 90:
                    cura[idx[i]] = 1

        self._panel["ever90m12"] = ever
        self._panel["over90m12"] = over
        self._panel["ever360m18"] = ever360
        self._panel["flag_cura"] = cura

        start_date = self._panel["data_ref"].min()
        self._snapshot = (
            self._panel[self._panel["data_ref"] == start_date]
            .copy()
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Properties de conveniência
    # ------------------------------------------------------------------
    @property
    def snapshot(self) -> pd.DataFrame:
        if self._snapshot is None:
            raise RuntimeError("Call generate() first.")
        return self._snapshot

    @property
    def panel(self) -> pd.DataFrame:
        if self._panel is None:
            raise RuntimeError("Call generate() first.")
        return self._panel

    @property
    def trace(self) -> pd.DataFrame:
        if self._trace is None:
            raise RuntimeError("Call generate() first.")
        return self._trace


# ###############################################################################
# # Script usage guard
# ###############################################################################

# if __name__ == "__main__":
#     synth = CreditDataSynthesizer(default_group_profiles(), contracts_per_group=1_000, n_safras=12)
#     snap, panel, trace = synth.generate()
#     print(snap.head())
#     print(panel.head())
#     print(trace.head())
