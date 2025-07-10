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
from itertools import count
import numpy as np
import pandas as pd

DEFAULT_BUCKETS = [0, 15, 30, 60, 90, 120, 180, 240, 360]

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

# matriz base padrao 9x9 para transicoes (sev=0)
BASE_9x9 = np.zeros((9, 9), dtype=float)
BASE_9x9[:5, :5] = BASE_5x5
BASE_9x9[0, 0] = 0.8
BASE_9x9[0, 1] = 0.2
for i in range(5, 9):
    BASE_9x9[i, i - 1] = 0.35
    BASE_9x9[i, i] = 0.65

import warnings
import logging
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

logger = logging.getLogger("creditlab").getChild("synth")
logger.setLevel(logging.INFO)

# =============================================================================
# GroupProfile
# =============================================================================

@dataclass
class GroupProfile:
    """Representa um grupo homogêneo (GH) com parâmetros de risco."""

    name: str
    pd_base: float  # PD 12m baseline (pior grupo ≈ 0.12, melhor ≈ 0.02)
    p_accept_refin: float = 0.5  # probabilidade de aceitar refin após elegibilidade
    reneg_prob_exog: float = 0.1  # prob mensal de renegociação exógena (fora trigger 90d)
    transition_matrix: np.ndarray | None = None  # matriz de transição
    num_clients: int | None = None  # numero de clientes distintos no grupo

    def __post_init__(self) -> None:
        if self.transition_matrix is None:
            sev = (self.pd_base - 0.02) / (0.12 - 0.02)
            base = BASE_5x5 if len(DEFAULT_BUCKETS) == 5 else (
                BASE_9x9 if len(DEFAULT_BUCKETS) == 9 else None
            )
            self.transition_matrix = generate_transition_matrix(
                DEFAULT_BUCKETS,
                sev,
                base_mat=base,
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
        ids: np.ndarray | None = None,
        buckets: List[int] | None = None,
        start_safra: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Gera *n* contratos sintéticos para este grupo."""

        df = pd.DataFrame(index=np.arange(n))

        # Identificadores
        if ids is None:
            df["id_contrato"] = id_offset + np.arange(n, dtype="int64")
        else:
            df["id_contrato"] = np.asarray(ids, dtype="int64")
        df["grupo_homogeneo"] = self.name

        if start_safra is None:
            ref_start = pd.Timestamp("today").normalize().replace(day=1)
        else:
            ref_start = start_safra
        # Datas
        df["data_inicio_contrato"] = ref_start - pd.to_timedelta(
            rng.integers(0, 180, n), unit="D"
        )
        df["data_ref"] = ref_start
        df["safra"] = df["data_ref"].dt.strftime("%Y%m")

        # Variáveis principais
        if buckets is None:
            buckets = DEFAULT_BUCKETS
        df["dias_atraso"] = rng.choice(
            buckets, size=n, p=self._delay_probs(buckets)
        ).astype("int16")
        df["nivel_refinanciamento"] = np.zeros(n, dtype="int8")

        # -----------------------------------------------------------------
        # Demográficas (estáticas) - preenchidas depois pelo sintetizador
        # -----------------------------------------------------------------
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

def random_contract_start(
    safra: pd.Timestamp,
    rng: np.random.Generator,
    *,
    max_lag_months: int = 6,
) -> pd.Timestamp:
    """Return random start date ``<= safra`` and within ``max_lag_months``."""
    delta_days = rng.integers(0, max_lag_months * 30 + 1)
    return safra - pd.to_timedelta(delta_days, unit="D")

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


def _apply_delay_jitter(
    arr: np.ndarray,
    buckets: list[int],
    rng: np.random.Generator,
    pct: float,
    min_days: int,
) -> np.ndarray:
    """Return array jittered around bucket values without crossing midpoints."""
    out = arr.astype(int)
    for i, b in enumerate(buckets):
        w = max(min_days, int(round(b * pct)))
        w = min(w, 8)
        lo = -min(w, b - (buckets[i - 1] if i > 0 else 0) // 2)
        hi = min(w, (buckets[i + 1] if i < len(buckets) - 1 else b + 10) - b)
        if i == len(buckets) - 1:
            lo = 0
        mask = arr == b
        if mask.any():
            out[mask] += rng.integers(lo, hi + 1, size=mask.sum())
    return out


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
        elif n == 9:
            mat = BASE_9x9.copy()
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

def default_group_profiles(
    n_groups: int,
    *,
    reneg_prob_line: np.ndarray | None = None,
    refin_prob: np.ndarray | None = None,
) -> List[GroupProfile]:
    """Cria *n_groups* perfis padrão numerados GH1…GHn.

    GH1 é o mais arriscado (PD ~ 12 %), GHn o menos arriscado (PD ~ 2 %).
    """
    worst_pd, best_pd = 0.12, 0.02
    if reneg_prob_line is None:
        reneg_prob_line = np.linspace(0.10, 0.015, n_groups)
    if refin_prob is None:
        refin_prob = np.linspace(0.05, 0.25, n_groups)
    profiles: List[GroupProfile] = []
    for i in range(n_groups):
        interp = i / (n_groups - 1) if n_groups > 1 else 0
        pd_base = worst_pd - interp * (worst_pd - best_pd)
        profiles.append(
            GroupProfile(
                name=f"GH{i+1}",
                pd_base=pd_base,
                p_accept_refin=float(refin_prob[i]),
                reneg_prob_exog=float(reneg_prob_line[i]),
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
        Lista global de buckets de atraso. Se ``None``, usa ``[0,15,30,60,90,120,180,240,360]``.
    writeoff_bucket : int, optional
        Se definido, marca ``write_off`` quando ``dias_atraso`` ≥ esse valor.
    base_matrix : np.ndarray, optional
        Matriz BASE para gerar transições dos grupos. Deve ter shape
        ``(len(buckets), len(buckets))``.
    force_event_rate : bool, default True
        Se True, aplica balanceamento de prevalência após gerar o painel.
    target_ratio : float, default 0.10
        Bad-rate desejado por safra (e por grupo se ``per_group=True``).
    sampler_kwargs : dict, optional
        Parâmetros extras repassados ao ``TargetSampler``.
    n_clients : int, optional
        Número total de clientes distintos na base gerada. Se ``None``, utiliza
        ``contracts_per_group * len(group_profiles) * 2``.
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
        force_event_rate: bool = True,
        target_ratio: float = 0.10,
        sampler_kwargs: dict | None = None,
        n_clients: int | None = None,
        new_contract_rate: float | List[float] | None = None,
        closure_rate: float | List[float] | None = None,
        jitter: bool = False,
        jitter_pct: float = 0.10,
        jitter_min: int = 2,
        max_duration_m: int = 72,
        p_existing_client: float = 0.7,
        id_start: int = 10_000_000,
        id_length: int = 8,
        sampler_strategy: str = "undersample",
        max_sampling_iter: int = 3,
        realloc_window: int = 3,
        tol_pp: float = 0.5,
        max_simultaneous_contracts: int = 3,
        max_recompute_iter: int = 2,
        min_volume: int = 30,
        verbose: bool = False,
    ) -> None:
        self.group_profiles = group_profiles
        self.contracts_per_group = contracts_per_group
        self.n_safras = n_safras
        self.kernel_trick = kernel_trick
        self.rng = np.random.default_rng(random_seed)
        self.buckets = sorted(buckets if buckets is not None else DEFAULT_BUCKETS)
        self.writeoff_bucket = writeoff_bucket
        self.base_matrix = base_matrix
        self.force_event_rate = force_event_rate
        self.target_ratio = target_ratio
        self._sampler_kwargs = sampler_kwargs or {}
        self._sampler_kwargs.setdefault("tol_pp", tol_pp)
        n_groups = len(group_profiles)
        pd_vals = np.array([gp.pd_base for gp in group_profiles], dtype=float)
        if new_contract_rate is None:
            self.new_contract_rate = np.interp(pd_vals, (pd_vals.min(), pd_vals.max()), (0.06, 0.03))
        else:
            arr = np.asarray(new_contract_rate, dtype=float)
            self.new_contract_rate = np.full(n_groups, arr) if arr.ndim == 0 else arr
        if closure_rate is None:
            self.closure_rate = np.interp(pd_vals, (pd_vals.min(), pd_vals.max()), (0.03, 0.06))
        else:
            arr = np.asarray(closure_rate, dtype=float)
            self.closure_rate = np.full(n_groups, arr) if arr.ndim == 0 else arr
        self.jitter = jitter
        self.jitter_pct = jitter_pct
        self.jitter_min = jitter_min
        self.max_duration_m = max_duration_m
        self.p_existing_client = p_existing_client
        self.id_start = id_start
        self.id_length = id_length
        self._id_pool = count(id_start)
        self.sampler_strategy = sampler_strategy
        self.max_sampling_iter = max_sampling_iter
        self.realloc_window = realloc_window
        self.max_simultaneous_contracts = max_simultaneous_contracts
        self.max_recompute_iter = max_recompute_iter
        self.min_volume = min_volume
        self.verbose = verbose
        self.logger = logger.getChild(self.__class__.__name__)

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

        # registro global de clientes ------------------------------------
        if n_clients is None:
            n_clients = contracts_per_group * len(group_profiles) * 2
        self._clients = self._build_clients(n_clients)
        self._start_pairs: set[tuple[int, str]] = set()

        # DataFrames de saída
        self._snapshot: pd.DataFrame | None = None
        self._panel: pd.DataFrame | None = None
        self._trace: pd.DataFrame | None = None
        self._closed: pd.DataFrame | None = None

        self._next_client_id = int(self._clients["id_cliente"].max()) + 1

    # ------------------------------------------------------------------
    def _log(self, msg: str, *args, lvl: int = logging.INFO) -> None:
        if self.verbose:
            self.logger.log(lvl, msg, *args)

    def _log_monthly_summary(self, ref_date: pd.Timestamp, noise: np.ndarray) -> None:
        if self.verbose and noise.size > 0:
            self.logger.info(
                "[month %s] jitter applied: mean(|noise|)=%.1f d, max=%d d",
                ref_date.strftime("%m"),
                float(np.abs(noise).mean()),
                int(np.abs(noise).max()),
            )

    def _log_rates(self, label: str, df: pd.DataFrame) -> None:
        if not self.verbose:
            return
        rates = df.groupby("grupo_homogeneo")["ever90m12"].mean()
        self._log("%s bad-rate %s", label, rates.round(3).to_dict())

    # ------------------------------------------------------------------
    def _build_clients(self, n_clients: int) -> pd.DataFrame:
        rng = self.rng
        # consume one random draw to keep RNG sequence comparable
        rng.integers(0, 1_000_000_000, size=n_clients)
        df = pd.DataFrame(
            {
                "id_cliente": np.arange(1, n_clients + 1, dtype="int64"),
                "sexo": rng.choice(["M", "F", "N"], size=n_clients, p=[0.48, 0.48, 0.04]),
                "data_nascimento": rng.choice(
                    pd.date_range("1940-01-01", "2005-12-31", freq="D"), size=n_clients
                ),
            }
        )
        return df

    # ------------------------------------------------------------------
    def _next_id(self) -> int:
        """Return next unique contract ID."""
        nxt = next(self._id_pool)
        if nxt >= 10 ** self.id_length:
            raise RuntimeError("ID pool exhausted")
        return nxt

    def _next_ids(self, n: int) -> np.ndarray:
        """Return array with ``n`` new unique contract IDs."""
        ids = np.fromiter((self._next_id() for _ in range(n)), dtype="int64", count=n)
        return ids

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
        self._log_rates("snapshot", self._snapshot)
        self._generate_panel()
        tol_pp = self._sampler_kwargs.get("tol_pp", 0.5)
        for k in range(self.max_recompute_iter):
            self._compute_targets()
            from credit_data_sampler import TargetSampler

            sampler = TargetSampler(
                target_ratio=self.target_ratio,
                strategy=self.sampler_strategy,
                **self._sampler_kwargs,
            )
            self._panel, overflow, info = sampler.fit_transform(
                self._panel,
                target_col="ever90m12",
                safra_col="safra",
                group_col="grupo_homogeneo",
                random_state=self.rng.integers(0, 1_000_000),
                return_info=True,
            )
            self._log("recompute %d: max_pp=%.2f", k + 1, info["max_pp"])
            if info["max_pp"] <= tol_pp and info["order_ok"]:
                break
            self._reinject(overflow)
        self._snapshot = (
            self._panel[self._panel["data_ref"] == self._panel["data_ref"].min()] 
            .copy()
            .reset_index(drop=True)
        )
        self._log_rates("final", self._panel)
        return self._snapshot, self._panel, self._trace

    # ------------------------------------------------------------------
    # Passo 1: snapshot
    # ------------------------------------------------------------------
    def _generate_snapshot(self) -> None:
        records: List[pd.DataFrame] = []
        start_date = self.start_safra

        contracts: List[pd.DataFrame] = []
        new_clients = 0
        reused_clients = 0
        for gp in self.group_profiles:
            df = gp.sample_contracts(
                self.contracts_per_group,
                rng=self.rng,
                ids=self._next_ids(self.contracts_per_group),
                buckets=self.buckets,
                start_safra=start_date,
            )
            if self.jitter:
                df["dias_atraso"] = _apply_delay_jitter(
                    df["dias_atraso"].to_numpy(),
                    self.buckets,
                    self.rng,
                    self.jitter_pct,
                    self.jitter_min,
                )
            if self.writeoff_bucket is not None:
                df["write_off"] = 0

            if (df["dias_atraso"] < 60).sum() == 0:
                df.at[df.index[0], "dias_atraso"] = 30

            contracts.append(df)
            self._log(
                "snapshot GH=%s rows=%d bad=%.2f%%",
                gp.name,
                len(df),
                df["ever90m12"].mean() * 100,
            )

        client_counts: dict[int, int] = {}
        client_delay: dict[int, int] = {}

        for df in contracts:
            for idx in df.index:
                # pick client respecting max_simultaneous_contracts
                for _ in range(100):
                    cli_idx = int(self.rng.integers(0, len(self._clients)))
                    cli = self._clients.iloc[cli_idx]
                    cid = int(cli.id_cliente)
                    if client_counts.get(cid, 0) < self.max_simultaneous_contracts:
                        break
                else:
                    raise RuntimeError("Not enough clients for concurrent contracts")

                if client_counts.get(cid, 0) == 0:
                    new_clients += 1
                else:
                    reused_clients += 1
                client_counts[cid] = client_counts.get(cid, 0) + 1
                start = random_contract_start(start_date, self.rng)
                while (cid, start.strftime("%Y%m")) in self._start_pairs:
                    start = random_contract_start(start_date, self.rng)
                self._start_pairs.add((cid, start.strftime("%Y%m")))

                if client_counts[cid] > 1:
                    prev = client_delay[cid]
                    idx_prev = bucket_index(prev, self.buckets)
                    low = max(0, idx_prev - 1)
                    high = min(len(self.buckets) - 1, idx_prev + 1)
                    new_idx = int(self.rng.integers(low, high + 1))
                    df.at[idx, "dias_atraso"] = self.buckets[new_idx]
                client_delay[cid] = int(df.at[idx, "dias_atraso"])

                df.at[idx, "data_inicio_contrato"] = start
                df.at[idx, "id_cliente"] = cid
                df.at[idx, "data_nascimento"] = cli.data_nascimento
                df.at[idx, "sexo"] = cli.sexo
                df.at[idx, "duration_m"] = self.rng.integers(6, self.max_duration_m + 1)
                df.at[idx, "age_months"] = 0
                df.at[idx, "data_fim_contrato"] = pd.NaT

            df["data_ref"] = start_date
            df["safra"] = start_date.strftime("%Y%m")

            if self.kernel_trick:
                age = ((df["data_ref"] - df["data_nascimento"]).dt.days // 365).astype("int16")
                q_renda = pd.qcut(df["renda_mensal"], q=4, labels=False, duplicates="drop")
                q_age = pd.qcut(age, q=4, labels=False, duplicates="drop")
                df["subcluster"] = ((q_renda.astype("int8") * 4) + q_age.astype("int8")).astype("int8")

            df["data_ref"] = start_date
            df["safra"] = start_date.strftime("%Y%m")

            if self.kernel_trick:
                age = ((df["data_ref"] - df["data_nascimento"]).dt.days // 365).astype("int16")
                q_renda = pd.qcut(df["renda_mensal"], q=4, labels=False, duplicates="drop")
                q_age = pd.qcut(age, q=4, labels=False, duplicates="drop")
                df["subcluster"] = ((q_renda.astype("int8") * 4) + q_age.astype("int8")).astype("int8")

            records.append(df)

        self._snapshot = pd.concat(records, ignore_index=True, copy=False)
        self._log("snapshot clients new=%d reused=%d", new_clients, reused_clients)

    # ------------------------------------------------------------------
    # Passo 2: evolução mensal
    # ------------------------------------------------------------------
    def _generate_panel(self) -> None:
        assert self._snapshot is not None

        current = self._snapshot.copy()
        start_date = current["data_ref"].iloc[0]
        traces: List[dict] = []
        records = [current.copy()]
        closed_recs: List[pd.DataFrame] = []

        for m in range(1, self.n_safras):
            ref_date = start_date + pd.DateOffset(months=m)
            panel_month, current, stats = self._evolve_one_month(current, ref_date, traces)
            records.append(panel_month.copy())
            closed_recs.append(panel_month[panel_month["data_fim_contrato"].notna()].copy())
            monthly_rate = panel_month["ever90m12"].mean() * 100
            delta_pp = monthly_rate - self.target_ratio * 100
            self._log(
                "%s summary rows=%d bad=%.2f%% \u0394pp=%.2f",
                ref_date.strftime("%Y-%m"),
                len(panel_month),
                monthly_rate,
                delta_pp,
            )
            self._log_rates(ref_date.strftime("%Y-%m"), current)

        self._panel = pd.concat(records, ignore_index=True)
        self._closed = pd.concat(closed_recs, ignore_index=True) if closed_recs else pd.DataFrame()
        self._trace = pd.DataFrame(traces, columns=["id_antigo", "id_novo", "data_evento"])

    # ------------------------------------------------------------------
    def _evolve_one_month(
        self,
        df: pd.DataFrame,
        ref_date: pd.Timestamp,
        trace_records: List[dict],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        df = df.copy()
        buckets = np.array(self.buckets, dtype=np.int16)
        n_buckets = len(buckets)
        stats = {gp.name: {"new": 0, "closed": 0, "defaults": 0} for gp in self.group_profiles}
        df["age_months"] += 1
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
            delay_buckets = np.array([bucket_index(d, list(buckets)) for d in new_delay], dtype=np.int16)
            bucket_factor = np.where(
                delay_buckets == 0,
                0.1,
                np.where(delay_buckets == 1, 0.3, np.where(delay_buckets == 2, 0.6, 1.0)),
            )
            trigger1_prob = gp.reneg_prob_exog * bucket_factor
            trigger1 = self.rng.random(len(sub)) < trigger1_prob
            streak_90 = np.where(new_delay == 90, prev_streak90 + 1, 0)
            trigger2 = (streak_90 >= 3) & (new_delay == 90)
            reneg_mask = trigger1 | trigger2
            stats[gp.name]["defaults"] = int((new_delay >= 90).sum())

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
            elig = on_time & (sub["num_parcelas_pagas_consecutivas"] >= 4)
            accept = self.rng.random(len(sub)) < gp.p_accept_refin
            refin_mask = elig & accept
            sub.loc[refin_mask, "nivel_refinanciamento"] += 1
            sub.loc[refin_mask, "saldo_devedor"] *= 1.1
            sub.loc[refin_mask, "num_parcelas_pagas_consecutivas"] = 0

            # Renegociação
            for idx in sub.index[reneg_mask]:
                old_id = int(sub.at[idx, "id_contrato"])
                new_id = int(self._next_id())
                trace_records.append(
                    {"id_antigo": old_id, "id_novo": new_id, "data_evento": ref_date}
                )
                sub.at[idx, "id_contrato"] = new_id
                sub.at[idx, "nivel_refinanciamento"] = 0
                sub.at[idx, "dias_atraso"] = 0
                sub.at[idx, "streak_90"] = 0
                sub.at[idx, "safra"] = ref_date.strftime("%Y%m")
                sub.at[idx, "num_parcelas_pagas"] = 0
                sub.at[idx, "data_ult_pgt"] = ref_date
                sub.at[idx, "num_parcelas_pagas_consecutivas"] = 0
                sub.at[idx, "qtd_renegociacoes"] += 1

            noise_arr = np.zeros(len(sub), dtype=int)
            df.loc[mask] = sub

        rates_close = {gp.name: rate for gp, rate in zip(self.group_profiles, self.closure_rate)}
        close_mask = np.zeros(len(df), dtype=bool)
        for gp in self.group_profiles:
            mask = df["grupo_homogeneo"] == gp.name
            prob = rates_close[gp.name]
            rand = self.rng.random(mask.sum())
            base = (df.loc[mask, "age_months"] >= df.loc[mask, "duration_m"]).to_numpy()
            close_mask[mask] = base | (rand < prob)
        df.loc[close_mask, "data_fim_contrato"] = ref_date
        open_df = df[~close_mask].copy()
        closed_ids = set(df.loc[close_mask, "id_cliente"].astype(int))
        for gp in self.group_profiles:
            mask = df["grupo_homogeneo"] == gp.name
            stats[gp.name]["closed"] = int(close_mask[mask].sum())

        # spawn new contracts -------------------------------------------------
        new_list = []
        for idx_gp, gp in enumerate(self.group_profiles):
            vivos = (open_df["grupo_homogeneo"] == gp.name).sum()
            if vivos < self.min_volume:
                continue
            rate_new = self.new_contract_rate[idx_gp]
            n_new = int(round(rate_new * max(vivos, 1)))
            if vivos == 0:
                n_new = max(n_new, 1)
            if n_new > 0:
                active = set(open_df["id_cliente"].astype(int)).union(closed_ids)
                new_df = self._spawn_new_contracts(gp, n_new, ref_date, active)
                new_list.append(new_df)
                stats[gp.name]["new"] = len(new_df)
        if new_list:
            open_df = pd.concat([open_df] + new_list, ignore_index=True)

        panel_df = pd.concat([open_df, df[close_mask]], ignore_index=True)
        panel_df["data_ref"] = ref_date
        panel_df["safra"] = ref_date.strftime("%Y%m")
        open_df["data_ref"] = ref_date
        open_df["safra"] = ref_date.strftime("%Y%m")
        for gp in self.group_profiles:
            mask = panel_df["grupo_homogeneo"] == gp.name
            size = mask.sum()
            defaults = stats[gp.name]["defaults"]
            bad = defaults / size * 100 if size > 0 else 0.0
            self._log(
                "%s GH=%s new=%d closed=%d bad=%.2f%%",
                ref_date.strftime("%Y-%m"),
                gp.name,
                stats[gp.name]["new"],
                stats[gp.name]["closed"],
                bad,
            )
        return panel_df, open_df, stats

    def _spawn_new_contracts(
        self, gp: GroupProfile, n: int, ref_date: pd.Timestamp, active_ids: set[int]
    ) -> pd.DataFrame:
        if n <= 0:
            return pd.DataFrame(columns=self._snapshot.columns if self._snapshot is not None else [])
        df = gp.sample_contracts(
            n,
            rng=self.rng,
            ids=self._next_ids(n),
            buckets=self.buckets,
            start_safra=ref_date,
        )
        if self.jitter:
            df["dias_atraso"] = _apply_delay_jitter(
                df["dias_atraso"].to_numpy(),
                self.buckets,
                self.rng,
                self.jitter_pct,
                self.jitter_min,
            )
        for idx in df.index:
            cid, birth, sexo = self._pick_client(ref_date, active_ids)
            start = ref_date - pd.to_timedelta(self.rng.integers(0, 30), unit="D")
            df.at[idx, "data_inicio_contrato"] = start
            df.at[idx, "id_cliente"] = cid
            df.at[idx, "data_nascimento"] = birth
            df.at[idx, "sexo"] = sexo
            df.at[idx, "duration_m"] = self.rng.integers(6, self.max_duration_m + 1)
            df.at[idx, "age_months"] = 0
            df.at[idx, "data_fim_contrato"] = pd.NaT
            active_ids.add(cid)
        return df

    def _pick_client(self, ref_date: pd.Timestamp, active_ids: set[int]) -> tuple[int, pd.Timestamp, str]:
        key = ref_date.strftime("%Y%m")
        if self.rng.random() < self.p_existing_client and len(self._clients) > 0:
            for _ in range(20):
                idx = int(self.rng.integers(0, len(self._clients)))
                cli = self._clients.iloc[idx]
                pair = (int(cli.id_cliente), key)
                if pair not in self._start_pairs and int(cli.id_cliente) not in active_ids:
                    self._start_pairs.add(pair)
                    return int(cli.id_cliente), cli.data_nascimento, cli.sexo
        cid = int(self._next_client_id)
        self._next_client_id += 1
        sexo = self.rng.choice(["M", "F", "N"], p=[0.48, 0.48, 0.04])
        birth = self.rng.choice(pd.date_range("1940-01-01", "2005-12-31", freq="D"))
        self._clients.loc[len(self._clients)] = [cid, sexo, birth]
        self._start_pairs.add((cid, key))
        return cid, birth, sexo

    # ------------------------------------------------------------------
    def _reinject(self, df: pd.DataFrame) -> None:
        """Reallocate removed contracts into future safras."""
        if df.empty:
            return
        df = df.copy()
        max_date = self._panel["data_ref"].max()
        for idx in df.index:
            add_m = int(self.rng.integers(1, self.realloc_window + 1))
            new_ref = pd.to_datetime(df.at[idx, "data_ref"]) + pd.DateOffset(months=add_m)
            if new_ref > max_date:
                new_ref = max_date
            start = new_ref - pd.to_timedelta(self.rng.integers(0, 30), unit="D")
            df.at[idx, "id_contrato"] = self._next_id()
            df.at[idx, "data_inicio_contrato"] = start
            df.at[idx, "data_ref"] = new_ref
            df.at[idx, "safra"] = new_ref.strftime("%Y%m")
            df.at[idx, "age_months"] = 0
            df.at[idx, "dias_atraso"] = 0
            df.at[idx, "streak_90"] = 0
            df.at[idx, "nivel_refinanciamento"] = 0
            df.at[idx, "num_parcelas_pagas"] = 0
            df.at[idx, "num_parcelas_pagas_consecutivas"] = 0
            df.at[idx, "qtd_renegociacoes"] = 0
            df.at[idx, "data_fim_contrato"] = pd.NaT
            df.at[idx, "ever90m12"] = 0
            df.at[idx, "over90m12"] = 0
            df.at[idx, "ever360m18"] = 0
            df.at[idx, "flag_cura"] = 0
        self._panel = pd.concat([self._panel, df], ignore_index=True)

    # ------------------------------------------------------------------
    def _compute_targets(self) -> None:
        assert self._panel is not None

        self._panel = self._panel.sort_values(["id_contrato", "data_ref"])  # type: ignore[assignment]
        ever = np.zeros(len(self._panel), dtype="int8")
        over = np.zeros(len(self._panel), dtype="int8")
        ever360 = np.zeros(len(self._panel), dtype="int8")
        cura = np.zeros(len(self._panel), dtype="int8")

        idx_90 = next(i for i, b in enumerate(self.buckets) if b >= 90)
        idx_60 = next(i for i, b in enumerate(self.buckets) if b >= 60)
        try:
            idx_360 = next(i for i, b in enumerate(self.buckets) if b >= 360)
        except StopIteration:
            idx_360 = len(self.buckets) - 1
        idx_30 = next(i for i, b in enumerate(self.buckets) if b >= 30)

        trace_map = {row.id_antigo: pd.Timestamp(row.data_evento) for row in self._trace.itertuples()}

        for cid, grp in self._panel.groupby("id_contrato"):
            delays = grp["dias_atraso"].to_numpy()
            delay_idx = np.array([bucket_index(d, self.buckets) for d in delays], dtype=int)
            dates = pd.to_datetime(grp["data_ref"]).to_numpy()
            idx = grp.index.to_numpy()
            event_date = trace_map.get(cid)
            end = grp["data_fim_contrato"].dropna()
            end_date = pd.Timestamp(end.iloc[0]) if len(end) > 0 else None
            for i in range(len(delays)):
                start = pd.Timestamp(dates[i])
                horizon_end_12 = start + pd.DateOffset(months=12)
                horizon_end_18 = start + pd.DateOffset(months=18)
                if end_date is not None:
                    if end_date < horizon_end_12:
                        horizon_end_12 = end_date
                    if end_date < horizon_end_18:
                        horizon_end_18 = end_date
                mask12 = (dates[i:] <= horizon_end_12)
                mask18 = (dates[i:] <= horizon_end_18)
                future_idx_12 = delay_idx[i:][mask12]
                future_idx_18 = delay_idx[i:][mask18]
                if (future_idx_12 >= idx_90).any() or (
                    event_date is not None
                    and start < event_date <= horizon_end_12
                    and delay_idx[i] >= idx_60
                ):
                    ever[idx[i]] = 1

                idx90 = np.where(future_idx_12 >= idx_90)[0]
                if len(idx90) > 0:
                    first = idx90[0]
                    if (future_idx_12[first:] >= idx_30).all():
                        over[idx[i]] = 1

                if (future_idx_18 >= idx_360).any():
                    ever360[idx[i]] = 1

                if i > 0 and delay_idx[i] < idx_90 and delay_idx[i - 1] >= idx_90:
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

    def _apply_sampling(self) -> None:
        pass  # deprecated


    def plot_volume_bad_rate(
        self,
        *,
        target_col: str = "ever90m12",
        safra_col: str = "safra",
        ax: "plt.Axes" | None = None,
    ):
        import matplotlib.pyplot as plt
        import pandas as pd

        df = self._panel
        vol = df.groupby(safra_col)["id_contrato"].nunique()
        bad = df.groupby(safra_col)[target_col].mean()

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(vol.index, vol.values, label="Volume", alpha=0.6)
        ax.set_ylabel("# Contracts")
        ax2 = ax.twinx()
        ax2.plot(bad.index, bad.values, "o-", label="Bad-Rate")
        ax2.set_ylabel("Bad-Rate")
        ax2.yaxis.set_major_formatter(lambda x, _: f"{x:.1%}")
        ax.set_title("Volume & Bad-Rate by Safra")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
        return ax

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

    @property
    def closed(self) -> pd.DataFrame:
        if self._closed is None:
            raise RuntimeError("Call generate() first.")
        return self._closed

    @property
    def clients(self) -> pd.DataFrame:
        return self._clients.copy()

    # ------------------------------------------------------------------
    def summary_by_safra(self) -> pd.DataFrame:
        return self._panel.groupby("safra").agg(
            vol=("id_contrato", "nunique"), bad=("ever90m12", "mean")
        )

    def summary_by_gh(self) -> pd.DataFrame:
        return self._panel.groupby("grupo_homogeneo").agg(
            vol=("id_contrato", "nunique"), bad=("ever90m12", "mean")
        )


# ###############################################################################
# # Script usage guard
# ###############################################################################

# if __name__ == "__main__":
#     synth = CreditDataSynthesizer(default_group_profiles(), contracts_per_group=1_000, n_safras=12)
#     snap, panel, trace = synth.generate()
#     print(snap.head())
#     print(panel.head())
#     print(trace.head())
