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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd

# =============================================================================
# GroupProfile
# =============================================================================

@dataclass
class GroupProfile:
    """Representa um grupo homogêneo (GH) com parâmetros de risco."""

    name: str
    pd_base: float  # PD 12m baseline (pior grupo ≈ 0.12, melhor ≈ 0.02)
    refin_prob: float  # probabilidade de aceitar refinanciamento quando elegível
    reneg_prob: float  # prob mensal de renegociação exógena (além do trigger 90d)

    # distribuições de variáveis demográficas/financeiras serão geradas on‑the‑fly

    def _delay_probs(self) -> np.ndarray:
        """Retorna vetor de probs para [0, 15, 30, 60, 90] dias."""
        base = np.array([1.0, 0.4, 0.25, 0.1, 0.05])
        weight = (self.pd_base - 0.02) / (0.12 - 0.02)  # 0 (melhor) → 1 (pior)
        # aumenta peso dos maiores atrasos nos grupos piores
        scale = 1 + weight
        scaled = np.array([
            base[0] / scale,            # menos pontualidade nos piores
            base[1],
            base[2] * scale,
            base[3] * scale * 1.3,
            base[4] * scale * 1.6,
        ])
        return scaled / scaled.sum()

    # ---------------------------------------------------------------------
    # Amostragem de contratos
    # ---------------------------------------------------------------------

    def sample_contracts(
        self, n: int, rng: np.random.Generator, *, id_offset: int = 0
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
        df["dias_atraso"] = rng.choice(
            [0, 15, 30, 60, 90], size=n, p=self._delay_probs()
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

        return df

# =============================================================================
# Helpers
# =============================================================================

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
                reneg_prob=reneg_prob,
            )
        )
    return profiles

# =============================================================================
# CreditDataSynthesizer
# =============================================================================

class CreditDataSynthesizer:
    """Gera snapshot, painel e rastro de renegociação."""

    def __init__(
        self,
        *,
        group_profiles: List[GroupProfile],
        contracts_per_group: int = 10_000,
        n_safras: int = 24,
        random_seed: int = 42,
        kernel_trick: bool = True,
    ) -> None:
        self.group_profiles = group_profiles
        self.contracts_per_group = contracts_per_group
        self.n_safras = n_safras
        self.kernel_trick = kernel_trick
        self.rng = np.random.default_rng(random_seed)

        # DataFrames de saída
        self._snapshot: pd.DataFrame | None = None
        self._panel: pd.DataFrame | None = None
        self._trace: pd.DataFrame | None = None

    # ---------------------------------------------------------------------
    # API pública
    # ---------------------------------------------------------------------
    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self._generate_snapshot()
        # Evolução & targets ficam para iteração futura — placeholder
        self._panel = self._snapshot.copy()  # painel = snapshot quando n_safras=1
        self._trace = pd.DataFrame(columns=["id_antigo", "id_novo", "data_evento"])
        return self._snapshot, self._panel, self._trace

    # ------------------------------------------------------------------
    # Passo 1: snapshot
    # ------------------------------------------------------------------
    def _generate_snapshot(self) -> None:
        records: List[pd.DataFrame] = []
        for g_idx, gp in enumerate(self.group_profiles):
            offset = g_idx * self.contracts_per_group
            df = gp.sample_contracts(
                self.contracts_per_group, rng=self.rng, id_offset=offset
            )

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
