import pandas as pd
from typing import List, Tuple, Literal, Dict, Sequence, Hashable
from collections import defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import random
from pandas.tseries.offsets import DateOffset

TargetType = Literal["ever", "over"]

class CalculadoraTargets:
    """
    Uma classe para calcular metas de inadimplência ('ever' e 'over') em um DataFrame.
    """

    def __init__(
        self,
        specs: List[Tuple[TargetType, int, int]],
        *,
        date_col: str = "anomes",
        contract_col: str = "ccb",
        dpd_col: str = "atraso",
    ):
        self.date_col = date_col
        self.contract_col = contract_col
        self.dpd_col = dpd_col
        self._parsed_specs = self._parse_specs(specs)
        self._ever_specs_grouped: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
        self._over_specs_grouped: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
        for name, t_type, days, horizon in self._parsed_specs:
            if t_type == "ever":
                self._ever_specs_grouped[horizon].append((name, days))
            else:
                self._over_specs_grouped[horizon].append((name, days))

    def _parse_specs(
        self, specs: List[Tuple[TargetType, int, int]]
    ) -> List[Tuple[str, TargetType, int, int]]:
        out = []
        for t_type, days, horizon in specs:
            if t_type not in {"ever", "over"}:
                raise ValueError(f"Tipo de meta desconhecido: '{t_type}'. Use 'ever' ou 'over'.")
            if not isinstance(days, int) or not isinstance(horizon, int) or days <= 0 or horizon <= 0:
                raise ValueError("Os parâmetros 'days' e 'horizon' devem ser inteiros positivos.")
            name = f"{t_type}{days}m{horizon}"
            out.append((name, t_type, days, horizon))
        return out

    def _calculate_ever_targets(self, df: pd.DataFrame) -> None:
        for horizon, specs_for_horizon in self._ever_specs_grouped.items():
            df_rev = df.sort_values([self.contract_col, "_date"], ascending=[True, False])
            
            window_in_days = f"{horizon * 31}D"
            
            max_future_dpd = (
                df_rev.groupby(self.contract_col, group_keys=False)
                      .rolling(window=window_in_days, on="_date", min_periods=1)[self.dpd_col]
                      .max()
            )
                        
            # >>> NOVO TRECHO <<< -----------------------------------------------
            col_aux = f"_max_dpd_{horizon}m"

            # 1) preenche em df_rev sem realinhamento
            df_rev[col_aux] = max_future_dpd.values

            # 2) copia para df na posição correta
            df.loc[df_rev.index, col_aux] = df_rev[col_aux]
            # -------------------------------------------------------------------

            for col_name, days_threshold in specs_for_horizon:
                    df[col_name] = (df[f"_max_dpd_{horizon}m"] >= days_threshold).astype("int8")

    def _calculate_over_targets(self, df: pd.DataFrame) -> None:
        for horizon, specs_for_horizon in self._over_specs_grouped.items():
            helper_df = df[[self.contract_col, "_date", self.dpd_col]].rename(
                columns={self.dpd_col: f"_dpd_h{horizon}"}
            )
            df[f"_h{horizon}"] = df["_date"] + pd.DateOffset(months=horizon)
            
            df_merged = pd.merge_asof(
                df.sort_values(f"_h{horizon}"),
                helper_df.sort_values("_date"),
                left_on=f"_h{horizon}",
                right_on="_date",
                by=self.contract_col,
                direction="nearest",
                tolerance=pd.Timedelta(days=7)
            )
            df_merged = df_merged.set_index(df.sort_values(f"_h{horizon}").index).sort_index()

            df[f"_dpd_h{horizon}"] = df_merged[f"_dpd_h{horizon}"]
            
            for col_name, days_threshold in specs_for_horizon:
                df[col_name] = (df[f"_dpd_h{horizon}"] >= days_threshold).astype("int8") 
    
    def calcular(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(df_out[self.date_col]):
             df_out["_date"] = pd.to_datetime(df_out[self.date_col], format="%Y%m")
        else:
            df_out["_date"] = df_out[self.date_col]
        
        df_out.sort_values([self.contract_col, "_date"], inplace=True)
        
        if self._ever_specs_grouped:
            self._calculate_ever_targets(df_out)
        
        if self._over_specs_grouped:
            self._calculate_over_targets(df_out)

        cols_to_drop = [c for c in df_out.columns if c.startswith("_")]
        df_out.drop(columns=cols_to_drop, inplace=True)

        return df_out
    

    def plot_targets(
        self,
        df: pd.DataFrame,
        target_cols: Sequence[str],
        *,
        title: str = "Contracts vs Targets over Time",
        normalize: bool = True,          # True  ➜ percentual; False ➜ contagem
        bar_color: str = "steelblue",
        line_mode: str = "lines+markers",
    ) -> go.Figure:
        """
        Cria um gráfico misto (barras + linhas) usando plotly.graph_objects.

        Parameters
        ----------
        df : DataFrame
            Saída já processada por `self.calcular`, contendo as colunas‑alvo.
        target_cols : list/tuple de str
            Nomes das colunas 0/1 que serão traçadas no eixo secundário.
        title : str
            Título do gráfico.
        normalize : bool
            - True  ➜ cada linha mostra a razão  (#1 / #contratos) * 100 (%)
            - False ➜ cada linha mostra a soma (#1) por período.
        bar_color : str
            Cor das barras (qualquer cor plotly válida).
        line_mode : str
            Modo das linhas (ex.: "lines", "markers", "lines+markers").

        Returns
        -------
        plotly.graph_objects.Figure
            Figura pronta para `fig.show()` ou fig.to_html(...).
        """
        # --- preparar eixo X -------------------------------------------------
        grp = df.groupby(self.date_col)

        # barras: nº de contratos (únicos) por mês
        n_contracts = grp[self.contract_col].nunique()

        # linhas: incidência ou contagem de cada target
        lines = {}
        for col in target_cols:
            if normalize:
                lines[col] = grp[col].mean() * 100  # %
            else:
                lines[col] = grp[col].sum()         # contagem

        # --- construir figura ------------------------------------------------
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # barras
        fig.add_trace(
            go.Bar(
                x=n_contracts.index,
                y=n_contracts.values,
                name="Contratos",
                marker_color=bar_color,
                opacity=0.7,
            ),
            secondary_y=False,
        )
        # linhas
        for col, series in lines.items():
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series.values,
                    name=col,
                    mode=line_mode,
                ),
                secondary_y=True,
            )

        # layout
        fig.update_layout(
            title=title,
            bargap=0.15,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title=self.date_col,
            yaxis_title="Number of contracts",
            yaxis2_title="Target % (normalized)" if normalize else "Target count",
            template="plotly_white",
        )
        if normalize:
            fig.update_yaxes(ticksuffix=" %", secondary_y=True)

        return fig
    

    def audit_contract(
        self,
        df: pd.DataFrame,
        *,
        contract_id: Hashable | None = None,
        random_state: int | None = None,
        show_only_mismatches: bool = True,
    ) -> pd.DataFrame:
        """
        Verifica, linha a linha, se cada target calculado para um contrato
        específico está correto.

        Parameters
        ----------
        df : DataFrame
            Saída do método `calcular`, contendo os targets 0/1.
        contract_id : valor hashable ou None
            Identificador do contrato a auditar. Se None → escolhe aleatoriamente.
        random_state : int | None
            Semente para reprodutibilidade quando `contract_id` é None.
        show_only_mismatches : bool
            True  ➜ retorna apenas divergências; False ➜ todas as linhas.

        Returns
        -------
        DataFrame
            Colunas: contrato, data, target, dpd, actual, expected, __match__.
        """
        # -------- 1. Seleção do contrato ---------------------------------
        if contract_id is None:
            rng = np.random.default_rng(random_state)
            contract_id = rng.choice(df[self.contract_col].unique())

        sub = (
            df.loc[df[self.contract_col] == contract_id]
              .sort_values(self.date_col)
              .copy()
        )
        if sub.empty:
            raise ValueError(f"Contrato '{contract_id}' não encontrado no DataFrame.")

        # garantir coluna de datas datetime
        if "_date" not in sub.columns:
            sub["_date"] = pd.to_datetime(sub[self.date_col], format="%Y%m")

        # -------- 2. Recalcular targets manualmente ----------------------
        evid_rows: list[dict] = []

        for _, row in sub.iterrows():
            row_date = row["_date"]

            for name, t_type, days, horizon in self._parsed_specs:
                horizon_end = row_date + DateOffset(months=horizon)

                if t_type == "ever":
                    mask = (sub["_date"] >= row_date) & (sub["_date"] <= horizon_end)
                    expected = int((sub.loc[mask, self.dpd_col] >= days).any())
                else:  # over
                    match_row = sub.loc[sub["_date"] == horizon_end, self.dpd_col]
                    expected = int((match_row >= days).any()) if not match_row.empty else 0

                actual = int(row[name])
                evid_rows.append(
                    {
                        self.contract_col: contract_id,
                        self.date_col: row[self.date_col],
                        "target": name,
                        "dpd": row[self.dpd_col],
                        "actual": actual,
                        "expected": expected,
                        "__match__": actual == expected,
                    }
                )

        evid = pd.DataFrame(evid_rows)
        if show_only_mismatches:
            evid = evid.loc[~evid["__match__"]]

        return evid