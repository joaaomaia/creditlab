# 💳 creditlab

**creditlab** é um laboratório didático para geração e exploração de dados sintéticos aplicados ao risco de crédito.  
Idealizado para fins educacionais, simula contratos, comportamento de clientes, eventos de inadimplência, renegociações, refinanciamentos e evolução temporal da carteira.

---

## 📦 Principais Funcionalidades

- Geração de dados **sintéticos e realistas** para contratos de crédito.
- Simulação de **múltiplos grupos homogêneos de risco**.
- Evolução mensal dos contratos com controle de:
  - Atrasos (dias em mora)
  - Refinanciamentos (clientes bons)
  - Renegociações (clientes em dificuldade)
  - Rastro de contratos renegociados
- Cálculo automático de targets supervisionados:
  - `ever90m12` e `over90m12`
  - Flags de **cura**
- Opções de **stress macroeconômico**
- **Score latente de risco**, para estudos de explicabilidade
- Estrutura orientada a **snapshot vs painel**
- Suporte a múltiplas safras

---

## 📁 Estrutura dos Dados

### `df_snapshot`
Safra inicial com contratos originados, com atributos cadastrais, contratuais e de risco.

### `df_panel`
Evolução mês a mês dos contratos, incluindo comportamento, eventos e targets supervisionados.

### `df_trace`
Rastro de contratos **renegociados**, com relação entre contratos antigos e novos.

### `df_clients`
Registro único de clientes com `id_cliente`, `data_nascimento` e `sexo`. Pode ser
associado aos contratos via `id_cliente`.
Caso precise da idade do cliente em determinado `data_ref`, basta calcular
`idade_cliente = (data_ref - data_nascimento).dt.days // 365`.

---

## 🧪 Exemplos de Aplicações

- Estudo de estratégias de amostragem: snapshot vs painel
- Avaliação de modelos de **PD**, **LGD**, **EAD**
- Análise de **curva de cura** e **tempo até default**
- Teste de algoritmos de explicabilidade com `score_latente`
- Visualização de targets `ever` vs `over`
- Criação de notebooks didáticos de ciência de dados aplicada ao crédito

---

## 🧠 Conceitos Explorados

- Grupos homogêneos de risco
- Eventos de refinanciamento vs renegociação
- Probabilidade de default (PD 12m e lifetime)
- Target `ever90m12` vs `over90m12`
- Curva de cura (curing curve)
- Performing status por modelo (PD, LGD, EAD)
- Estratégias de sampling em modelagem regulatória
- Simulação de eventos sob stress macroeconômico
- Explicabilidade de modelos com *surrogate scores*

---

## ⚙️ Como Usar

```python
from creditlab import CreditDataSynthesizer

synth = CreditDataSynthesizer(
    n_groups=4,
    contracts_per_group=10000,
    n_safras=24,
    seed=42,
    buckets=[0,15,30,60,90,120,180,240,360]
)

df_snapshot, df_panel, df_trace = synth.generate()
df_panel = df_panel.merge(synth.clients, on="id_cliente")
```

Exemplo definindo uma matriz BASE personalizada:

```python
import numpy as np
from creditlab import default_group_profiles

base = np.eye(5)
synth = CreditDataSynthesizer(
    group_profiles=default_group_profiles(3),
    base_matrix=base,
)
```

## Balanced Sampling & Monitoring

```python
synth = CreditDataSynthesizer(
    group_profiles=default_group_profiles(4),
    contracts_per_group=5_000,
    n_safras=36,
    force_event_rate=True,      # balance after generation
    target_ratio=0.10,
    buckets=[0,15,30,60,90,120,180,240,360],
)
_, panel, _ = synth.generate()

# quick QA plot
a = synth.plot_volume_bad_rate()
```

Para preservar a ordem natural dos grupos homogêneos durante o balanceamento,
use ``preserve_gh_order=True``:

```python
from credit_data_sampler import TargetSampler
sampler = TargetSampler(
    target_ratio=0.10,
    preserve_gh_order=True,
    random_state=42,
)
panel_bal = sampler.fit_transform(panel)
```

* ``per_group=True`` força que cada GH tenha a mesma prevalência final.
* ``preserve_gh_order=True`` apenas garante que ``bad_rate(GH_i) ≥ bad_rate(GH_{i+1})``.

---


## 📚 Requisitos
Python 3.8+

pandas, numpy, dataclasses

(futuramente) matplotlib, seaborn, scikit-learn para notebooks

## 🎓 Mentoria
Este repositório foi desenvolvido como parte de uma mentoria gratuita em ciência de dados aplicada ao risco de crédito, com foco técnico, regulatório e estratégico.

Quer aprender mais ou participar?
Entre em contato pelo LinkedIn ou envie um e-mail para [contato_riskpilot@gmail.com].

## 📄 Licença
Este projeto é livre para uso educacional e pessoal.
Reutilização comercial requer autorização prévia.
MIT-like para fins didáticos — cite com carinho. 🤝

## 🚧 Roadmap (em desenvolvimento)
 Gerador de carteiras com múltiplos produtos (consignado, pessoal, auto etc.)

 Geração automática de curvas de cura

 Módulo de explicabilidade com surrogate models

 Simulação de perdas esperadas sob IFRS 9

 Interface via notebook interativo (JupyterLab)