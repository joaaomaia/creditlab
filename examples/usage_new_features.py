from credit_data_synthesizer import CreditDataSynthesizer, default_group_profiles
from credit_data_sampler import TargetSampler

# gerar painel com churn dinamico
synth = CreditDataSynthesizer(
    group_profiles=default_group_profiles(2),
    contracts_per_group=100,
    n_safras=12,
    new_contract_rate=0.05,
    closure_rate=0.03,
)
_, panel, _ = synth.generate()

# rebalancear bad-rate preservando ranking dos grupos
sampler = TargetSampler(target_ratio=0.08, preserve_rank=True)
panel_bal = sampler.fit_transform(panel, target_col="ever90m12", safra_col="safra")
print(panel_bal.head())
