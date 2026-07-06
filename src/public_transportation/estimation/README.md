# estimation

Replacement for `bayesian_estimation`.

The package is organized around one common statistical model specification and
two estimation engines.

```text
estimation/
  common/
    model_blackbox.py
    persistence_utils.py
  bayesian/
    core.py
    config.py
    results.py
    ...
  maximum_likelihood/
    core.py
    config.py
    results.py
    ...
```

## Shared model contract

Both engines use the same functions:

```python
loglik(theta, data) -> scalar
logprior(theta) -> scalar
```

For Bayesian estimation, `logprior` contributes to the posterior. For ML,
`prior_weight=0` ignores it and `prior_weight=1` gives the penalized ML/MAP
criterion induced by the same prior.

## Bayesian

```python
from estimation.bayesian import run_vi

result = run_vi(
    dim=dim,
    data=data,
    loglik=loglik,
    logprior=logprior,
)
```

## Maximum likelihood

```python
from estimation.maximum_likelihood import run_ml

result = run_ml(
    dim=dim,
    data=data,
    loglik=loglik,
    logprior=logprior,
    prior_weight=0.0,  # pure ML
)
```

Use `prior_weight=1.0` for MAP/penalized ML using the same prior strength as
Bayesian estimation.
