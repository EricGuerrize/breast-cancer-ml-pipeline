# Dataset — Breast Cancer Wisconsin (Diagnostic)

## Equivalência das fontes

| Fonte | Como carregar | Credenciais |
|-------|---------------|-------------|
| scikit-learn | `load_breast_cancer()` / `--source sklearn` | Não |
| UCI Archive | HTTP `wdbc.data` / `--source uci` | Não |
| Kaggle | `uciml/breast-cancer-wisconsin-data` / `--source kaggle` | `KAGGLE_USERNAME` + `KAGGLE_KEY` |

As três fontes representam o **mesmo** conjunto Diagnostic (Wolberg et al.). O padrão do projeto é **sklearn**, para CI e demos offline.

## Schema

- 30 features numéricas contínuas (raio, textura, perímetro, área, suavidade, etc. — mean / se / worst)
- Alvo binário: `0` maligno, `1` benigno (sklearn)
- **Sem** coluna `id` no conjunto de treino (anti-vazamento)

## Referência

Wolberg, W. H., Street, W. N., & Mangasarian, O. L. Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository.
