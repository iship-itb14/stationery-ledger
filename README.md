# The Stationery Ledger
Data science on real Amazon stationery listings and reviews (Amazon Reviews 2023, McAuley Lab).

Pipeline: `data/*.csv` -> `python analysis.py` (EDA, tests, NLP, model -> data/results.json) -> `python build_site.py` (site/index.html).

Methods: keyword product-type classifier, pack-size parsing, Spearman correlations, log-log pack discount regression,
Mann-Whitney (Japanese brands, type-adjusted), TF-IDF + logistic regression on review text, aspect mining by regex,
review-date seasonality, gradient boosting (5-fold CV AUC, permutation importance), hidden-gem ranking.

Deploy: `site/index.html` is one static file. GitHub Pages (serve /site) or drag the folder into Netlify.

Limitations: product types come from title keywords; reviews are the first 60k in the source file (not random);
review dates are not sales; Amazon only; ratings are bunched at 4 to 5 stars; correlations only.
See scripts/README_data_prep.md for how the CSVs were made.
