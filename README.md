# KLTN
Global Digital Inclusion: Constructing a Country-Level Digital Inclusion Index from World Bank Data and Using Machine Learning to Detect Latent Regional and Income Clusters

dii_thesis/
  config/
    indicator_catalog.csv
  data/
    processed/
  logs/
  src/
    00_settings.py
    01_fetch_countries.py
    02_fetch_wdi_indicators.py
    03_build_wide_panel.py
    04_dq_report.py


python dii_thesis/src/s11_deep_collect.py --config dii_thesis/config/dii_deep_sources.yaml
python dii_thesis/src/s12_deep_build.py   --config dii_thesis/config/dii_deep_sources.yaml
python dii_thesis/src/s13_deep_audit.py   --config dii_thesis/config/dii_deep_sources.yaml
