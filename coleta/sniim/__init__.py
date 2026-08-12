"""Collector for Mexican wholesale food prices published by SNIIM
(Sistema Nacional de Informacion e Integracion de Mercados, Secretaria de Economia).

Modules
-------
config     load config.yml, resolve paths
http       polite single-threaded HTTP session with backoff
parse      header-driven parser for the SNIIM results tables
catalog    refresh product / origin / destination catalogs from the form dropdowns
frutas     daily Frutas y Hortalizas collector (date-range endpoint)
granos     weekly Granos Basicos collector (Semana/Mes/Anio endpoint) - frijol
store      Parquet store + resumable job manifest + DuckDB build
aggregate  INPC-category aggregates
"""

__version__ = "1.0.0"
