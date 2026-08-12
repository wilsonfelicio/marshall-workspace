"""INPC (Mexican CPI) collection, for comparison against the SNIIM wholesale series.

Why this package exists
-----------------------
The 32 SNIIM categories collected by the `sniim` package are not an arbitrary
list. They are *exactly* INEGI's published non-core analytical group
**"Frutas y verduras"** (serie 865557, weight 4.78 of the INPC). Verified three
independent ways:

  * count   - INPC 2024 methodological document, Cuadro 11: Frutas y verduras = 32 genéricos
  * weight  - Cuadro 20: Frutas y verduras = 4.78; our 32 ponderadores sum to 4.77888
  * numeric - aggregating the 32 published genérico indices reproduces serie 865557's
              month-on-month change to 3 decimal places (RMSE 0.029 pp vs 0.46 pp for
              the nearest alternative grouping)

That gives the project a free, published, headline-relevant validation target:
if the 32 item models are any good, they must aggregate to 865557.

Note also that Frijol (068) and Chile seco (064) ARE inside this non-core group -
they are not core "mercancías alimenticias", which is the intuitive but wrong
guess. Only "Otras legumbres secas" (079) from the same subclase is core.

Modules
-------
catalogo   the 32 genéricos, their serie ids, INPC weights and chaining factors
genericos  index series (monthly + quincenal + the analytical subindices)
precios    Precios Promedio - actual retail prices in MXN per unit, by city
"""

__version__ = "1.0.0"
