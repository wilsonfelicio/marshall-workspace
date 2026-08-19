"""Parser tests for the Pecuarios and Pesqueros tables, against saved fixtures.

Network-free. Each fixture is one real response, trimmed to a few days. The four cover
the four structural shapes, and each assertion below corresponds to a bug the first
version of the parser actually had:

  pec_sub_bov   wide: products are column groups     -> unpivot
  pec_can_por   no date column, date in a banner,    -> carry the banner date down,
                data cells use class=DatosNum,          accept DatosNum,
                two price concepts share a row          keep canal and capote apart
  pec_hue       long: product column, three prices   -> prefer "frecuente" over a range
  pesca_fil     long, and the market arrives in an   -> read Fuente: from encabfec, which
                encabfec banner labelled "Fuente:"      the Can shape uses for dates
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from sniim import pecuarios as P  # noqa: E402

FIX = pathlib.Path(__file__).resolve().parent / "fixtures"


def load(name):
    return (FIX / f"{name}.html").read_text(encoding="utf-8")


def test_wide_table_unpivots_products():
    recs, meta = P.parse(load("pec_sub_bov"))
    df = P.tidy(recs, "res_subproductos")
    assert meta["malformed"] == 0
    assert set(df.producto) == {"Vísceras", "Piel sangre"}
    assert df.destino.notna().all(), "every row must carry the market it came from"
    assert (df.precio > 0).all()
    # the geometric centre of the quoted range, not the arithmetic one
    r = df.iloc[0]
    assert abs(r.precio - (r.precio_min * r.precio_max) ** 0.5) < 1e-9


def test_canal_carries_banner_date_and_splits_price_concepts():
    recs, meta = P.parse(load("pec_can_por"))
    df = P.tidy(recs, "cerdo_canal")
    assert meta["malformed"] == 0, "DatosNum rows must not read as malformed"
    assert df.fecha.notna().all(), "the date comes from the Fecha: banner, not a column"
    # canal and capote are different prices on the same row and must not overwrite each
    # other. Not every row quotes both — plenty of markets publish canal only — so the
    # invariant is checked on the rows where both are present, not on an arbitrary row.
    assert {"canal", "capote"} <= set(df.variante)
    key = ["fecha", "destino", "producto"]
    both = df.pivot_table(index=key, columns="variante", values="precio")
    paired = both.dropna(subset=["canal", "capote"])
    assert len(paired) > 0, "the fixture should contain at least one row quoting both"
    assert (paired.canal != paired.capote).all(), "capote overwrote canal"


def test_long_table_prefers_the_frequent_price():
    recs, meta = P.parse(load("pec_hue"))
    df = P.tidy(recs, "huevo")
    assert meta["malformed"] == 0
    assert len(df) > 0 and df.producto.str.contains("Huevo").all()
    # where a modal price is published it wins over the midpoint of the range
    with_frec = df[df.precio_min.notna() & df.precio_max.notna()]
    assert len(with_frec) > 0
    mids = (with_frec.precio_min * with_frec.precio_max) ** 0.5
    assert (with_frec.precio != mids).any(), "frecuente should differ from the midpoint"


def test_pesca_reads_the_market_from_the_fuente_banner():
    recs, meta = P.parse(load("pesca_fil"))
    df = P.tidy(recs, "pesca_filetes")
    assert meta["malformed"] == 0
    assert df.destino.notna().all()
    assert any("Nueva Viga" in d for d in df.destino.unique())
    assert (df.precio > 0).all()


def test_pagination_banner_is_read():
    assert P.n_pages("<td class=encabPAG> Página 1 de 2 </td>") == (1, 2)
    assert P.n_pages("<td>no banner here</td>") == (1, 1)


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok   {name}")
            except Exception:
                fails += 1
                print(f"  FAIL {name}")
                traceback.print_exc()
    print("\nall passed" if not fails else f"\n{fails} failed")
    raise SystemExit(1 if fails else 0)
