# -*- coding: utf-8 -*-
"""Vatnafræði Lokaverkefni
Authors: Ásgeir Pálsson, Ísak Heiðar Magnússon

# Nauðsynlegir pakkar fyrir kóðan
"""

# Pakkar
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from scipy import stats

"""# Stillingar
Þetta eru stillingar fyrir þann sem er búinn að sækja sér þessu gögn og vill keyra kóðan með með, til dæmis fyrir annað vatnasvið.
"""

# Stillingar fyrir gögn
DATA_DIR = Path("data")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# Lesa inn gögn
WEATHER_FILE = DATA_DIR / "Vedurgogn_ID_66.csv"
FLOW_FILE = DATA_DIR / "Rennslisgogn_ID_66.csv"
ATTRIBUTE_FILE = DATA_DIR / "Eiginleikar Vatnasviðs Catchment_attributes.csv"

# Veldu hvaða vatnasvið þú vilt vinna með
BASIN_ID = 66

START_DATE = "1993-10-01"
END_DATE   = "2023-09-30"

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

"""Hér eru föll sem að lesa inn gögnin, flokka þau og ná í upplýsingar sem samsvara völdu ID númeri. Gögnin sem eru lesin inn nýtast síðan fyrir alla liði."""

# Les veðurgögn úr skrá og skilar aðeins þeim dálkum sem eru notaðir áfram
def load_weather_data(filepath: Path) -> pd.DataFrame:
    # Les inn skrána; gert ráð fyrir semikommu-aðskilnaði
    df = pd.read_csv(filepath, sep=";")

    # Býr til dagsetningardálk úr ár-, mánaðar- og dagsdálkum
    df["date"] = pd.to_datetime(
        dict(year=df["YYYY"], month=df["MM"], day=df["DD"]),
        errors="coerce"  # rangar dagsetningar verða NaT í stað villu
    )

    # Heldur aðeins gögnum innan valins tímabils
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].copy()

    # Velur aðeins þá dálka sem eru notaðir í frekari úrvinnslu
    df = df[["date", "prec_carra", "2m_temp_carra"]]

    return df


# Les rennslisgögn úr skrá og finnur sjálfkrafa líklegan rennslisdálk
def load_flow_data(filepath: Path) -> pd.DataFrame:
    # Les inn skrána; gert ráð fyrir semikommu-aðskilnaði
    df = pd.read_csv(filepath, sep=";")

    # Býr til dagsetningardálk úr ár-, mánaðar- og dagsdálkum
    df["date"] = pd.to_datetime(
        dict(year=df["YYYY"], month=df["MM"], day=df["DD"]),
        errors="coerce"
    )

    # Heldur aðeins gögnum innan valins tímabils
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].copy()

    # Reynir að finna réttan dálk fyrir rennsli, þar sem nöfn geta verið mismunandi milli skráa
    possible_q_cols = ["qobs", "q_obs", "Qobs", "discharge", "q"]
    q_col = None

    for col in possible_q_cols:
        if col in df.columns:
            q_col = col
            break

    # Ef enginn líklegur rennslisdálkur fannst, eru villuskilaboð sýnd með upplýsingum um dálkana í skránni
    if q_col is None:
        raise ValueError(
            f"Fann engan rennslisdálk. Dálkar í skránni eru: {list(df.columns)}"
        )

    # Endurnefnir rennslisdálkinn í sameiginlegt nafn til að auðvelda frekari vinnslu
    df = df[["date", q_col]].rename(columns={q_col: "flow_mean"})

    return df


# Les attributes-skrá; reynir fyrst venjulegt csv og síðan semikommuaðskilið ef þarf
def load_attribute_data(filepath: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)

        if df.shape[1] == 1:
            df = pd.read_csv(filepath, sep=";")

    # Ef lestur mistekst alveg er reynt aftur með semikommu
    except Exception:
        df = pd.read_csv(filepath, sep=";")

    return df


# Sækir eina línu úr attribute-skránni fyrir tiltekið basin ID
def get_basin_row(filepath: Path, basin_id: int) -> pd.Series:
    # Les inn alla attribute-skrána
    df = load_attribute_data(filepath)

    # Möguleg heiti á ID-dálki
    possible_id_cols = ["id", "ID", "gauge_id", "gauge_ID"]
    id_col = None

    # Finnur fyrsta dálk sem passar sem auðkennisdálkur
    for col in possible_id_cols:
        if col in df.columns:
            id_col = col
            break

    # Hendir villu ef enginn ID-dálkur fannst
    if id_col is None:
        raise ValueError(f"Fann ekki ID dálk. Dálkar eru: {list(df.columns)}")

    # Velur þá línu sem samsvarar basin_id
    row = df[df[id_col] == basin_id]

    # Hendir villu ef basin_id fannst ekki í skránni
    if row.empty:
        raise ValueError(f"Fann ekki basin_id = {basin_id}")

    # Skilar fyrstu línu sem pandas Series
    return row.iloc[0]


# Sækir gildi úr attribute-línu ef dálkurinn er til; annars NaN
def safe_get(attr: pd.Series, colname: str):
    return attr[colname] if colname in attr.index else np.nan


# Velur út helstu eiginleika vatnasviðsins í sömu röð og þeir birtast í töflu
def extract_selected_attributes(attr: pd.Series) -> dict:
    return {
        "Flatarmál A [km²]": safe_get(attr, "area_calc"),
        "Meðalhæð [m]": safe_get(attr, "elev_mean"),
        "Miðhæð [m]": safe_get(attr, "elev_med"),
        "Hæðarsvið [m]": safe_get(attr, "elev_ran"),
        "Staðalfrávik hæðar [m]": safe_get(attr, "elev_std"),
        "Meðalhalli": safe_get(attr, "slope_mean"),

        "Meðalúrkoma P": safe_get(attr, "p_mean"),
        "Þurrkstuðull": safe_get(attr, "aridity"),
        "Hlutfall snjókomu": safe_get(attr, "frac_snow"),

        "Jöklar (glac_fra)": safe_get(attr, "glac_fra"),
        "Vötn (lake_fra)": safe_get(attr, "lake_fra"),
        "Þéttbýli (urban_fra)": safe_get(attr, "urban_fra"),

        "Berangur (bare_fra)": safe_get(attr, "bare_fra"),
        "Skógur (forest_fra)": safe_get(attr, "forest_fra"),
        "NDVI (hámark)": safe_get(attr, "ndvi_max"),

        "Jarðvegsgleypni (porosity)": safe_get(attr, "soil_poros"),
        "Rótardýpt [m]": safe_get(attr, "root_dep"),
        "Sandhlutfall": safe_get(attr, "sand_fra"),
        "Leirhlutfall": safe_get(attr, "clay_fra"),
    }


# Breytir dictionary af eiginleikum í DataFrame sem auðvelt er að prenta eða setja í töflu
def attributes_to_dataframe(attr_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame({
        "Eiginleiki": list(attr_dict.keys()),
        "Gildi": list(attr_dict.values())
    })
    return df

"""# Liður 1 Upplýsingar um vatnasviðið"""

# Keyrslukóði fyrir lið 1
# Tekur út helstu stærðir úr Catchment_attributes skránni og skilar þeim út í töflu
attr = get_basin_row(ATTRIBUTE_FILE, BASIN_ID)
selected_attr = extract_selected_attributes(attr)
attr_table = attributes_to_dataframe(selected_attr)

print(attr_table)

"""# Liður 2 Árstíðarsveifla"""

# Hjálparföll Liður 2

# Reiknar meðalmánuð fyrir úrkomu og hitastig út frá daglegum veðurgögnum
def monthly_climatology_weather(df: pd.DataFrame) -> pd.DataFrame:
    # Bætir við mánuðardálki út frá dagsetningu og reiknar meðaltöl fyrir hvern mánuð
    out = (
        df.assign(month=df["date"].dt.month)
          .groupby("month", as_index=False)
          .agg(
              prec_mean=("prec_carra", "mean"),
              temp_mean=("2m_temp_carra", "mean")
          )
    )
    return out


# Reiknar meðalmánuð fyrir rennsli út frá daglegum rennslisgögnum
def monthly_climatology_flow(df: pd.DataFrame) -> pd.DataFrame:
    # Bætir við mánuðardálki út frá dagsetningu og reiknar meðaltal rennslis fyrir hvern mánuð
    out = (
        df.assign(month=df["date"].dt.month)
          .groupby("month", as_index=False)
          .agg(
              flow_mean=("flow_mean", "mean")
          )
    )
    return out


# Sameinar climatology fyrir veður og rennsli í eina töflu
def combine_climatologies(weather_clim: pd.DataFrame, flow_clim: pd.DataFrame) -> pd.DataFrame:
    # Sameinar gögnin á month-dálknum þannig að úrkoma, hitastig og rennsli séu í sömu töflu
    out = pd.merge(weather_clim, flow_clim, on="month", how="inner")
    return out

# Teikniföll fyrir lið 2

# Teiknar meðaltalsár fyrir úrkomu, hitastig og rennsli
def plot_full_climatology(clim: pd.DataFrame, outfile: Path) -> None:
    # Heiti mánaða á x-ás
    month_labels = [
        "Jan", "Feb", "Mar", "Apr", "Maí", "Jún",
        "Júl", "Ágú", "Sep", "Okt", "Nóv", "Des"
    ]

    # Býr til grunnmynd og fyrsta ásinn
    fig, ax1 = plt.subplots(figsize=(11, 6))

    # Úrkoma er sýnd sem súlurit á vinstri y-ás
    bars = ax1.bar(
        clim["month"],
        clim["prec_mean"],
        width=0.7,
        color="steelblue",
        alpha=0.75,
        label="Úrkoma",
        zorder=1
    )
    ax1.set_xlabel("Mánuður")
    ax1.set_ylabel("Úrkoma [mm/dag]")
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(month_labels)
    ax1.grid(axis="y", alpha=0.25)

    # Annar y-ás hægra megin fyrir hitastig
    ax2 = ax1.twinx()
    line_temp, = ax2.plot(
        clim["month"],
        clim["temp_mean"],
        color="firebrick",
        marker="o",
        linewidth=2.2,
        label="Hitastig",
        zorder=3
    )
    ax2.set_ylabel("Hitastig [°C]")

    # Þriðji y-ás enn lengra til hægri fyrir rennsli
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))
    line_flow, = ax3.plot(
        clim["month"],
        clim["flow_mean"],
        color="darkgreen",
        marker="s",
        linewidth=2.2,
        label="Rennsli",
        zorder=4
    )
    ax3.set_ylabel("Rennsli [m$^3$/s]")

    # Titill á mynd
    plt.title("Meðaltalsár fyrir úrkomu, hitastig og rennsli")

    # Sameiginleg skýring fyrir öll gagnasett
    handles = [bars, line_temp, line_flow]
    labels = ["Úrkoma", "Hitastig", "Rennsli"]
    ax1.legend(handles, labels, loc="upper right", frameon=True)

    # Þrengir jaðra og vistar myndina
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")

    # Sýnir myndina á skjá og lokar henni svo
    plt.show()
    plt.close()

# Keyrslukóði fyrir lið 2

# Les inn veður- og rennslisgögn
weather = load_weather_data(WEATHER_FILE)
flow = load_flow_data(FLOW_FILE)

# Reiknar climatology fyrir hvort gagnasafn fyrir sig
clim_weather = monthly_climatology_weather(weather)
clim_flow = monthly_climatology_flow(flow)

# Sameinar niðurstöðurnar í eina töflu
clim_all = combine_climatologies(clim_weather, clim_flow)

# Teiknar og vistar myndina
plot_full_climatology(clim_all, FIG_DIR / "climatology_id66.png")

# Prentar staðfestingu og töflu í console
print("Klárað: figures/climatology_id66.png")
print(clim_all)

"""# Liður 3 Mat á grunnvatnsframlagi


"""

# Hjálparföll fyrir lið 3
# Baseflow separation með Ladson / Lyne-Hollick

def prepare_weather_series(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Hreinsar veðurgögn."""
    df = weather_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    if "prec_carra" in df.columns:
        df["prec_carra"] = df["prec_carra"].clip(lower=0)

    return df


def prepare_flow_series(flow_df: pd.DataFrame) -> pd.DataFrame:
    """Hreinsar rennslisgögn."""
    df = flow_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "flow_mean"]).sort_values("date").reset_index(drop=True)
    df = df[df["flow_mean"] >= 0].copy()
    return df

def lh_pass(q: np.ndarray, alpha: float = 0.925) -> np.ndarray:
    # Framkvæmir eitt pass af Lyne-Hollick.
    # Fallið skilar quickflow, þ.e. þeim hluta rennslisins sem talinn er
    # bregðast hratt við úrkomu / atburðum.
    q = np.asarray(q, dtype=float)
    n = len(q)

    # Ef inntakið er tómt er skilað tómu fylki
    if n == 0:
        return np.array([])

    # Upphafsstillum quickflow
    quick = np.zeros(n, dtype=float)
    quick[0] = 0.0

    # Reiknum quickflow stig fyrir stig
    for i in range(1, n):
        quick[i] = alpha * quick[i - 1] + ((1 + alpha) / 2.0) * (q[i] - q[i - 1])

        # Quickflow má ekki verða neikvætt
        if quick[i] < 0:
            quick[i] = 0.0

        # Quickflow má ekki vera stærra en heildarrennsli sama dag
        if quick[i] > q[i]:
            quick[i] = q[i]

    return quick


def ladson_baseflow(flow_df: pd.DataFrame,
                    alpha: float = 0.925,
                    passes: int = 3) -> pd.DataFrame:
    # Reiknar grunnrennsli (baseflow) með 3-pass Ladson / Lyne-Hollick síu.
    #
    # Hugmyndin er:
    # 1) Meta quickflow með síu
    # 2) Snúa röðinni við og sía aftur til að minnka stefnuáhrif
    # 3) Sía enn einu sinni áfram
    #
    # Að lokum er:
    #   baseflow = total flow - quickflow
    df = prepare_flow_series(flow_df)
    q = df["flow_mean"].to_numpy(dtype=float)

    # Fyrsti pass: áfram
    quick = lh_pass(q, alpha=alpha)

    # Annar pass: afturábak
    if passes >= 2:
        quick = lh_pass(quick[::-1], alpha=alpha)[::-1]

    # Þriðji pass: aftur áfram
    if passes >= 3:
        quick = lh_pass(quick, alpha=alpha)

    # Passum að quickflow sé innan raunhæfra marka
    quick = np.clip(quick, 0, q)

    # Grunnrennsli fæst sem mismunur heildarrennslis og quickflow
    base = q - quick
    base = np.clip(base, 0, q)

    # Skilum nýju DataFrame með bæði quickflow og baseflow
    out = df.copy()
    out["quickflow"] = quick
    out["baseflow"] = base
    return out


def compute_bfi(baseflow_df: pd.DataFrame) -> float:
    # Reiknar Baseflow Index (BFI).
    #
    # BFI = summa(baseflow) / summa(heildarrennslis)
    #
    # Gildi nálægt 1 benda til þess að grunnrennsli sé stór hluti
    # af heildarrennsli. Lægri gildi benda til meiri þátts quickflow.
    total_q = baseflow_df["flow_mean"].sum()
    total_b = baseflow_df["baseflow"].sum()

    # Forðumst deilingu með 0 eða neikvæðu heildarrennsli
    if total_q <= 0:
        return np.nan

    return total_b / total_q


def annual_bfi(baseflow_df: pd.DataFrame) -> pd.DataFrame:
    # Reiknar BFI fyrir hvert almanaksár út frá daglegum gögnum.
    #
    # Athugið:
    # Hér er notað calendar year (date.dt.year), ekki vatnsár.
    df = baseflow_df.copy()
    df["year"] = df["date"].dt.year

    out = (
        df.groupby("year", as_index=False)
          .agg(
              total_flow=("flow_mean", "sum"),
              total_baseflow=("baseflow", "sum")
          )
    )

    out["BFI"] = out["total_baseflow"] / out["total_flow"]
    return out

# Recession analysis


def find_recession_segments(flow_df: pd.DataFrame,
                            min_length: int = 8,
                            allow_equal: bool = True) -> list:
    # Finnur samfelld tímabil þar sem rennsli er minnkandi dag frá degi.
    #
    # Dæmi:
    # ef Q[i] <= Q[i-1] í nokkra daga í röð, þá telst það recession-tímabil.
    #
    # Skilar lista af tuple:
    #   (start_idx, end_idx)
    df = prepare_flow_series(flow_df)
    q = df["flow_mean"].to_numpy(dtype=float)

    segments = []
    start = None

    for i in range(1, len(q)):
        # Veljum hvort jafnt rennsli teljist með sem hluti recession
        cond = q[i] <= q[i - 1] if allow_equal else q[i] < q[i - 1]

        if cond:
            # Byrjum nýtt recession-segment ef við erum ekki þegar inni í einu
            if start is None:
                start = i - 1
        else:
            # Ef minnkandi kafla lýkur, athugum hvort hann sé nógu langur
            if start is not None:
                end = i - 1
                if (end - start + 1) >= min_length:
                    segments.append((start, end))
                start = None

    # Ef röðin endar inni í recession-segmenti þarf líka að vista það
    if start is not None:
        end = len(q) - 1
        if (end - start + 1) >= min_length:
            segments.append((start, end))

    return segments


def recession_constant_from_segment(flow_df: pd.DataFrame,
                                    start_idx: int,
                                    end_idx: int) -> dict | None:
    # Metur recession-constant fyrir eitt valið segment.
    #
    # Gert er ráð fyrir veldisvísisfalli:
    #   Q(t) = Q0 * exp(b*t)
    #
    # Með því að taka náttúrulega logra fæst:
    #   ln(Q) = a + b*t
    #
    # Þar af leiðandi er hægt að nota línulegt aðhvarf á ln(Q).
    df = prepare_flow_series(flow_df)

    # Veljum tímabilið sem á að greina
    seg = df.iloc[start_idx:end_idx + 1].copy()

    # Notum aðeins jákvæð rennslisgildi þar sem ln(Q) er annars ekki skilgreint
    seg = seg[seg["flow_mean"] > 0].copy()

    if len(seg) < 2:
        return None

    seg = seg.reset_index(drop=True)

    # t = 0, 1, 2, ... innan segmentsins
    t = np.arange(len(seg), dtype=float)
    q = seg["flow_mean"].to_numpy(dtype=float)
    lnq = np.log(q)

    # Línuleg aðhvarfsgreining á ln(Q) = a + b*t
    slope, intercept = np.polyfit(t, lnq, 1)

    # Umbreyting yfir í hydrology-stærðir
    k_daily = np.exp(slope)
    tau_days = -1.0 / slope if slope < 0 else np.nan
    q0 = np.exp(intercept)

    # Reiknum R² til að meta hversu vel línulegt líkan passar
    lnq_fit = intercept + slope * t
    ss_res = np.sum((lnq - lnq_fit) ** 2)
    ss_tot = np.sum((lnq - np.mean(lnq)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "start_date": seg["date"].iloc[0],
        "end_date": seg["date"].iloc[-1],
        "n_days": len(seg),
        "k": k_daily,
        "tau": tau_days,
        "q0": q0,
        "slope_lnq": slope,
        "intercept_lnq": intercept,
        "r2": r2
    }


def recession_analysis(flow_df: pd.DataFrame,
                       min_length: int = 8,
                       allow_equal: bool = True) -> pd.DataFrame:
    # Finnur öll recession-segments og reiknar recession-stærðir fyrir hvert þeirra.
    segments = find_recession_segments(
        flow_df,
        min_length=min_length,
        allow_equal=allow_equal
    )

    results = []

    for start_idx, end_idx in segments:
        res = recession_constant_from_segment(flow_df, start_idx, end_idx)
        if res is not None:
            results.append(res)

    # Ef ekkert fannst er skilað tómu DataFrame með fyrirfram skilgreindum dálkum
    if len(results) == 0:
        return pd.DataFrame(columns=[
            "start_date", "end_date", "n_days",
            "k", "tau", "q0", "slope_lnq", "intercept_lnq", "r2"
        ])

    return pd.DataFrame(results)


def filter_recession_segments(rec: pd.DataFrame) -> pd.DataFrame:
    # Síar út recession-segments sem teljast hentugri til túlkunar eða birtingar.
    #
    # Hér er valið:
    # - nokkuð löng segment
    # - góð línuleg fylgni í ln(Q)
    # - en sleppum þeim sem eru "of perfect", sem geta verið grunsamleg
    cand = rec[
        (rec["n_days"] >= 12) &
        (rec["n_days"] <= 30) &
        (rec["r2"] >= 0.95) &
        (rec["r2"] <= 0.995)
    ].copy()

    # Röðum þannig að "bestu" segmentin komi efst
    cand = cand.sort_values(
        ["r2", "n_days"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return cand


def select_recession_segment(recession_df: pd.DataFrame,
                             min_days: int = 10,
                             max_days: int = 25) -> pd.Series:
    # Velur eitt recession-segment til að sýna í mynd eða umfjöllun.
    #
    # Forgangur:
    # 1) innan [min_days, max_days]
    # 2) hátt R²
    # 3) lengra segment
    cand = recession_df[
        (recession_df["n_days"] >= min_days) &
        (recession_df["n_days"] <= max_days)
    ].copy()

    # Ef ekkert segment uppfyllir skilyrðin er allt DataFrame notað
    if len(cand) == 0:
        cand = recession_df.copy()

    cand = cand.sort_values(
        ["r2", "n_days", "start_date"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    return cand.iloc[0]

# Teikniföll fyrir lið 3

def plot_annual_bfi(bfi_df: pd.DataFrame, outfile: Path) -> None:
    # Teiknar BFI fyrir hvert ár og bætir við láréttri meðaltalslínu.
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(bfi_df["year"], bfi_df["BFI"], alpha=0.7)

    mean_bfi = bfi_df["BFI"].mean()
    ax.axhline(
        mean_bfi,
        linestyle="--",
        color="red",
        label=f"Meðal BFI = {mean_bfi:.2f}"
    )

    ax.set_xlabel("Ár")
    ax.set_ylabel("BFI")
    ax.set_title("Baseflow Index (BFI) eftir árum")

    ax.grid(alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_baseflow_separation(baseflow_df: pd.DataFrame,
                             outfile: Path,
                             start=None,
                             end=None) -> None:
    # Teiknar heildarrennsli og grunnrennsli.
    # Svæðið á milli línanna er skyggt og táknar quickflow / beint afrennsli.
    df = baseflow_df.copy()

    # Ef tímabil er skilgreint er myndin klippt niður á það tímabil
    if start is not None:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end is not None:
        df = df[df["date"] <= pd.to_datetime(end)]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df["date"], df["flow_mean"], linewidth=1.0, label="Heildarrennsli")
    ax.plot(df["date"], df["baseflow"], linewidth=2.0, label="Grunnrennsli (Ladson)")

    ax.fill_between(
        df["date"],
        df["baseflow"],
        df["flow_mean"],
        alpha=0.25,
        label="Beint afrennsli"
    )

    ax.set_xlabel("Dagsetning")
    ax.set_ylabel("Rennsli [m$^3$/s]")
    ax.set_title("Baseflow separation með Ladson-aðferð")
    ax.grid(alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_single_recession_segment(flow_df: pd.DataFrame,
                                  segment: pd.Series,
                                  outfile: Path,
                                  pad_days: int = 3) -> None:
    # Teiknar eitt valið recession-tímabil.
    #
    # Auk þess eru sýndir nokkrir dagar fyrir og eftir tímabilið
    # svo auðveldara sé að sjá það í samhengi.
    df = prepare_flow_series(flow_df)

    start = pd.to_datetime(segment["start_date"]) - pd.Timedelta(days=pad_days)
    end = pd.to_datetime(segment["end_date"]) + pd.Timedelta(days=pad_days)

    win = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    seg = df[
        (df["date"] >= pd.to_datetime(segment["start_date"])) &
        (df["date"] <= pd.to_datetime(segment["end_date"]))
    ].copy()

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(win["date"], win["flow_mean"], linewidth=1.3, label="Heildarrennsli")
    ax.plot(seg["date"], seg["flow_mean"], marker="o", linewidth=2.2, label="Valið recession-tímabil")

    ax.set_xlabel("Dagsetning")
    ax.set_ylabel("Rennsli [m$^3$/s]")
    ax.set_title("Valið recession-tímabil")
    ax.grid(alpha=0.25)
    ax.legend()

    # Þéttari dagsetningar á x-ás fyrir zoom-mynd
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_recession_lnq(flow_df: pd.DataFrame,
                       segment: pd.Series,
                       outfile: Path) -> None:
    # Teiknar ln(Q) sem fall af tíma fyrir eitt recession-segment.
    # Einnig er sýnd aðhvarfslína og samantekt á helstu stærðum.
    df = prepare_flow_series(flow_df)

    seg = df[
        (df["date"] >= pd.to_datetime(segment["start_date"])) &
        (df["date"] <= pd.to_datetime(segment["end_date"]))
    ].copy()

    # ln(Q) er aðeins skilgreint fyrir jákvæð gildi
    seg = seg[seg["flow_mean"] > 0].reset_index(drop=True)

    t = np.arange(len(seg), dtype=float)
    q = seg["flow_mean"].to_numpy(dtype=float)
    lnq = np.log(q)

    slope = segment["slope_lnq"]
    intercept = segment["intercept_lnq"]
    r2 = segment["r2"]
    tau = segment["tau"]

    lnq_fit = intercept + slope * t

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.scatter(t, lnq, s=45, label="ln(Q) mælt")
    ax.plot(t, lnq_fit, linewidth=2.0, label="Línuleg aðhvarfsgreining")

    # Textabox með helstu niðurstöðum úr aðhvarfsgreiningunni
    text = (
        f"ln(Q) = {intercept:.2f} + ({slope:.4f})·t\n"
        f"R² = {r2:.3f}\n"
        f"tau = {tau:.1f} dagar"
    )

    ax.text(
        0.98, 0.95, text,
        transform=ax.transAxes,
        ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85)
    )

    ax.set_xlabel("Tími t (dagar frá upphafi)")
    ax.set_ylabel("ln(Q)")
    ax.set_title("Recession greining: ln(Q) sem fall af tíma")
    ax.grid(alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# Keyrslukóði liður 3

# Les inn dagleg rennslisgögn
flow = load_flow_data(FLOW_FILE)

# 1) Baseflow separation

# Reiknum grunnrennsli og quickflow með Ladson-síu
bf = ladson_baseflow(flow, alpha=0.925, passes=3)

# Reiknum heildar-BFI fyrir allt tímabilið
bfi_value = compute_bfi(bf)
print(f"BFI = {bfi_value:.3f}")

# Reiknum BFI fyrir hvert ár
bfi_year = annual_bfi(bf)
print("\nÁrlegt BFI:")
print(bfi_year.head())

# Teiknum BFI eftir árum
plot_annual_bfi(
    bfi_year,
    FIG_DIR / "bfi_annual_id66.png"
)

# Teiknum baseflow separation fyrir allt tímabilið
plot_baseflow_separation(
    bf,
    FIG_DIR / "baseflow_ladson_id66_all.png"
)

# Teiknum zoom inn á styttra tímabil til að sjá síuna betur
plot_baseflow_separation(
    bf,
    FIG_DIR / "baseflow_ladson_id66_zoom.png",
    start="2005-01-01",
    end="2006-12-31"
)

# 2) Recession analysis

# Finnum recession-segments í rennslisröðinni
rec = recession_analysis(flow, min_length=8, allow_equal=True)

# Veljum "snyrtilegri" segment til frekari skoðunar
cand = filter_recession_segments(rec)

print("\nCandidate recession segments:")
print(cand[["start_date", "end_date", "n_days", "r2", "tau"]].head(10))

print("\nFyrstu recession-segments:")
print(rec.head(10))

# Samantekt á k fyrir öll segment
print(f"\nMeðal k = {rec['k'].mean():.4f}")
print(f"Miðgildi k = {rec['k'].median():.4f}")

# Hreinsum út mjög stór tau gildi sem geta skekkt meðaltal
rec_clean = rec[rec["tau"] < 200].copy()

print(f"Meðal tau (hreinsað) = {rec_clean['tau'].mean():.2f} dagar")
print(f"Miðgildi tau (hreinsað) = {rec_clean['tau'].median():.2f} dagar")

# 3) Velja eitt recession-segment til myndbirtingar

# Hér er einfaldlega valið efsta segmentið úr candidate-lista
# Einnig væri hægt að nota select_recession_segment(...)
seg = cand.iloc[0]

print("\nValið recession-tímabil:")
print(seg)

# Teiknum valið recession-segment í samhengi við nærliggjandi daga
plot_single_recession_segment(
    flow,
    seg,
    FIG_DIR / "recession_segment_zoom_id66.png",
    pad_days=3
)

# Teiknum ln(Q) á móti tíma ásamt aðhvarfslínu
plot_recession_lnq(
    flow,
    seg,
    FIG_DIR / "recession_lnq_id66.png"
)

"""# Liður 4 Tenging við grunnlíkingu"""

# Hjálparföll fyrir lið 4

def discharge_to_mm_per_day(flow_m3s: pd.Series, area_km2: float) -> pd.Series:
    # Breytir rennsli [m3/s] í afrennsli [mm/dag] yfir allt vatnasviðið.
    #
    # Hugmyndin er að dreifa rúmmáli dagsins yfir flatarmál vatnasviðsins:
    # 1 mm yfir 1 km² samsvarar 1000 m³
    #
    # því fæst:
    # Q [mm/dag] = Q [m3/s] * 86400 / (area_km2 * 1000)

    if area_km2 is None or area_km2 <= 0:
        raise ValueError("Flatarmál vatnasviðs þarf að vera > 0 km².")

    return flow_m3s * 86400.0 / (area_km2 * 1000.0)


def build_water_balance_dataframe(
    weather_df: pd.DataFrame,
    flow_df: pd.DataFrame,
    area_km2: float
) -> pd.DataFrame:
    # Sameinar veður- og rennslisgögn á dagsetningu og býr til DataFrame
    # fyrir vatnsjöfnuna.
    #
    # Útkoman inniheldur:
    # - úrkomu P [mm/dag]
    # - hitastig T [°C]
    # - rennsli Q [m3/s]
    # - afrennsli Q [mm/dag]
    # - leif P - Q [mm/dag]
    #
    # Hér er ET (uppgufun/gufun) ekki reiknað sérstaklega,
    # þannig að P_minus_Q er einföld leif.

    # Hreinsum gögnin fyrst
    w = prepare_weather_series(weather_df)
    q = prepare_flow_series(flow_df)

    # Sameinum á dagsetningu; aðeins dagar sem eru til í báðum gagnasöfnum haldast inni
    df = pd.merge(w, q, on="date", how="inner")

    # Gefum breytunum skýrari nöfn fyrir vatnsjöfnuna
    df["P_mm"] = df["prec_carra"]         # úrkoma [mm/dag]
    df["T_C"] = df["2m_temp_carra"]       # hitastig [°C]
    df["Q_m3s"] = df["flow_mean"]         # rennsli [m3/s]

    # Breytum rennsli yfir í mm/dag yfir vatnasviðið
    df["Q_mm"] = discharge_to_mm_per_day(df["Q_m3s"], area_km2)

    # Reiknum einfalda leif P - Q (Delta S)
    # Þetta má túlka sem þann hluta sem fer í geymslubreytingu, uppgufun o.fl.
    df["residual_P_minus_Q"] = df["P_mm"] - df["Q_mm"]

    return df


def aggregate_water_balance(df: pd.DataFrame, freq: str = "MS") -> pd.DataFrame:
    # Leggur saman vatnsjöfnuna á lengra tímabil.
    #
    # freq="MS"  -> mánaðarlegt (month start)
    # freq="YS"  -> árlegt frá janúar
    # freq="YS-OCT" -> vatnsár sem byrjar í október

    out = (
        df.set_index("date")
          .resample(freq)
          .agg(
              P_mm=("P_mm", "sum"),
              Q_mm=("Q_mm", "sum"),
              residual_P_minus_Q=("residual_P_minus_Q", "sum"),
              T_mean=("T_C", "mean")
          )
          .reset_index()
    )

    return out


def annual_water_balance_summary(df: pd.DataFrame) -> pd.DataFrame:
    # Reiknar árlega vatnsjöfnu fyrir vatnsár (október–september).
    yearly = aggregate_water_balance(df, freq="YS-OCT").copy()

    # Ár dálkurinn sem birtist hér samsvarar upphafsári vatnsársins í pandas resample
    yearly["year"] = yearly["date"].dt.year

    # Reiknum runoff ratio = Q / P
    yearly["runoff_ratio_calc"] = yearly["Q_mm"] / yearly["P_mm"]

    return yearly


def mean_balance_summary(df: pd.DataFrame) -> pd.Series:
    # Reiknar meðaltal árlegra stærða yfir allt tímabilið.
    yearly = annual_water_balance_summary(df)

    out = {
        "P_mean_mm_per_year": yearly["P_mm"].mean(),
        "Q_mean_mm_per_year": yearly["Q_mm"].mean(),
        "Residual_mean_mm_per_year": yearly["residual_P_minus_Q"].mean(),
        "Runoff_ratio_mean_calc": yearly["runoff_ratio_calc"].mean()
    }

    return pd.Series(out)

# Teikniföll fyrir lið 4

# Samræmdir litir fyrir allar myndir í lið 4
COLOR_P = "steelblue"      # Úrkoma P
COLOR_Q = "darkgreen"      # Afrennsli Q
COLOR_RES = "darkorange"   # Leif P - Q
COLOR_T = "firebrick"      # Hitastig T
COLOR_RR = "orange"        # Runoff ratio


def plot_mean_monthly_water_balance(df: pd.DataFrame, outfile: Path) -> None:
    # Teiknar meðalársferil fyrir:
    # - úrkomu P
    # - afrennsli Q
    # - leif P - Q
    # - hitastig T
    #
    # Þetta hjálpar við að tengja vatnsjöfnuna við árstíðasveiflu.

    tmp = df.copy()
    tmp["month"] = tmp["date"].dt.month

    # Reiknum meðal daglegra gilda fyrir hvern mánuð yfir allt tímabilið
    clim = (
        tmp.groupby("month", as_index=False)
           .agg(
               P_mean=("P_mm", "mean"),
               Q_mean=("Q_mm", "mean"),
               residual_mean=("residual_P_minus_Q", "mean"),
               T_mean=("T_C", "mean")
           )
    )

    month_labels = [
        "Jan", "Feb", "Mar", "Apr", "Maí", "Jún",
        "Júl", "Ágú", "Sep", "Okt", "Nóv", "Des"
    ]

    fig, ax1 = plt.subplots(figsize=(11, 6))

    # Úrkoma sem súlurit
    bars = ax1.bar(
        clim["month"],
        clim["P_mean"],
        width=0.7,
        color=COLOR_P,
        alpha=0.75,
        label="P"
    )

    # Afrennsli sem lína
    line_q, = ax1.plot(
        clim["month"],
        clim["Q_mean"],
        color=COLOR_Q,
        marker="o",
        linewidth=2.2,
        label="Q"
    )

    # Leif P - Q sem lína
    line_res, = ax1.plot(
        clim["month"],
        clim["residual_mean"],
        color=COLOR_RES,
        marker="s",
        linewidth=2.0,
        label="P - Q"
    )

    # Núlllína hjálpar að sjá hvort leifin er jákvæð eða neikvæð
    ax1.axhline(0, color="black", linewidth=1.0, alpha=0.7)

    ax1.set_xlabel("Mánuður")
    ax1.set_ylabel("mm/dag")
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(month_labels)
    ax1.grid(axis="y", alpha=0.25)

    # Hitastig á öðrum y-ás
    ax2 = ax1.twinx()
    line_t, = ax2.plot(
        clim["month"],
        clim["T_mean"],
        color=COLOR_T,
        linestyle="--",
        linewidth=2.0,
        label="T"
    )
    ax2.set_ylabel("Hitastig [°C]")

    # Sameiginleg skýring
    handles = [bars, line_q, line_res, line_t]
    labels = ["P", "Q", "P - Q", "T"]
    ax1.legend(handles, labels, loc="upper right")

    ax1.set_title("Tenging við grunnlíkingu: meðal dagleg gildi eftir mánuðum")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_monthly_balance_timeseries(monthly_df: pd.DataFrame, outfile: Path) -> None:
    # Teiknar mánaðarlega tímaraðamynd fyrir:
    # - úrkomu P [mm/mán]
    # - afrennsli Q [mm/mán]
    # - leif P - Q [mm/mán]

    fig, ax = plt.subplots(figsize=(12, 5))

    # Úrkoma sem súlurit
    bars = ax.bar(
        monthly_df["date"],
        monthly_df["P_mm"],
        width=25,
        color=COLOR_P,
        alpha=0.7,
        label="P [mm/mán]"
    )

    # Afrennsli sem lína
    line_q, = ax.plot(
        monthly_df["date"],
        monthly_df["Q_mm"],
        color=COLOR_Q,
        linewidth=2.0,
        label="Q [mm/mán]"
    )

    # Leif P - Q sem lína
    line_res, = ax.plot(
        monthly_df["date"],
        monthly_df["residual_P_minus_Q"],
        color=COLOR_RES,
        linewidth=1.8,
        label="P - Q [mm/mán]"
    )

    ax.axhline(0, color="black", linewidth=1.0, alpha=0.7)

    ax.set_xlabel("Dagsetning")
    ax.set_ylabel("mm á mánuði")
    ax.set_title("Vatnsjafna á mánaðargrunni")
    ax.grid(alpha=0.25)
    ax.legend(handles=[bars, line_q, line_res])

    # Merki á x-ás á tveggja ára fresti
    ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_annual_balance(yearly_df: pd.DataFrame, outfile: Path) -> None:
    # Teiknar árlegan samanburð fyrir:
    # - úrkomu P [mm/ár]
    # - afrennsli Q [mm/ár]
    # - runoff ratio [-]

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Úrkoma og afrennsli sem tvö súlurit hlið við hlið
    bars_p = ax1.bar(
        yearly_df["year"] - 0.2,
        yearly_df["P_mm"],
        width=0.4,
        color=COLOR_P,
        alpha=0.9,
        label="P"
    )

    bars_q = ax1.bar(
        yearly_df["year"] + 0.2,
        yearly_df["Q_mm"],
        width=0.4,
        color=COLOR_Q,
        alpha=0.9,
        label="Q"
    )

    ax1.set_xlabel("Ár")
    ax1.set_ylabel("mm/ár")
    ax1.set_title("Árleg vatnsjafna")
    ax1.grid(alpha=0.25)

    # Runoff ratio á öðrum y-ás
    ax2 = ax1.twinx()
    line_rr, = ax2.plot(
        yearly_df["year"],
        yearly_df["runoff_ratio_calc"],
        color=COLOR_RR,
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.8,
        linewidth=2.8,
        label="Runoff ratio",
        zorder=5
    )

    # Viðmiðunarlína við runoff ratio = 1
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax2.set_ylabel("Runoff ratio [-]")

    handles = [bars_p, bars_q, line_rr]
    labels = ["P", "Q", "Runoff ratio"]
    ax1.legend(handles, labels, loc="upper right")

    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# Keyrslukóði fyrir lið 4

# Les inn veður- og rennslisgögn
weather = load_weather_data(WEATHER_FILE)
flow = load_flow_data(FLOW_FILE)

# Sækir attribute-gögn fyrir valið vatnasvið
attr = get_basin_row(ATTRIBUTE_FILE, BASIN_ID)
selected_attr = extract_selected_attributes(attr)

# Flatarmál vatnasviðs er notað til að breyta Q úr m3/s í mm/dag
area_km2 = selected_attr["Flatarmál A [km²]"]

print("--------------------------------------------------")
print("Valið vatnasvið")
print("--------------------------------------------------")
print(f"Basin ID: {BASIN_ID}")
print(f"Flatarmál [km²]: {selected_attr['Flatarmál A [km²]']}")
print(f"Meðalhæð [m]: {selected_attr['Meðalhæð [m]']}")
print(f"Meðalhalli: {selected_attr['Meðalhalli']}")
print(f"Meðalúrkoma P: {selected_attr['Meðalúrkoma P']}")
print(f"Hlutfall snjókomu: {selected_attr['Hlutfall snjókomu']}")
print(f"Jöklar (glac_fra): {selected_attr['Jöklar (glac_fra)']}")
print("--------------------------------------------------")

# Reiknum daglega vatnsjöfnu
wb = build_water_balance_dataframe(
    weather_df=weather,
    flow_df=flow,
    area_km2=area_km2
)

# Leggjum saman á mánuði og vatnsár
monthly_wb = aggregate_water_balance(wb, freq="MS")
yearly_wb = annual_water_balance_summary(wb)

# Samantekt yfir allt tímabilið
summary_wb = mean_balance_summary(wb)

print("\nMeðaltal yfir tímabilið:")
print(summary_wb)

print("\nÁrleg samantekt (fyrstu 5 ár):")
print(yearly_wb.head())

# Vistum niðurstöður sem csv ef óskað er
wb.to_csv(FIG_DIR / f"water_balance_daily_id{BASIN_ID}.csv", index=False)
monthly_wb.to_csv(FIG_DIR / f"water_balance_monthly_id{BASIN_ID}.csv", index=False)
yearly_wb.to_csv(FIG_DIR / f"water_balance_yearly_id{BASIN_ID}.csv", index=False)

# Teiknum allar myndir fyrir lið 4
plot_mean_monthly_water_balance(
    wb,
    FIG_DIR / f"water_balance_mean_monthly_id{BASIN_ID}.png"
)

plot_monthly_balance_timeseries(
    monthly_wb,
    FIG_DIR / f"water_balance_monthly_timeseries_id{BASIN_ID}.png"
)

plot_annual_balance(
    yearly_wb,
    FIG_DIR / f"water_balance_annual_id{BASIN_ID}.png"
)

print("\nKlárað lið 4.")
print(f"Myndir vistaðar í: {FIG_DIR}")

"""# Liður 5 Langæislína rennslis"""

# Hjálparföll fyrir lið 5

def compute_flow_duration_curve(flow_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # Reiknar langæislínu rennslis (flow duration curve, FDC).
    #
    # Aðferð:
    # 1) Tökum öll rennslisgildi úr gagnasafninu
    # 2) Röðum þeim frá hæsta til lægsta
    # 3) Reiknum exceedance probability fyrir hvert gildi
    #
    # Skilar:
    # - DataFrame með exceedance probability og rennsli
    # - dictionary með einkennandi gildum Q5, Q50 og Q95

    # Tökum aðeins gild rennslisgildi
    flows = flow_df["flow_mean"].dropna().to_numpy()

    # Öryggi ef neikvæð gildi skyldu leynast í gögnunum
    flows = flows[flows >= 0]

    # Röðum rennslinu frá hæsta til lægsta
    flows_sorted = np.sort(flows)[::-1]
    n = len(flows_sorted)

    # Röðunartölur 1, 2, 3, ..., n
    ranks = np.arange(1, n + 1)

    # Exceedance probability í prósentum
    # Þetta segir hversu oft viðkomandi rennsli er jafnað eða exceeded
    exceedance = 100 * ranks / (n + 1)

    # Setjum niðurstöðurnar í DataFrame
    fdc_df = pd.DataFrame({
        "exceedance": exceedance,
        "flow": flows_sorted
    })

    # Finna einkennandi rennslisgildi með línulegri interpolation
    # Q5  = hátt rennsli sem exceeded er aðeins 5% tímans
    # Q50 = miðgildi rennslis
    # Q95 = lágt rennsli sem exceeded er 95% tímans
    q_values = {
        "Q5": np.interp(5, fdc_df["exceedance"], fdc_df["flow"]),
        "Q50": np.interp(50, fdc_df["exceedance"], fdc_df["flow"]),
        "Q95": np.interp(95, fdc_df["exceedance"], fdc_df["flow"]),
    }

    return fdc_df, q_values

# Teiknifall fyrir lið 5

def plot_flow_duration_curve(fdc_df: pd.DataFrame, q_values: dict) -> None:
    # Teiknar langæislínu rennslis bæði:
    # 1) á venjulegum skala
    # 2) á log-skala á y-ás
    #
    # Á x-ás er exceedance probability [%]
    # Á y-ás er rennsli [m3/s]

    # Punktar fyrir Q5, Q50 og Q95
    x_points = [5, 50, 95]
    y_points = [q_values["Q5"], q_values["Q50"], q_values["Q95"]]


    # Mynd 1: Venjulegur skali
    plt.figure(figsize=(9, 6))

    # Aðallínan fyrir flow duration curve
    plt.plot(fdc_df["exceedance"], fdc_df["flow"], linewidth=2)

    # Punktar fyrir Q5, Q50 og Q95
    plt.scatter(x_points, y_points, zorder=3)

    # Textamerkingar við punktana
    plt.text(5, q_values["Q5"], f'  Q5 = {q_values["Q5"]:.2f}', va="bottom")
    plt.text(50, q_values["Q50"], f'  Q50 = {q_values["Q50"]:.2f}', va="bottom")
    plt.text(95, q_values["Q95"], f'  Q95 = {q_values["Q95"]:.2f}', va="bottom", ha="right")

    plt.xlabel("Exceedance probability (%)")
    plt.ylabel("Rennsli [m$^3$/s]")
    plt.title("Langæislína rennslis (venjulegur skali)")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)

    plt.tight_layout()
    plt.show()

    # Mynd 2: Log-skali á y-ás
    plt.figure(figsize=(9, 6))

    # Aðallínan fyrir flow duration curve
    plt.plot(fdc_df["exceedance"], fdc_df["flow"], linewidth=2)

    # Punktar fyrir Q5, Q50 og Q95
    plt.scatter(x_points, y_points, zorder=3)

    # Textamerkingar við punktana
    plt.text(5, q_values["Q5"], f'  Q5 = {q_values["Q5"]:.2f}', va="bottom")
    plt.text(50, q_values["Q50"], f'  Q50 = {q_values["Q50"]:.2f}', va="bottom")
    plt.text(95, q_values["Q95"], f'  Q95 = {q_values["Q95"]:.2f}', va="bottom", ha="right")

    plt.xlabel("Exceedance probability (%)")
    plt.ylabel("Rennsli [m$^3$/s]")
    plt.title("Langæislína rennslis (log-skali)")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)
    plt.yscale("log")

    plt.tight_layout()
    plt.show()

# Keyrslukóði fyrir lið 5

# Les inn rennslisgögn
flow_df = load_flow_data(FLOW_FILE)

# Reiknar langæislínu rennslis og sækir Q5, Q50 og Q95
fdc_df, q_values = compute_flow_duration_curve(flow_df)

# Prentar helstu einkennigildi í console
print(f"Q5  = {q_values['Q5']:.2f}")
print(f"Q50 = {q_values['Q50']:.2f}")
print(f"Q95 = {q_values['Q95']:.2f}")

# Teiknar langæislínuna
plot_flow_duration_curve(fdc_df, q_values)

"""# Liður 6 Flóðagreining"""

# Hjálparföll fyrir lið 6

def get_water_year(date_series: pd.Series) -> pd.Series:
    # Skilar vatnsári þar sem október–desember tilheyra næsta ári.
    # Dæmi:
    # 1993-10-01 -> vatnsár 1994
    return np.where(date_series.dt.month >= 10, date_series.dt.year + 1, date_series.dt.year)


def extract_annual_peak_flows(flow_df: pd.DataFrame) -> pd.DataFrame:
    # Finnur hæsta daglega rennsli í hverju vatnsári.
    #
    # Skilar DataFrame með:
    # - water_year : vatnsár
    # - date_peak  : dagsetning toppsins
    # - peak_flow  : hæsta daglega rennsli þess vatnsárs
    # - peak_month : mánuður toppsins
    df = flow_df.copy()

    # Bætum við vatnsári
    df["water_year"] = get_water_year(df["date"])

    # Finnum index á hæsta rennsli í hverju vatnsári
    idx = df.groupby("water_year")["flow_mean"].idxmax()

    # Sækjum aðeins annual peaks
    annual_peaks = (
        df.loc[idx, ["water_year", "date", "flow_mean"]]
        .rename(columns={"date": "date_peak", "flow_mean": "peak_flow"})
        .sort_values("water_year")
        .reset_index(drop=True)
    )

    # Sækjum líka mánuð toppsins fyrir seasonality-greiningu
    annual_peaks["peak_month"] = annual_peaks["date_peak"].dt.month

    return annual_peaks


def plot_peak_month_bar_chart(annual_peaks: pd.DataFrame) -> None:
    # Teiknar súlurit sem sýnir í hvaða mánuðum annual peaks falla oftast.
    # Þetta er einföld leið til að skoða flood seasonality.
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "Maí", "Jún",
        "Júl", "Ágú", "Sep", "Okt", "Nóv", "Des"
    ]

    # Teljum hversu margir annual peaks falla í hvern mánuð
    counts = (
        annual_peaks["peak_month"]
        .value_counts()
        .reindex(range(1, 13), fill_value=0)
    )

    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(1, 13), counts.values)

    plt.xticks(range(1, 13), month_names)
    plt.xlabel("Mánuður")
    plt.ylabel("Fjöldi annual peaks")
    plt.title("Árstíðasveifla flóða (flood seasonality)")
    plt.grid(axis="y", alpha=0.3)

    # Setjum fjöldann fyrir ofan hverja súlu
    for i, v in enumerate(counts.values, start=1):
        plt.text(i, v + 0.1, str(v), ha="center", va="bottom")

    plt.tight_layout()
    plt.show()


def gringorten_plotting_positions(n: int) -> np.ndarray:
    # Reiknar Gringorten plotting positions fyrir annual maxima.
    #
    # Skilar non-exceedance probability F fyrir röðuð annual peak gildi.
    # Röðin er hugsuð frá stærsta til minnsta gildi.
    m = np.arange(1, n + 1)  # rank, 1 = stærsta gildi
    F = 1 - (m - 0.44) / (n + 0.12)
    return F


def empirical_return_periods(peak_flows: np.ndarray) -> pd.DataFrame:
    # Býr til empirical gögn fyrir flóðagreiningu með Gringorten plotting positions.
    #
    # Skilar DataFrame með:
    # - peak_flow : röðuð annual peak gildi
    # - F         : non-exceedance probability
    # - T         : endurkomutími í árum
    x = np.sort(peak_flows)[::-1]   # stærsta -> minnsta
    n = len(x)

    F = gringorten_plotting_positions(n)
    T = 1 / (1 - F)

    return pd.DataFrame({
        "peak_flow": x,
        "F": F,
        "T": T
    })


def fit_gumbel(peaks: np.ndarray) -> dict:
    # Fit-ar Gumbel dreifingu á annual peak gögnin
    loc, scale = stats.gumbel_r.fit(peaks)
    return {"name": "Gumbel", "dist": stats.gumbel_r, "params": (loc, scale)}


def fit_lognormal3(peaks: np.ndarray) -> dict:
    # Fit-ar 3-parameter lognormal dreifingu á annual peak gögnin
    shape, loc, scale = stats.lognorm.fit(peaks)
    return {"name": "Log Normal 3", "dist": stats.lognorm, "params": (shape, loc, scale)}


def fit_logpearson3(peaks: np.ndarray) -> dict:
    # Fit-ar Log Pearson Type III dreifingu.
    #
    # Aðferð:
    # 1) Tekur log10 af annual peak gildunum
    # 2) Fit-ar Pearson III á log-gildin
    logx = np.log10(peaks)
    skew, loc, scale = stats.pearson3.fit(logx)
    return {"name": "Log Pearson 3", "dist": stats.pearson3, "params": (skew, loc, scale)}


def fitted_quantiles_at_F(fit_result: dict, F: np.ndarray) -> np.ndarray:
    # Reiknar fitted quantiles við gefin non-exceedance probabilities F.
    #
    # Fyrir Log Pearson 3 þarf að fara aftur úr log-rými yfir í upprunalega einingu.
    if fit_result["name"] == "Log Pearson 3":
        q_log = fit_result["dist"].ppf(F, *fit_result["params"])
        return 10 ** q_log
    else:
        return fit_result["dist"].ppf(F, *fit_result["params"])


def return_level(fit_result: dict, T: float) -> float:
    # Reiknar return level Q_T fyrir gefinn endurkomutíma T.
    #
    # Fyrir annual maxima gildir:
    # F = 1 - 1/T
    F = 1 - 1 / T

    if fit_result["name"] == "Log Pearson 3":
        q_log = fit_result["dist"].ppf(F, *fit_result["params"])
        return 10 ** q_log
    else:
        return fit_result["dist"].ppf(F, *fit_result["params"])


def evaluate_fit(peaks: np.ndarray, fit_result: dict) -> float:
    # Metur hversu vel dreifing passar við empirical annual peak gögnin.
    #
    # Hér er notað RMSE milli:
    # - empirical peak gilda
    # - fitted quantiles við sömu plotting positions
    emp = empirical_return_periods(peaks)
    fitted = fitted_quantiles_at_F(fit_result, emp["F"].values)

    rmse = np.sqrt(np.mean((emp["peak_flow"].values - fitted) ** 2))
    return rmse


def fit_all_distributions(peaks: np.ndarray) -> pd.DataFrame:
    # Fit-ar allar dreifingarnar og skilar samanburðartöflu með RMSE.
    #
    # Minna RMSE bendir til betri aðlögunar að gögnunum.
    fits = [
        fit_gumbel(peaks),
        fit_lognormal3(peaks),
        fit_logpearson3(peaks)
    ]

    rows = []
    for fit in fits:
        rows.append({
            "distribution": fit["name"],
            "rmse": evaluate_fit(peaks, fit),
            "fit_object": fit
        })

    results = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    return results


def plot_flood_frequency(peaks: np.ndarray, fit_results: pd.DataFrame) -> None:
    # Teiknar flood frequency plot með:
    # - empirical annual peak punktum
    # - fitted línum fyrir allar prófaðar dreifingar
    emp = empirical_return_periods(peaks)

    # T-grid fyrir sléttar fitted línur
    T_grid = np.linspace(1.01, 200, 500)
    F_grid = 1 - 1 / T_grid

    plt.figure(figsize=(10, 6))

    # Empirical punktar
    plt.scatter(emp["T"], emp["peak_flow"], label="Annual peaks (Gringorten)", zorder=3)

    # Fitted línur fyrir hverja dreifingu
    for _, row in fit_results.iterrows():
        fit = row["fit_object"]
        q_grid = fitted_quantiles_at_F(fit, F_grid)
        plt.plot(T_grid, q_grid, linewidth=2, label=f'{fit["name"]} (RMSE={row["rmse"]:.2f})')

    plt.xscale("log")
    plt.xlabel("Endurkomutími T (ár)")
    plt.ylabel("Hámarksrennsli [m$^3$/s]")
    plt.title("Flóðagreining á annual peak flows")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def bootstrap_return_levels(
    peaks: np.ndarray,
    best_fit_name: str,
    return_periods=(10, 50, 100),
    n_boot=1000,
    ci=0.90,
    random_state=42
) -> pd.DataFrame:
    # Reiknar bootstrap confidence interval fyrir return levels.
    #
    # Fyrir hvert bootstrap-sýni:
    # 1) endursýnum annual peaks með replacement
    # 2) fit-um valda dreifingu á ný
    # 3) reiknum Q_T fyrir gefna endurkomutíma
    rng = np.random.default_rng(random_state)
    n = len(peaks)

    # Orðabók til að safna bootstrap gildum fyrir hvern T
    q_boot = {T: [] for T in return_periods}

    for _ in range(n_boot):
        sample = rng.choice(peaks, size=n, replace=True)

        # Fit-um sömu dreifingu og valin var sem "best"
        if best_fit_name == "Gumbel":
            fit = fit_gumbel(sample)
        elif best_fit_name == "Log Normal 3":
            fit = fit_lognormal3(sample)
        elif best_fit_name == "Log Pearson 3":
            fit = fit_logpearson3(sample)
        else:
            raise ValueError("Óþekkt dreifing í bootstrap.")

        # Reiknum return level fyrir hvert T
        for T in return_periods:
            q_boot[T].append(return_level(fit, T))

    alpha = 1 - ci
    rows = []

    # Reiknum neðri, miðgildi og efri mörk confidence interval
    for T in return_periods:
        vals = np.array(q_boot[T])
        rows.append({
            "return_period": T,
            "lower_90": np.quantile(vals, alpha / 2),
            "median_boot": np.quantile(vals, 0.50),
            "upper_90": np.quantile(vals, 1 - alpha / 2)
        })

    return pd.DataFrame(rows)


def summarize_peak_months(annual_peaks: pd.DataFrame) -> pd.Series:
    # Skilar fjölda annual peaks í hverjum mánuði.
    return (
        annual_peaks["peak_month"]
        .value_counts()
        .reindex(range(1, 13), fill_value=0)
        .rename("count")
    )

# Keyrslukóði fyrir lið 6

# Les inn rennslisgögn
flow_df = load_flow_data(FLOW_FILE)

# 1) Annual peak flows

# Finnum hæsta daglega rennsli í hverju vatnsári
annual_peaks = extract_annual_peak_flows(flow_df)

# Tökum annual peak gildin út sem numpy array fyrir frekari greiningu
peaks = annual_peaks["peak_flow"].values

print("Annual peak flows:")
print(annual_peaks.head())
print()

# 2) Flood seasonality

# Skoðum í hvaða mánuðum annual peaks koma oftast fram
month_counts = summarize_peak_months(annual_peaks)

print("Fjöldi annual peaks í hverjum mánuði:")
print(month_counts)
print()

# Teiknum súlurit fyrir seasonality
plot_peak_month_bar_chart(annual_peaks)

# 3) Fit dreifingar og velja bestu

# Prófum nokkrar dreifingar og berum þær saman með RMSE
fit_results = fit_all_distributions(peaks)

print("Niðurstöður úr dreifingafitti:")
print(fit_results[["distribution", "rmse"]])
print()

# Veljum þá dreifingu sem fær lægsta RMSE
best_fit = fit_results.loc[0, "fit_object"]

print(f'Best fitting dreifing: {best_fit["name"]}')
print()

# ---------------------------------------------------------
# 4) Reikna return levels
# ---------------------------------------------------------

# Reiknum Q10, Q50 og Q100 fyrir bestu dreifingu
for T in [10, 50, 100]:
    qT = return_level(best_fit, T)
    print(f"Q{T} = {qT:.2f}")

print()

# 5) Bootstrap confidence interval

# Reiknum 90% confidence interval fyrir Q10, Q50 og Q100
ci_df = bootstrap_return_levels(
    peaks=peaks,
    best_fit_name=best_fit["name"],
    return_periods=(10, 50, 100),
    n_boot=2000,
    ci=0.90,
    random_state=42
)

print("90% confidence interval:")
print(ci_df)
print()

# 6) Plot frequency analysis

# Teiknum empirical annual peak punktana og fitted dreifingarlínur
plot_flood_frequency(peaks, fit_results)

"""# Liður 7 Leitnigreining"""

# Hjálparföll fyrir lið 7

def add_time_columns(flow_df: pd.DataFrame) -> pd.DataFrame:
    # Bætir við gagnlegum tímadálkum fyrir leitnigreiningu:
    # - year       : almanaksár
    # - month      : mánuður
    # - water_year : vatnsár
    # - season     : árstíð
    #
    # Notum vatnsár svo vetur brotni ekki milli tveggja ára.

    df = flow_df.copy()

    # Tryggjum að date sé á datetime-formi
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Tökum út raðir með vantar dagsetningu eða rennsli
    df = df.dropna(subset=["date", "flow_mean"]).sort_values("date").reset_index(drop=True)

    # Sækjum einfaldar tímabreytur úr dagsetningu
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["water_year"] = get_water_year(df["date"])

    # Flokkum mánuði í árstíðir
    season_map = {
        12: "Vetur", 1: "Vetur", 2: "Vetur",
        3: "Vor",   4: "Vor",   5: "Vor",
        6: "Sumar", 7: "Sumar", 8: "Sumar",
        9: "Haust", 10: "Haust", 11: "Haust"
    }
    df["season"] = df["month"].map(season_map)

    return df


def aggregate_for_trend(flow_df: pd.DataFrame,
                        group_col: str = "water_year",
                        value_col: str = "flow_mean",
                        agg: str = "mean") -> pd.DataFrame:
    # Býr til samandregna tímarað fyrir leitnigreiningu.
    #
    # Dæmi:
    # - eitt meðalrennsli fyrir hvert vatnsár
    # - eitt miðgildi fyrir hvert ár
    # - ein summa fyrir hvert tímabil
    #
    # Skilar DataFrame með dálkunum:
    # - time
    # - value

    df = add_time_columns(flow_df)

    if agg == "mean":
        out = (
            df.groupby(group_col, as_index=False)
              .agg(value=(value_col, "mean"))
              .rename(columns={group_col: "time"})
        )
    elif agg == "median":
        out = (
            df.groupby(group_col, as_index=False)
              .agg(value=(value_col, "median"))
              .rename(columns={group_col: "time"})
        )
    elif agg == "sum":
        out = (
            df.groupby(group_col, as_index=False)
              .agg(value=(value_col, "sum"))
              .rename(columns={group_col: "time"})
        )
    else:
        raise ValueError("agg þarf að vera 'mean', 'median' eða 'sum'.")

    # Röðum í tímaröð
    return out.sort_values("time").reset_index(drop=True)


def aggregate_seasonal_trends(flow_df: pd.DataFrame,
                              agg: str = "mean") -> pd.DataFrame:
    # Reiknar eitt gildi fyrir hverja árstíð í hverju vatnsári.
    #
    # Notum water_year svo vetur (DJF) haldist saman.

    df = add_time_columns(flow_df)

    if agg == "mean":
        out = (
            df.groupby(["water_year", "season"], as_index=False)
              .agg(value=("flow_mean", "mean"))
              .rename(columns={"water_year": "time"})
        )
    elif agg == "median":
        out = (
            df.groupby(["water_year", "season"], as_index=False)
              .agg(value=("flow_mean", "median"))
              .rename(columns={"water_year": "time"})
        )
    elif agg == "sum":
        out = (
            df.groupby(["water_year", "season"], as_index=False)
              .agg(value=("flow_mean", "sum"))
              .rename(columns={"water_year": "time"})
        )
    else:
        raise ValueError("agg þarf að vera 'mean', 'median' eða 'sum'.")

    return out.sort_values(["season", "time"]).reset_index(drop=True)


def lag1_autocorrelation(x: np.ndarray) -> float:
    # Metur lag-1 sjálffylgni (autocorrelation).
    #
    # Þetta segir okkur hversu mikið hvert gildi tengist næsta gildi í röðinni.
    # Slík sjálffylgni getur haft áhrif á leitnipróf eins og Mann-Kendall.

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    # Ef röðin er of stutt getum við ekki metið sjálffylgni á áreiðanlegan hátt
    if len(x) < 3:
        return np.nan

    # Berum saman röðina við sjálfa sig einu skrefi seinna
    x0 = x[:-1]
    x1 = x[1:]

    # Ef staðalfrávik er 0 er fylgni ekki vel skilgreind;
    # þá skilar fallið 0.0
    if np.std(x0) == 0 or np.std(x1) == 0:
        return 0.0

    return np.corrcoef(x0, x1)[0, 1]


def mann_kendall_stats(x: np.ndarray) -> dict:
    # Reiknar hefðbundnar Mann-Kendall stærðir:
    # - S
    # - var(S)
    # - Z
    # - p-gildi
    #
    # Mann-Kendall prófið metur hvort marktæk monotón leitni sé í tímarað.

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)

    # Ef röðin er of stutt er ekki hægt að framkvæma marktæka leitnigreiningu
    if n < 3:
        return {
            "n": n, "S": np.nan, "var_s": np.nan,
            "z": np.nan, "p": np.nan
        }

    # Reiknum S-statistic:
    # summum formerki mismunar milli allra para í röðinni
    S = 0
    for k in range(n - 1):
        S += np.sum(np.sign(x[k + 1:] - x[k]))

    # Leiðrétting fyrir ties (jafn gildi)
    unique_x, counts = np.unique(x, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))

    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    # Umbreytum S yfir í Z-statistic
    if S > 0:
        z = (S - 1) / np.sqrt(var_s)
    elif S < 0:
        z = (S + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    # Tvíhliða p-gildi
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "n": n,
        "S": S,
        "var_s": var_s,
        "z": z,
        "p": p
    }


def modified_mk_test_hamed_rao(x: np.ndarray) -> dict:
    # Einföld practical útgáfa af modified Mann-Kendall prófi
    # með leiðréttingu fyrir lag-1 sjálffylgni.
    #
    # Hugmyndin er:
    # - ef jákvæð sjálffylgni er í röðinni
    # - þá verður venjulegt MK-próf stundum of "bjartsýnt"
    # - því stækkum við dreifnina og minnkum effective sample size

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)

    if n < 3:
        return {
            "n": n,
            "S": np.nan,
            "var_s": np.nan,
            "var_s_mod": np.nan,
            "z_mod": np.nan,
            "p_mod": np.nan,
            "r1": np.nan
        }

    # Byrjum á venjulegu Mann-Kendall prófi
    mk = mann_kendall_stats(x)

    # Metum lag-1 sjálffylgni
    r1 = lag1_autocorrelation(x)

    if not np.isfinite(r1):
        r1 = 0.0

    # Klippum öfgagildi til öryggis
    r1 = np.clip(r1, -0.99, 0.99)

    # Áætlað effective sample size
    n_eff = n * (1 - r1) / (1 + r1)
    n_eff = max(3.0, min(n, n_eff))

    # Leiðrétt dreifni
    var_s_mod = mk["var_s"] * (n / n_eff)

    S = mk["S"]
    if S > 0:
        z_mod = (S - 1) / np.sqrt(var_s_mod)
    elif S < 0:
        z_mod = (S + 1) / np.sqrt(var_s_mod)
    else:
        z_mod = 0.0

    p_mod = 2 * (1 - stats.norm.cdf(abs(z_mod)))

    return {
        "n": n,
        "S": S,
        "var_s": mk["var_s"],
        "var_s_mod": var_s_mod,
        "z_mod": z_mod,
        "p_mod": p_mod,
        "r1": r1,
        "n_eff": n_eff
    }


def theil_sen_trend(time: np.ndarray, values: np.ndarray) -> dict:
    # Reiknar Theil-Sen leitni:
    # - slope
    # - intercept
    # - 95% confidence interval fyrir slope
    #
    # Theil-Sen er robust mat á leitni og þolir betur útliggjara
    # en venjuleg línuleg aðhvarfsgreining.

    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)

    # Notum aðeins endanleg gildi
    mask = np.isfinite(time) & np.isfinite(values)
    time = time[mask]
    values = values[mask]

    if len(values) < 2:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "slope_low": np.nan,
            "slope_high": np.nan
        }

    slope, intercept, slope_low, slope_high = stats.theilslopes(values, time, 0.95)

    return {
        "slope": slope,
        "intercept": intercept,
        "slope_low": slope_low,
        "slope_high": slope_high
    }


def classify_trend(slope: float, p_value: float, alpha: float = 0.05) -> str:
    # Flokkar niðurstöðuna í texta sem auðvelt er að túlka.
    #
    # alpha = marktektarmörk, venjulega 0.05

    if not np.isfinite(slope) or not np.isfinite(p_value):
        return "Óákveðið"

    if p_value >= alpha:
        return "Engin marktæk leitni"
    elif slope > 0:
        return "Marktæk aukning"
    elif slope < 0:
        return "Marktæk minnkun"
    else:
        return "Engin marktæk leitni"


def run_trend_analysis(series_df: pd.DataFrame,
                       time_col: str = "time",
                       value_col: str = "value",
                       alpha: float = 0.05) -> dict:
    # Keyrir bæði:
    # - Theil-Sen leitnimat
    # - modified Mann-Kendall próf
    #
    # á eina tímarað og skilar öllum helstu niðurstöðum í orðabók.

    # Tökum aðeins viðeigandi dálka, hendum tómum gildum og röðum í tíma
    df = series_df[[time_col, value_col]].dropna().sort_values(time_col).reset_index(drop=True)

    time = df[time_col].to_numpy(dtype=float)
    values = df[value_col].to_numpy(dtype=float)

    # Reiknum leitnilínu og marktekt
    ts = theil_sen_trend(time, values)
    mk_mod = modified_mk_test_hamed_rao(values)

    result = {
        "n": len(df),
        "slope": ts["slope"],
        "intercept": ts["intercept"],
        "slope_low_95": ts["slope_low"],
        "slope_high_95": ts["slope_high"],
        "p_value": mk_mod["p_mod"],
        "z_value": mk_mod["z_mod"],
        "lag1_r": mk_mod["r1"],
        "n_eff": mk_mod["n_eff"],
        "trend_text": classify_trend(ts["slope"], mk_mod["p_mod"], alpha=alpha)
    }

    return result


def trend_results_to_df(results: dict, label: str) -> pd.DataFrame:
    # Setur eina leitniniðurstöðu í snyrtilegt DataFrame.
    # Þetta er þægilegt fyrir útprentun eða samantekt í töflu.
    return pd.DataFrame([{
        "flokkur": label,
        "n": results["n"],
        "Theil-Sen slope": results["slope"],
        "95% CI low": results["slope_low_95"],
        "95% CI high": results["slope_high_95"],
        "MK p-gildi": results["p_value"],
        "MK Z": results["z_value"],
        "lag1 r": results["lag1_r"],
        "n_eff": results["n_eff"],
        "túlkun": results["trend_text"]
    }])


def plot_trend_series(series_df: pd.DataFrame,
                      result: dict,
                      title: str,
                      ylabel: str,
                      outfile: Path | None = None) -> None:
    # Teiknar:
    # - mældu tímaraðina
    # - Theil-Sen leitnilínuna
    #
    # Einnig birtast slope, p-gildi og textatúlkun inni á myndinni.

    df = series_df.copy().sort_values("time").reset_index(drop=True)

    x = df["time"].to_numpy(dtype=float)
    y = df["value"].to_numpy(dtype=float)

    # Reiknum fitted Theil-Sen línu
    y_fit = result["intercept"] + result["slope"] * x

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["time"], df["value"], marker="o", linewidth=1.6, label="Gögn")
    ax.plot(df["time"], y_fit, linewidth=2.3, label="Theil-Sen leitnilína")

    # Textabox með helstu tölum
    txt = (
        f"Slope = {result['slope']:.4f}\n"
        f"p = {result['p_value']:.4f}\n"
        f"{result['trend_text']}"
    )

    ax.text(
        0.02, 0.98, txt,
        transform=ax.transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    ax.set_title(title)
    ax.set_xlabel("Ár / vatnsár")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()

    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_seasonal_trends(seasonal_df: pd.DataFrame,
                         alpha: float = 0.05,
                         outfile: Path | None = None) -> pd.DataFrame:
    # Teiknar leitni fyrir hverja árstíð sérstaklega í 2x2 mynd.
    #
    # Fyrir hverja árstíð:
    # - keyrum leitnigreiningu
    # - teiknum gögn + Theil-Sen línu
    # - vistum helstu niðurstöður í töflu

    season_order = ["Vetur", "Vor", "Sumar", "Haust"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    axes = axes.ravel()

    rows = []

    for ax, season in zip(axes, season_order):
        # Veljum gögn fyrir eina árstíð
        sub = seasonal_df[seasonal_df["season"] == season].copy().sort_values("time")

        # Keyrum leitnigreiningu á viðkomandi árstíð
        res = run_trend_analysis(sub, time_col="time", value_col="value", alpha=alpha)

        x = sub["time"].to_numpy(dtype=float)
        y = sub["value"].to_numpy(dtype=float)
        y_fit = res["intercept"] + res["slope"] * x

        # Teiknum árstíðargögnin og fitted línu
        ax.plot(sub["time"], sub["value"], marker="o", linewidth=1.4, label=season)
        ax.plot(sub["time"], y_fit, linewidth=2.0, label="Theil-Sen")

        ax.set_title(f"{season}\n p={res['p_value']:.3f}, slope={res['slope']:.3f}")
        ax.set_xlabel("Vatnsár")
        ax.set_ylabel("Meðalrennsli [m$^3$/s]")
        ax.grid(alpha=0.25)
        ax.legend()

        # Söfnum niðurstöðum fyrir töflu
        rows.append({
            "árstíð": season,
            "n": res["n"],
            "Theil-Sen slope": res["slope"],
            "95% CI low": res["slope_low_95"],
            "95% CI high": res["slope_high_95"],
            "MK p-gildi": res["p_value"],
            "MK Z": res["z_value"],
            "lag1 r": res["lag1_r"],
            "n_eff": res["n_eff"],
            "túlkun": res["trend_text"]
        })

    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    return pd.DataFrame(rows)


def aggregate_monthly_trends(flow_df: pd.DataFrame,
                             agg: str = "mean") -> pd.DataFrame:
    # Reiknar eitt gildi fyrir hvern mánuð í hverju ári.
    #
    # Þetta er gagnlegt ef þú vilt síðar skoða leitni mánaðarlega,
    # t.d. hvort júní hafi aðra þróun en desember.

    df = add_time_columns(flow_df)

    if agg == "mean":
        out = (
            df.groupby(["year", "month"], as_index=False)
              .agg(value=("flow_mean", "mean"))
              .rename(columns={"year": "time"})
        )
    elif agg == "median":
        out = (
            df.groupby(["year", "month"], as_index=False)
              .agg(value=("flow_mean", "median"))
              .rename(columns={"year": "time"})
        )
    elif agg == "sum":
        out = (
            df.groupby(["year", "month"], as_index=False)
              .agg(value=("flow_mean", "sum"))
              .rename(columns={"year": "time"})
        )
    else:
        raise ValueError("agg þarf að vera 'mean', 'median' eða 'sum'.")

    return out.sort_values(["month", "time"]).reset_index(drop=True)

# Keyrslukóði fyrir lið 7

# Les inn rennslisgögn
flow_df = load_flow_data(FLOW_FILE)

# 1) Árleg leitni

# Býr til eina tímarað með einu meðalrennslisgildi fyrir hvert vatnsár
annual_series = aggregate_for_trend(
    flow_df,
    group_col="water_year",
    agg="mean"
)

# Keyrir leitnigreiningu á árlegu tímaraðina
# Hér eru notuð bæði:
# - Theil-Sen til að meta hallatölu leitni
# - modified Mann-Kendall til að meta marktækni
annual_trend = run_trend_analysis(annual_series, alpha=0.05)

# Setur niðurstöðuna í töfluform fyrir snyrtilega birtingu
annual_results_df = trend_results_to_df(annual_trend, "Ársgrunni")

print("Árleg leitnigreining:")
print(annual_results_df)
print()

# Teiknar árlegu tímaraðina og Theil-Sen leitnilínuna
plot_trend_series(
    annual_series,
    annual_trend,
    title="Leitni í meðalrennsli á ársgrunni",
    ylabel="Meðalrennsli [m$^3$/s]",
    outfile=FIG_DIR / f"trend_annual_id{BASIN_ID}.png"
)

# 2) Árstíðaleitni

# Reiknar eitt meðalrennslisgildi fyrir hverja árstíð í hverju vatnsári
seasonal_series = aggregate_seasonal_trends(flow_df, agg="mean")

# Teiknar leitni fyrir vetur, vor, sumar og haust í 2x2 mynd
# og skilar jafnframt niðurstöðutöflu fyrir allar árstíðir
seasonal_results_df = plot_seasonal_trends(
    seasonal_series,
    alpha=0.05,
    outfile=FIG_DIR / f"trend_seasonal_id{BASIN_ID}.png"
)

print("Árstíðaleitni:")
print(seasonal_results_df)
print()

# 3) Mánaðarleitni

# Þetta skref er valfrjálst, en gagnlegt ef þú vilt skoða
# hvort einstakir mánuðir hafi mismunandi þróun í tíma

# Reiknar eitt meðalrennslisgildi fyrir hvern mánuð í hverju ári
monthly_series = aggregate_monthly_trends(flow_df, agg="mean")

monthly_rows = []

# Förum í gegnum alla 12 mánuði og keyrum leitnigreiningu á hvern þeirra
for m in range(1, 13):
    # Veljum aðeins gögn fyrir einn mánuð í einu
    sub = monthly_series[monthly_series["month"] == m].copy()

    # Keyrum leitnigreiningu á viðkomandi mánuði
    res = run_trend_analysis(sub, alpha=0.05)

    # Söfnum niðurstöðunum í lista
    monthly_rows.append({
        "mánuður": m,
        "n": res["n"],
        "Theil-Sen slope": res["slope"],
        "95% CI low": res["slope_low_95"],
        "95% CI high": res["slope_high_95"],
        "MK p-gildi": res["p_value"],
        "MK Z": res["z_value"],
        "lag1 r": res["lag1_r"],
        "n_eff": res["n_eff"],
        "túlkun": res["trend_text"]
    })

# Breytum niðurstöðunum í DataFrame
monthly_results_df = pd.DataFrame(monthly_rows)

print("Mánaðarleitni:")
print(monthly_results_df)
print()

# 4) Vista töflur

# Vistum allar niðurstöðutöflur sem csv-skrár
annual_results_df.to_csv(
    FIG_DIR / f"trend_annual_results_id{BASIN_ID}.csv",
    index=False
)

seasonal_results_df.to_csv(
    FIG_DIR / f"trend_seasonal_results_id{BASIN_ID}.csv",
    index=False
)

monthly_results_df.to_csv(
    FIG_DIR / f"trend_monthly_results_id{BASIN_ID}.csv",
    index=False
)

print("Klárað lið 7.")
print(f"Myndir og töflur vistaðar í: {FIG_DIR}")

"""# Liður 8 Greining á rennslisatburði"""

# Hjálparföll fyrir lið 8

def get_top_n_events_from_annual_peaks(flow_df: pd.DataFrame, n_top: int = 5) -> pd.DataFrame:
    """
    Velur n stærstu annual-peak atburði út frá hámarksrennsli í hverju vatnsári.

    Inntak:
        flow_df : DataFrame með rennslisgögnum
        n_top   : fjöldi toppatburða sem á að skila

    Úttak:
        DataFrame með n stærstu annual peak atburðunum
    """
    # Finnum annual peaks með hjálparfalli sem þú ert búinn að skilgreina áður
    annual_peaks = extract_annual_peak_flows(flow_df).copy()

    # Raða eftir peak_flow frá stærsta niður og velja efstu n
    top_events = (
        annual_peaks.sort_values("peak_flow", ascending=False)
        .head(n_top)
        .reset_index(drop=True)
    )
    return top_events


def build_event_dataframe(
    weather_df: pd.DataFrame,
    flow_df: pd.DataFrame,
    peak_date,
    days_before: int = 10,
    days_after: int = 10
) -> pd.DataFrame:
    """
    Býr til sameiginlegan gagnaramma fyrir stakan atburðaglugga kringum valinn topp.

    Tekur:
    - veðurgögn (úrkoma og hiti)
    - rennslisgögn
    - dagsetningu topps

    og skilar DataFrame með:
    - date
    - P_mm   : úrkoma [mm/d]
    - T_C    : hiti [°C]
    - Q_m3s  : rennsli [m³/s]
    """
    # Samræma veður- og rennsligögn með þeim hjálparföllum sem þú hefur þegar búið til
    w = prepare_weather_series(weather_df)
    q = prepare_flow_series(flow_df)

    # Tryggja að peak_date sé datetime
    peak_date = pd.to_datetime(peak_date)

    # Skilgreina tímaglugga fyrir atburðagreiningu
    start_date = peak_date - pd.Timedelta(days=days_before)
    end_date = peak_date + pd.Timedelta(days=days_after)

    # Velja aðeins gögn sem falla innan gluggans
    w_sub = w[(w["date"] >= start_date) & (w["date"] <= end_date)].copy()
    q_sub = q[(q["date"] >= start_date) & (q["date"] <= end_date)].copy()

    # Sameina gögnin á dagsetningu
    event_df = pd.merge(w_sub, q_sub, on="date", how="inner")

    # Raða eftir dagsetningu svo allt sé í réttri tímaröð
    event_df = event_df.sort_values("date").reset_index(drop=True)

    # Endurnefna dálka í styttri og þægilegri nöfn fyrir áframhaldandi vinnslu
    event_df["P_mm"] = event_df["prec_carra"]
    event_df["T_C"] = event_df["2m_temp_carra"]
    event_df["Q_m3s"] = event_df["flow_mean"]

    return event_df


def estimate_event_metrics(
    event_df: pd.DataFrame,
    rain_threshold_mm: float = 1.0,
    rise_tol: float = 0.5,
    recovery_tol_frac: float = 0.05
) -> dict:
    """
    Metur helstu stærðir fyrir rennslisatburð.

    Reiknar meðal annars:
    - baseline flow
    - start of rise
    - Qpeak
    - end of excess rainfall
    - recovery date
    - time-to-peak
    - excess rain release time
    - recession time

    Ath:
    Þetta er heuristic nálgun sem hentar vel í verkefni,
    en er ekki fullkomin hydrologísk skilgreining fyrir öll tilvik.
    """
    df = event_df.copy().reset_index(drop=True)

    if df.empty:
        raise ValueError("event_df er tómt.")

    # 1) Finna topp rennslis
    peak_idx = df["Q_m3s"].idxmax()
    peak_date = df.loc[peak_idx, "date"]
    q_peak = df.loc[peak_idx, "Q_m3s"]

    # 2) Skilgreina baseline sem fyrsta rennslisgildi í glugganum
    baseline_flow = df.loc[0, "Q_m3s"]
    baseline_date = df.loc[0, "date"]

    # 3) Finna "start of rise"
    #    Fyrsti dagur fyrir topp þar sem Q fer yfir baseline + rise_tol
    rise_threshold = baseline_flow + rise_tol
    rise_candidates = df.index[
        (df.index <= peak_idx) & (df["Q_m3s"] >= rise_threshold)
    ]

    if len(rise_candidates) > 0:
        rise_start_idx = int(rise_candidates[0])
    else:
        rise_start_idx = 0

    rise_start_date = df.loc[rise_start_idx, "date"]
    q_rise_start = df.loc[rise_start_idx, "Q_m3s"]

    # 4) Finna síðasta úrkomudag fyrir eða við topp
    #    þar sem úrkoma er yfir rain_threshold_mm
    rain_candidates = df.index[
        (df.index <= peak_idx) & (df["P_mm"] >= rain_threshold_mm)
    ]

    if len(rain_candidates) > 0:
        rain_end_idx = int(rain_candidates[-1])
        rain_end_date = df.loc[rain_end_idx, "date"]
    else:
        rain_end_idx = None
        rain_end_date = pd.NaT

    # 5) Meta "recovery"
    #    Hér notum við meðalrennsli síðustu 5 daga sem nálgun á
    #    post-event jafnvægisrennsli.
    post_event_mean = df.iloc[-5:]["Q_m3s"].mean()

    # Notum recovery_tol_frac sem hlutfallslegt vikmörk
    # en tryggjum að það verði ekki of lítið
    recovery_tol = max(1.0, recovery_tol_frac * post_event_mean)
    recovery_threshold = post_event_mean + recovery_tol

    # Fyrsti dagur eftir topp þar sem Q er komið niður fyrir threshold
    recovery_candidates = df.index[
        (df.index >= peak_idx) &
        (df["Q_m3s"] <= recovery_threshold)
    ]

    if len(recovery_candidates) > 0:
        recovery_idx = int(recovery_candidates[0])
        recovery_date = df.loc[recovery_idx, "date"]
    else:
        recovery_idx = None
        recovery_date = pd.NaT

    # 6) Reikna tímalengdir í dögum
    if pd.notna(rise_start_date):
        time_to_peak_days = (peak_date - rise_start_date).days
    else:
        time_to_peak_days = np.nan

    if (rain_end_idx is not None) and pd.notna(recovery_date):
        excess_rain_release_days = (recovery_date - rain_end_date).days
    else:
        excess_rain_release_days = np.nan

    if pd.notna(recovery_date):
        recession_time_days = (recovery_date - peak_date).days
    else:
        recession_time_days = np.nan

    # Skila öllu saman í dictionary
    return {
        "baseline_date": baseline_date,
        "baseline_flow": baseline_flow,
        "rise_start_idx": rise_start_idx,
        "rise_start_date": rise_start_date,
        "q_rise_start": q_rise_start,
        "peak_idx": peak_idx,
        "peak_date": peak_date,
        "q_peak": q_peak,
        "rain_end_idx": rain_end_idx,
        "rain_end_date": rain_end_date,
        "recovery_idx": recovery_idx,
        "recovery_date": recovery_date,
        "recovery_threshold": recovery_threshold,
        "time_to_peak_days": time_to_peak_days,
        "excess_rain_release_days": excess_rain_release_days,
        "recession_time_days": recession_time_days
    }


def summarize_event_type(event_df: pd.DataFrame, metrics: dict) -> str:
    """
    Gróf flokkun atburðar út frá úrkomu og hitastigi dagana fyrir topp.
    """
    df = event_df.copy()
    peak_idx = metrics["peak_idx"]

    # Skoðum nokkra daga fyrir topp þar sem orsakaþættir liggja oft helst
    pre = df.iloc[max(0, peak_idx - 5):peak_idx + 1].copy()

    rain_sum = pre["P_mm"].sum()
    temp_mean = pre["T_C"].mean()

    if rain_sum >= 15 and temp_mean > 1:
        return "Líklega regnflóð eða regn-/leysingaflóð"
    elif temp_mean > 3 and rain_sum < 10:
        return "Líklega leysingaflóð / bráðnunardrifið flóð"
    elif rain_sum >= 10 and temp_mean <= 1:
        return "Líklega úrkomuatburður við kalt veður / blandaður atburður"
    else:
        return "Blandaður eða óljós atburður"


def plot_event_analysis(
    event_df: pd.DataFrame,
    metrics: dict,
    title: str,
    outfile: Path | None = None
) -> None:
    """
    Teiknar rennsli, úrkomu og hitastig fyrir valinn atburð.

    Merkir sérstaklega inn:
    - start of rise
    - Qpeak
    - end of excess rainfall
    - recovery
    """
    df = event_df.copy()

    # Litasetning til aðgreiningar
    color_q = "tab:blue"       # rennsli
    color_p = "tab:blue"      # úrkoma
    color_t = "tab:red"        # hitastig
    color_peak = "orange"  # topppunktur
    color_recovery = "tab:purple"
    color_rise = "tab:gray"
    color_rain_end = "orange"

    fig, ax1 = plt.subplots(figsize=(13, 6))

    # 1) Úrkoma sem súlurit á andhverfum hægri ás
    #    Mikil úrkoma birtist þá "niður" frá toppi myndar
    axp = ax1.twinx()
    axp.bar(
        df["date"],
        df["P_mm"],
        width=0.8,
        color=color_p,
        alpha=0.28,
        label="Úrkoma P [mm/d]",
        zorder=1
    )
    axp.invert_yaxis()
    axp.set_ylabel("Úrkoma [mm/dag]", color=color_p)
    axp.tick_params(axis="y", labelcolor=color_p)

    # 2) Hitastig á þriðja ás
    axt = ax1.twinx()
    axt.spines["right"].set_position(("outward", 60))
    line_t, = axt.plot(
        df["date"],
        df["T_C"],
        linestyle="--",
        linewidth=2.0,
        color=color_t,
        label="Hitastig T [°C]",
        zorder=3
    )
    axt.set_ylabel("Hitastig [°C]", color=color_t)
    axt.tick_params(axis="y", labelcolor=color_t)

    # 3) Rennsli á aðalás
    line_q, = ax1.plot(
        df["date"],
        df["Q_m3s"],
        marker="o",
        markersize=5.5,
        linewidth=2.2,
        color=color_q,
        label="Rennsli Q [m$^3$/s]",
        zorder=4
    )
    ax1.set_xlabel("Dagsetning")
    ax1.set_ylabel("Rennsli [m$^3$/s]", color=color_q)
    ax1.tick_params(axis="y", labelcolor=color_q)
    ax1.set_title(title)
    ax1.grid(alpha=0.25)

    # 4) Baseline-lína
    ax1.axhline(
        metrics["baseline_flow"],
        linestyle=":",
        linewidth=1.6,
        color="black",
        alpha=0.8,
        label=f'Baseline Q = {metrics["baseline_flow"]:.1f}'
    )

    # 5) Rise start
    ax1.axvline(
        metrics["rise_start_date"],
        linestyle="--",
        linewidth=1.8,
        color=color_rise,
        alpha=0.95,
        label="Start of rise"
    )


    # 6) Peak punktur
    ax1.scatter(
        metrics["peak_date"],
        metrics["q_peak"],
        s=90,
        color=color_peak,
        edgecolor="black",
        zorder=6,
        label="Qpeak"
    )


    # 7) Lok excess rainfall
    if pd.notna(metrics["rain_end_date"]):
        ax1.axvline(
            metrics["rain_end_date"],
            linestyle="-.",
            linewidth=1.8,
            color=color_rain_end,
            alpha=0.95,
            label="End of excess rainfall"
        )


    # 8) Recovery
    if pd.notna(metrics["recovery_date"]):
        ax1.axvline(
            metrics["recovery_date"],
            linestyle=":",
            linewidth=2.2,
            color=color_recovery,
            alpha=0.95,
            label="Recovery date"
        )
        ax1.scatter(
            metrics["recovery_date"],
            df.loc[metrics["recovery_idx"], "Q_m3s"],
            s=70,
            color=color_recovery,
            edgecolor="black",
            zorder=6,
            label="Recovery point"
        )


    # 9) Textakassi með samantekt
    txt = (
        f"Time-to-peak = {metrics['time_to_peak_days']} dagar\n"
        f"Excess rain release time = {metrics['excess_rain_release_days']} dagar\n"
        f"Recession time = {metrics['recession_time_days']} dagar"
    )

    ax1.text(
        0.02, 0.97, txt,
        transform=ax1.transAxes,
        ha="left", va="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.90)
    )

    # Sameina legend úr öllum ásum
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = axp.get_legend_handles_labels()
    handles3, labels3 = axt.get_legend_handles_labels()

    ax1.legend(
        handles1 + handles2 + handles3,
        labels1 + labels2 + labels3,
        loc="upper right",
        frameon=True
    )

    plt.tight_layout()

    if outfile is not None:
        plt.savefig(outfile, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()


def event_metrics_to_dataframe(metrics: dict, event_type: str) -> pd.DataFrame:
    """
    Setur helstu niðurstöður fyrir stakan atburð í eina töflu.
    """
    return pd.DataFrame([{
        "baseline_date": metrics["baseline_date"],
        "baseline_flow": metrics["baseline_flow"],
        "rise_start_date": metrics["rise_start_date"],
        "peak_date": metrics["peak_date"],
        "q_peak": metrics["q_peak"],
        "rain_end_date": metrics["rain_end_date"],
        "recovery_date": metrics["recovery_date"],
        "time_to_peak_days": metrics["time_to_peak_days"],
        "excess_rain_release_days": metrics["excess_rain_release_days"],
        "recession_time_days": metrics["recession_time_days"],
        "event_type": event_type
    }])

# Keyrslukóði fyrir lið 8

# Lesa inn gögn
weather_df = load_weather_data(WEATHER_FILE)
flow_df = load_flow_data(FLOW_FILE)

# 1) Finna 5 stærstu annual peak atburði
top_events = get_top_n_events_from_annual_peaks(flow_df, n_top=5)

print("Top 5 atburðir úr annual peaks:")
print(top_events)
print()

# 2) Velja einn atburð til nánari skoðunar
#
# index:
# 0 = hæsti toppurinn
# 1 = næsthæsti
# 2 = þriðji hæsti
# o.s.frv.
selected_event = top_events.iloc[0]

peak_date = selected_event["date_peak"]
peak_flow = selected_event["peak_flow"]

print("Valinn atburður:")
print(selected_event)
print()

# 3) Búa til gagnaglugga kringum toppinn
#
# Hér er valið:
# - 10 dagar fyrir topp
# - 30 dagar eftir topp
event_df = build_event_dataframe(
    weather_df=weather_df,
    flow_df=flow_df,
    peak_date=peak_date,
    days_before=10,
    days_after=30
)

# 4) Meta helstu metrics fyrir atburðinn
metrics = estimate_event_metrics(
    event_df,
    rain_threshold_mm=1.0,   # úrkoma þarf að vera >= 1 mm/d til að teljast virk
    rise_tol=0.5,            # Q þarf að fara 0.5 m³/s yfir baseline til að teljast byrjað að rísa
    recovery_tol_frac=0.05   # recovery threshold = post_event_mean + 5%
)

# Gróf flokkun atburðar eftir veðurfarslegum einkennum
event_type = summarize_event_type(event_df, metrics)

# Setja samantekt í töflu
event_summary_df = event_metrics_to_dataframe(metrics, event_type)

print("Samantekt fyrir atburð:")
print(event_summary_df)
print()

# 5) Teikna mynd fyrir atburðinn
plot_event_analysis(
    event_df,
    metrics,
    title=f"Greining á rennslisatburði við topp {peak_date.date()} (Qpeak = {peak_flow:.2f} m³/s)",
    outfile=FIG_DIR / f"event_analysis_id{BASIN_ID}.png"
)

# 6) Vista niðurstöður
event_df.to_csv(FIG_DIR / f"event_window_id{BASIN_ID}.csv", index=False)
event_summary_df.to_csv(FIG_DIR / f"event_summary_id{BASIN_ID}.csv", index=False)

print("Klárað lið 8.")
print(f"Mynd vistuð í: {FIG_DIR / f'event_analysis_id{BASIN_ID}.png'}")
