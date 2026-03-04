"""
TV Intelligence Dashboard — Data Preparation Script
====================================================
Processes 4.5 GB+ of Samba TV CSVs and outputs a small
aggregated JSON (~3–8 MB) that loads instantly in the browser.

SETUP (one time):
    pip3 install pandas pyarrow

USAGE:
    1. Put all your CSV files inside the csv_files/ folder
    2. Run:  python3 prepare_dashboard_data.py
    3. Load dashboard_data.json into the dashboard HTML

EXPECTED OUTPUT SIZE: 3–8 MB regardless of input size
EXPECTED RUNTIME:     2–5 minutes for 4.5 GB of CSVs
"""

import os, json, gc, time
import pandas as pd
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
CSV_FOLDER  = "./csv_files"
OUTPUT_FILE = "./dashboard_data.json"
CHUNK_SIZE  = 500_000   # rows per chunk — reduce to 200_000 if you hit RAM issues

# ── COLUMNS TO KEEP (everything else is dropped on read — saves RAM & time) ──
KEEP_COLS = [
    # Identity
    "smba_id",
    # Time
    "exposure_start_ts", "exposure_duration",
    "scheduled_program_start_ts", "scheduled_program_end_ts",
    "yyyy", "mm", "dd",
    # Content type & app
    "content_type", "application", "tv_input_type",
    "app_probability_score",
    # Linear TV
    "network", "network_title", "affiliate_call_sign",
    "is_live", "channel_content_offset_s",
    # Programme metadata
    "title", "episode_title", "genres",
    "season", "episode", "program_content_offset_s",
    # Device
    "source_device_category", "source_device_os",
    # Geo
    "city", "subdivision",
]

# ── COLUMNS INTENTIONALLY DROPPED & WHY ──────────────────────────────────────
# exposure_end_ts          → = exposure_start_ts + exposure_duration (100% redundant)
# feed_id                  → single constant value across all rows
# intermediate_id          → 0% fill
# title_id, network_id     → 0% fill
# content_id, channel_id   → raw numeric IDs, name cols already present
# app_id                   → raw numeric ID, application name already present
# mvpd, mvpd_id            → 0% fill
# syndicated_by            → 0% fill
# total_duration           → 0% fill
# content_format           → 0% fill
# source_device_id         → only 9 unique values = device type, not unique device
# postal_code              → city + subdivision covers geo sufficiently
# release_date             → <8% fill, not relevant to viewership analysis
# description              → free text, not aggregatable
# application_availability → only 4% fill
# program_content_offset_s → kept above if you want engagement depth, else remove

STATE_NAMES = {
    "mh":"Maharashtra","up":"Uttar Pradesh","tn":"Tamil Nadu",
    "dl":"Delhi","ap":"Andhra Pradesh","wb":"West Bengal",
    "hr":"Haryana","ka":"Karnataka","mp":"Madhya Pradesh",
    "rj":"Rajasthan","ts":"Telangana","gj":"Gujarat",
    "pb":"Punjab","kl":"Kerala","br":"Bihar","or":"Odisha",
    "jh":"Jharkhand","uk":"Uttarakhand","as":"Assam",
    "hp":"Himachal Pradesh","ch":"Chandigarh","ga":"Goa",
    "jk":"Jammu & Kashmir","mn":"Manipur","ml":"Meghalaya",
    "mz":"Mizoram","nl":"Nagaland","sk":"Sikkim","tr":"Tripura",
}

# ── ACCUMULATOR — collects aggregated stats across all chunks ─────────────────
class Accumulator:
    def __init__(self):
        self.total_rows        = 0
        self.total_exp_secs    = 0
        self.unique_devices    = set()
        self.dates             = set()

        # Pivot: keyed by (content_type, application, device_category, is_live,
        #                  date, city, subdivision, genre, network, hour_bucket)
        # value: [impressions, unique_devices_approx, exposure_seconds]
        self.pivot             = {}

        # Top-N counters
        self.ct_counts         = {}   # content_type
        self.dev_counts        = {}   # device_category
        self.app_counts        = {}   # application (streaming)
        self.os_counts         = {}   # source_device_os (streaming)
        self.net_counts        = {}   # network (linear)
        self.live_counts       = {}   # is_live (linear)
        self.city_counts       = {}   # city
        self.state_counts      = {}   # subdivision
        self.genre_counts      = {}   # genre
        self.title_counts      = {}   # title
        self.hour_counts       = {}   # hour of day (from exposure_start_ts)
        self.input_counts      = {}   # tv_input_type
        self.freq_by_device    = {}   # smba_id -> impression count (for freq dist)
        self.freq_by_device_ct = {}  # content_type -> {smba_id -> count}

        # ── True overlap sets (smba_id level) — global ──
        self.streaming_ids     = set()  # all smba_ids seen in Streaming rows
        self.linear_ids        = set()  # all smba_ids seen in Linear TV rows

        # ── Per-date smba_id sets for daily true-overlap analysis ──
        # Maps date_str → set of smba_ids for that date + content_type
        self.streaming_ids_by_date = {}   # date → set of smba_ids
        self.linear_ids_by_date    = {}   # date → set of smba_ids

    def add(self, counter_dict, key, n=1):
        if key and key not in ("null","None","nan",""):
            counter_dict[key] = counter_dict.get(key, 0) + n

    def top_n(self, counter_dict, n=12):
        return sorted(counter_dict.items(), key=lambda x: -x[1])[:n]


def clean_genre(raw):
    s = str(raw)
    if s in ("nan","null","None",""): return ""
    if ", " in s: return s.split(", ")[1].rstrip("}]").strip()
    return s.strip()


def ts_to_hour(ts):
    """Convert unix timestamp to IST hour bucket (UTC+5:30)."""
    try:
        return (int(ts) // 3600 + 5) % 24   # rough IST
    except:
        return -1


# ── PROCESS ONE CHUNK ─────────────────────────────────────────────────────────
def process_chunk(chunk: pd.DataFrame, acc: Accumulator):
    # ── clean ──
    chunk = chunk.fillna("null")
    chunk["exposure_duration"] = pd.to_numeric(chunk["exposure_duration"], errors="coerce").fillna(0)
    chunk["application"]       = chunk["application"].replace("null", "Linear TV")

    # genre
    chunk["genre"] = chunk["genres"].apply(clean_genre) if "genres" in chunk.columns else ""

    # network: prefer network_title, fall back to network, then application
    if "network_title" in chunk.columns:
        chunk["network_final"] = chunk["network_title"].where(
            (chunk["network_title"] != "null") & (chunk["network_title"] != ""),
            chunk.get("network", chunk["application"])
        )
    elif "network" in chunk.columns:
        chunk["network_final"] = chunk["network"].where(
            chunk["network"] != "null", chunk["application"])
    else:
        chunk["network_final"] = chunk["application"]

    # date
    if all(c in chunk.columns for c in ["yyyy","mm","dd"]):
        chunk["date"] = (chunk["yyyy"].astype(str) + "-" +
                         chunk["mm"].astype(str).str.zfill(2) + "-" +
                         chunk["dd"].astype(str).str.zfill(2))
        chunk.loc[chunk["yyyy"] == "null", "date"] = ""
    else:
        chunk["date"] = ""

    # hour bucket
    if "exposure_start_ts" in chunk.columns:
        chunk["hour"] = chunk["exposure_start_ts"].apply(ts_to_hour)
    else:
        chunk["hour"] = -1

    # device category clean
    dev_map = {"This TV": "Smart TV", "null": "STB"}
    chunk["device_cat"] = chunk["source_device_category"].replace(dev_map) if "source_device_category" in chunk.columns else "Unknown"

    # live label clean
    live_map = {"LIVE":"Live","NOT LIVE":"Not Live","null":"N/A","NULL OFFSET":"Null Offset","UNKNOWN":"Unknown"}
    chunk["live_label"] = chunk["is_live"].replace(live_map) if "is_live" in chunk.columns else "N/A"

    is_streaming = chunk["content_type"] == "Streaming"
    is_linear    = chunk["content_type"] == "Linear TV"

    # ── accumulate global stats ──
    acc.total_rows     += len(chunk)
    acc.total_exp_secs += int(chunk["exposure_duration"].sum())
    acc.unique_devices.update(chunk["smba_id"].tolist())
    acc.dates.update(d for d in chunk["date"].unique() if d and d != "null")

    # frequency distribution (per device impression count)
    for dev_id, cnt in chunk["smba_id"].value_counts().items():
        acc.freq_by_device[dev_id] = acc.freq_by_device.get(dev_id, 0) + cnt

    # per-content-type frequency distribution
    for ct_val, grp in chunk.groupby("content_type"):
        if ct_val in ("null", "None", "nan", ""): continue
        if ct_val not in acc.freq_by_device_ct:
            acc.freq_by_device_ct[ct_val] = {}
        for dev_id, cnt in grp["smba_id"].value_counts().items():
            acc.freq_by_device_ct[ct_val][dev_id] = acc.freq_by_device_ct[ct_val].get(dev_id, 0) + cnt

    # ── top-N counters ──
    for v in chunk["content_type"]:           acc.add(acc.ct_counts,  v)
    for v in chunk["device_cat"]:             acc.add(acc.dev_counts, v)
    for v in chunk["genre"]:                  acc.add(acc.genre_counts, v)
    for v in chunk["date"]:                   acc.dates.add(v)

    if "tv_input_type" in chunk.columns:
        for v in chunk["tv_input_type"]:      acc.add(acc.input_counts, v)

    for v in chunk["city"] if "city" in chunk.columns else []:
        acc.add(acc.city_counts, str(v))
    for v in chunk["subdivision"] if "subdivision" in chunk.columns else []:
        acc.add(acc.state_counts, str(v))
    for v in chunk.get("title", pd.Series(dtype=str)):
        acc.add(acc.title_counts, str(v))

    # streaming-only
    st = chunk[is_streaming]
    acc.streaming_ids.update(st["smba_id"].tolist())   # ← true overlap tracking
    # per-date streaming ids
    for date_val, grp in st.groupby("date"):
        if date_val and date_val != "null":
            if date_val not in acc.streaming_ids_by_date:
                acc.streaming_ids_by_date[date_val] = set()
            acc.streaming_ids_by_date[date_val].update(grp["smba_id"].tolist())
    for v in st["application"]:              acc.add(acc.app_counts, v)
    if "source_device_os" in st.columns:
        for v in st["source_device_os"]:     acc.add(acc.os_counts,  v)

    # linear-only
    ln = chunk[is_linear]
    acc.linear_ids.update(ln["smba_id"].tolist())      # ← true overlap tracking
    # per-date linear ids
    for date_val, grp in ln.groupby("date"):
        if date_val and date_val != "null":
            if date_val not in acc.linear_ids_by_date:
                acc.linear_ids_by_date[date_val] = set()
            acc.linear_ids_by_date[date_val].update(grp["smba_id"].tolist())
    for v in ln["network_final"]:            acc.add(acc.net_counts,  v)
    for v in ln["live_label"]:               acc.add(acc.live_counts, v)

    for v in chunk["hour"]:
        if v >= 0:                           acc.add(acc.hour_counts, str(v))

    # ── pivot ──
    # NOTE: "title" is included here to enable per-app/network title filtering
    # in the dashboard.  This increases output JSON size (~10–30 MB depending
    # on title cardinality).  Remove "title" if file size is a concern.
    pivot_cols = ["content_type","application","device_cat","live_label",
                  "date","city","subdivision","genre","network_final","title"]
    pivot_cols = [c for c in pivot_cols if c in chunk.columns]

    grp = (chunk.groupby(pivot_cols, dropna=False)
                .agg(impressions=("smba_id","count"),
                     unique_devices=("smba_id","nunique"),
                     exposure_seconds=("exposure_duration","sum"))
                .reset_index())

    for _, row in grp.iterrows():
        key = tuple(str(row[c]) for c in pivot_cols)
        if key in acc.pivot:
            acc.pivot[key][0] += int(row["impressions"])
            acc.pivot[key][1] += int(row["unique_devices"])
            acc.pivot[key][2] += int(row["exposure_seconds"])
        else:
            acc.pivot[key] = [int(row["impressions"]),
                               int(row["unique_devices"]),
                               int(row["exposure_seconds"])]

    pivot_col_names = [c.replace("network_final","network")
                         .replace("device_cat","device_category")
                         .replace("live_label","is_live")
                       for c in pivot_cols]
    return pivot_col_names


def compute_freq_dist(freq_dict):
    """Turn {device: count} → standard 5-bucket distribution dict."""
    fc = freq_dict.values()
    return {
        "1":    sum(1 for c in fc if c == 1),
        "2-3":  sum(1 for c in fc if 2 <= c <= 3),
        "4-6":  sum(1 for c in fc if 4 <= c <= 6),
        "7-10": sum(1 for c in fc if 7 <= c <= 10),
        "11+":  sum(1 for c in fc if c >= 11),
    }


def compute_quintiles(freq_dict):
    """Compute avg frequency per quintile (Top 20% .. Bottom 20%)."""
    if not freq_dict:
        return []
    vals = sorted(freq_dict.values(), reverse=True)   # descending
    n = len(vals)
    labels = ["Top 20%", "Second 20%", "Third 20%", "Fourth 20%", "Bottom 20%"]
    result = []
    for i, label in enumerate(labels):
        lo = int(n * i / 5)
        hi = int(n * (i + 1) / 5)
        if hi <= lo:
            result.append({"label": label, "avg_freq": 0.0})
        else:
            chunk = vals[lo:hi]
            result.append({"label": label, "avg_freq": round(sum(chunk) / len(chunk), 2)})
    return result


# ── SERIALISE ACCUMULATOR ─────────────────────────────────────────────────────
def build_output(acc: Accumulator, pivot_col_names: list) -> dict:
    def top(d, n=12, exclude=("null","None","nan","","Linear TV")):
        return [[k, v] for k,v in sorted(d.items(), key=lambda x:-x[1])[:n+len(exclude)]
                if k not in exclude][:n]

    def agg_list(d, name_map=None, n=12, exclude=("null","None","nan","")):
        items = [(name_map.get(k,k) if name_map else k, v)
                 for k,v in sorted(d.items(), key=lambda x:-x[1])
                 if k not in exclude][:n]
        return [{"name":k,"impressions":v} for k,v in items]

    # Frequency distribution (global + per content-type + quintiles)
    freq_dist = compute_freq_dist(acc.freq_by_device)
    freq_dist_by_type = {ct: compute_freq_dist(fd) for ct, fd in acc.freq_by_device_ct.items()}
    freq_quintiles = compute_quintiles(acc.freq_by_device)
    freq_quintiles_by_type = {ct: compute_quintiles(fd) for ct, fd in acc.freq_by_device_ct.items()}

    unique_devs = len(acc.unique_devices)
    total       = acc.total_rows
    dates       = sorted(d for d in acc.dates if d and d != "null")

    # ── True smba_id set-intersection for overlap analysis ──
    both_ids       = acc.streaming_ids & acc.linear_ids
    streaming_only = acc.streaming_ids - acc.linear_ids
    linear_only    = acc.linear_ids    - acc.streaming_ids
    unduplicated   = acc.streaming_ids | acc.linear_ids
    overlap_analysis = {
        "streaming_devices":      len(acc.streaming_ids),     # total unique streaming devices
        "linear_devices":         len(acc.linear_ids),        # total unique linear devices
        "streaming_only_devices": len(streaming_only),        # saw streaming only
        "linear_only_devices":    len(linear_only),           # saw linear only (= incremental)
        "both_devices":           len(both_ids),              # cross-platform (duplicated)
        "unduplicated_total":     len(unduplicated),          # true deduplicated reach
    }

    # ── Per-date true overlap (smba_id set ops per day) ──
    all_overlap_dates = sorted(
        set(list(acc.streaming_ids_by_date.keys()) +
            list(acc.linear_ids_by_date.keys()))
    )
    daily_overlap = []
    for dv in all_overlap_dates:
        st_ids = acc.streaming_ids_by_date.get(dv, set())
        lt_ids = acc.linear_ids_by_date.get(dv, set())
        both_d  = st_ids & lt_ids
        undup_d = st_ids | lt_ids
        daily_overlap.append({
            "date":             dv,
            "streaming_total":  len(st_ids),
            "linear_total":     len(lt_ids),
            "streaming_only":   len(st_ids - lt_ids),
            "linear_only":      len(lt_ids - st_ids),   # ← incremental from linear
            "both":             len(both_d),             # ← cross-platform audience
            "unduplicated":     len(undup_d),
        })

    # Pivot → list of dicts
    pivot_records = []
    for key, (imp, ud, exp) in acc.pivot.items():
        if imp == 0: continue
        rec = dict(zip(pivot_col_names, key))
        rec["impressions"]      = imp
        rec["unique_devices"]   = ud
        rec["exposure_seconds"] = exp
        pivot_records.append(rec)

    # Filter values for chips
    fv_ct  = sorted(set(r.get("content_type","") for r in pivot_records if r.get("content_type") not in ("null","",None)))
    fv_app = sorted(set(r.get("application","")  for r in pivot_records if r.get("application")  not in ("null","",None)))
    fv_dev = sorted(set(r.get("device_category","") for r in pivot_records if r.get("device_category") not in ("null","",None)))
    fv_liv = sorted(set(r.get("is_live","")      for r in pivot_records if r.get("is_live")      not in ("null","",None)))

    # Hour of day labels
    hour_data = []
    for h in range(24):
        label = f"{'0'+str(h) if h<10 else str(h)}:00"
        cnt   = acc.hour_counts.get(str(h), 0)
        hour_data.append({"hour": label, "impressions": cnt})

    return {
        "summary": {
            "total_impressions":      total,
            "unique_devices":         unique_devs,
            "avg_frequency":          round(total / unique_devs, 2) if unique_devs else 0,
            "total_exposure_seconds": acc.total_exp_secs,
            "streaming_impressions":  acc.ct_counts.get("Streaming", 0),
            "linear_impressions":     acc.ct_counts.get("Linear TV", 0),
            "date_range":             {"start": dates[0] if dates else "",
                                       "end":   dates[-1] if dates else ""},
            "generated_at":           pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "content_type_split":  top(acc.ct_counts),
        "device_split":        top(acc.dev_counts),
        "freq_distribution":   freq_dist,
        "freq_distribution_by_type": freq_dist_by_type,
        "freq_quintiles":      freq_quintiles,
        "freq_quintiles_by_type": freq_quintiles_by_type,
        "hour_of_day":         hour_data,
        "streaming_apps":      agg_list(acc.app_counts, n=10,
                                   exclude=("null","None","nan","","Linear TV","linear tv","Unknown")),
        "streaming_os":        agg_list(acc.os_counts,  n=8,
                                   exclude=("null","None","nan","")),
        "linear_networks":     agg_list(acc.net_counts, n=12,
                                   exclude=("null","None","nan","","Linear TV","linear tv")),
        "live_split":          top(acc.live_counts),
        "tv_input_types":      agg_list(acc.input_counts, n=8,
                                   exclude=("null","None","nan","")),
        "top_cities":          agg_list(acc.city_counts, n=15,
                                   exclude=("null","None","nan","")),
        "top_states":          agg_list(acc.state_counts, n=15,
                                   name_map=STATE_NAMES,
                                   exclude=("null","None","nan","")),
        "top_genres":          agg_list(acc.genre_counts, n=12,
                                   exclude=("null","None","nan","")),
        "top_titles":          agg_list(acc.title_counts, n=15,
                                   exclude=("null","None","nan","")),
        "filter_values": {
            "content_type":    fv_ct,
            "application":     fv_app,
            "device_category": fv_dev,
            "is_live":         fv_liv,
            "dates":           dates,
        },
        "overlap_analysis":    overlap_analysis,
        "daily_overlap":       daily_overlap,   # per-date true smba_id segments
        "aggregated": pivot_records,
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()
    print("=" * 62)
    print("  TV Intelligence Dashboard — Data Preparation")
    print("=" * 62)

    folder = Path(CSV_FOLDER)
    files  = sorted(folder.glob("*.csv")) + sorted(folder.glob("*.CSV"))
    if not files:
        print(f"\n❌  No CSV files found in {folder.resolve()}")
        print(f"    Create the folder and place your CSVs inside it.")
        raise SystemExit(1)

    total_size = sum(f.stat().st_size for f in files) / 1e9
    print(f"\n📂  {len(files)} file(s)  |  {total_size:.1f} GB total")
    for f in files:
        print(f"    • {f.name}  ({f.stat().st_size/1e6:.0f} MB)")

    acc             = Accumulator()
    pivot_col_names = []
    total_chunks    = 0

    for f in files:
        print(f"\n⏳  Processing {f.name} ...")
        file_rows = 0

        # Read only the columns that exist in this file
        file_cols = pd.read_csv(f, nrows=0).columns.tolist()
        use_cols  = [c for c in KEEP_COLS if c in file_cols]

        reader = pd.read_csv(f, usecols=use_cols, chunksize=CHUNK_SIZE,
                             low_memory=False)
        for i, chunk in enumerate(reader):
            pivot_col_names = process_chunk(chunk, acc)
            file_rows  += len(chunk)
            total_chunks += 1
            print(f"    chunk {i+1}  |  {file_rows:,} rows  |  "
                  f"pivot size: {len(acc.pivot):,}  |  "
                  f"elapsed: {time.time()-t0:.0f}s", end="\r")
            del chunk
            gc.collect()

        print(f"    ✓ {file_rows:,} rows processed")

    print(f"\n📊  Building output JSON ...")
    output = build_output(acc, pivot_col_names)

    s  = output["summary"]
    dr = s["date_range"]
    print(f"\n📈  Summary:")
    print(f"    Total impressions:  {s['total_impressions']:,}")
    print(f"    Unique devices:     {s['unique_devices']:,}")
    print(f"    Avg frequency:      {s['avg_frequency']}")
    print(f"    Exposure:           {s['total_exposure_seconds']/3600:.1f} hrs")
    print(f"    Date range:         {dr['start']} → {dr['end']}")
    print(f"    Pivot rows:         {len(output['aggregated']):,}")

    oa = output["overlap_analysis"]
    print(f"\n🔁  Overlap Analysis (true smba_id set-intersection):")
    print(f"    Streaming devices:  {oa['streaming_devices']:,}")
    print(f"    Linear devices:     {oa['linear_devices']:,}")
    print(f"    Streaming only:     {oa['streaming_only_devices']:,}  (not reachable via Linear)")
    print(f"    Linear only:        {oa['linear_only_devices']:,}  ← incremental reach from Linear")
    print(f"    Both (overlap):     {oa['both_devices']:,}  (cross-platform audience)")
    print(f"    Unduplicated total: {oa['unduplicated_total']:,}")
    print(f"    Daily overlap rows: {len(output['daily_overlap']):,}")

    print(f"\n💾  Writing {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, "w") as fout:
        json.dump(output, fout, separators=(",", ":"))

    kb = os.path.getsize(OUTPUT_FILE) / 1024
    elapsed = time.time() - t0
    print(f"✅  Done!  {kb:.0f} KB  ({kb/1024:.1f} MB)  in {elapsed:.0f}s")
    print(f"\n👉  Open the dashboard HTML and click '⬆ Load data'")
    print(f"    then select {OUTPUT_FILE}")
    print("=" * 62)
