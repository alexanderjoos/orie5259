import json
from pathlib import Path


SRC = Path("pbv_v5_causal.ipynb")
DST = Path("pbv_v5_causal_true_twap.ipynb")


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise ValueError(f"Expected snippet not found:\n{old}")
    return text.replace(old, new, 1)


nb = json.loads(SRC.read_text())

# Clear stale outputs because the benchmark definition changes.
for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


cell0 = "".join(nb["cells"][0]["source"])
cell0 = replace_once(cell0, "# PBV v5 — Causal Variant", "# PBV v5 — Causal Variant (True TWAP)")
cell0 = replace_once(
    cell0,
    "- retain the `v3` rule baseline as an interpretable comparator",
    "- replace the minute-open benchmark with **true in-window TWAP** over the eligible execution window\n- retain the `v3` rule baseline as an interpretable comparator",
)
nb["cells"][0]["source"] = cell0.splitlines(keepends=True)


cell2 = "".join(nb["cells"][2]["source"])
cell2 = replace_once(
    cell2,
    '    d["tick_in_minute"] = d.groupby("Minute").cumcount()\n\n    mid = d["MidPrice"]\n',
    '    d["tick_in_minute"] = d.groupby("Minute").cumcount()\n\n    # Eligible real-time execution window\n    d["eligible"] = (d["tick_in_minute"] >= WARMUP_TICKS) & (d["elapsed"] <= DEADLINE_S)\n\n    mid = d["MidPrice"]\n',
)
cell2 = replace_once(
    cell2,
    '    # Minute-open benchmark and current-tick realized improvement\n    twap_ref = grp[EXEC_PRICE_COL].transform("first")\n    d["twap_ref"] = twap_ref\n    d["realized_improvement"] = (twap_ref - d[EXEC_PRICE_COL]) if SIDE == "BUY" else (d[EXEC_PRICE_COL] - twap_ref)\n\n    # Eligible real-time execution window\n    d["eligible"] = (d["tick_in_minute"] >= WARMUP_TICKS) & (d["elapsed"] <= DEADLINE_S)\n',
    '    # True in-window TWAP benchmark and current-tick realized improvement\n    twap_true = d.loc[d["eligible"]].groupby("Minute")[EXEC_PRICE_COL].mean()\n    d["twap_ref"] = d["Minute"].map(twap_true)\n    d["realized_improvement"] = (d["twap_ref"] - d[EXEC_PRICE_COL]) if SIDE == "BUY" else (d[EXEC_PRICE_COL] - d["twap_ref"])\n',
)
nb["cells"][2]["source"] = cell2.splitlines(keepends=True)


cell7 = "".join(nb["cells"][7]["source"])
cell7 = cell7.replace("% minutes beating TWAP", "% minutes beating true TWAP")
cell7 = cell7.replace("Mean improvement —", "Mean improvement vs true TWAP —")
nb["cells"][7]["source"] = cell7.splitlines(keepends=True)


cell9 = "".join(nb["cells"][9]["source"])
cell9 = cell9.replace("Mean improvement vs TWAP", "Mean improvement vs true TWAP")
cell9 = cell9.replace("% minutes beating TWAP", "% minutes beating true TWAP")
nb["cells"][9]["source"] = cell9.splitlines(keepends=True)


cell10 = "".join(nb["cells"][10]["source"])
cell10 = replace_once(
    cell10,
    "- The `v3` baseline remains causal and serves as the hand-crafted comparator.",
    "- The benchmark is **true in-window TWAP** over the eligible execution window rather than the minute-open quote.\n- The `v3` baseline remains causal and serves as the hand-crafted comparator.",
)
nb["cells"][10]["source"] = cell10.splitlines(keepends=True)


DST.write_text(json.dumps(nb, indent=1))
