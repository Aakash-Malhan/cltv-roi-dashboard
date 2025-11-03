import io, json, os
import numpy as np
import pandas as pd
import plotly.express as px
import gradio as gr

# Config
DATAFILE = os.getenv("DEFAULT_DATAFILE", "customer_acquisition_data.csv")
REQUIRED_COLS = ["customer_id", "channel", "cost", "conversion_rate", "revenue"]

def _ensure_cols(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nExpected: {REQUIRED_COLS}")

def _compute_metrics(df: pd.DataFrame):
    _ensure_cols(df)
    df = df.copy()
    df["roi"]  = df["revenue"] / df["cost"]
    df["cltv"] = ((df["revenue"] - df["cost"]) * df["conversion_rate"]) / df["cost"]

    by_channel = (
        df.groupby("channel")
          .agg(
              customers=("customer_id","count"),
              avg_cost=("cost","mean"),
              avg_conv_rate=("conversion_rate","mean"),
              total_revenue=("revenue","sum"),
              avg_roi=("roi","mean"),
              avg_cltv=("cltv","mean"),
          )
          .sort_values("avg_cltv", ascending=False)
          .reset_index()
    )
    by_channel["revenue_share_%"] = (
        100 * by_channel["total_revenue"] / by_channel["total_revenue"].sum()
    )
    meta = {
        "rows": int(len(df)),
        "channels": int(df["channel"].nunique()),
        "current_weighted_roi": float(df["roi"].mean()),
        "current_avg_cltv": float(df["cltv"].mean()),
    }
    return df, by_channel, meta

def load_builtin():
    if not os.path.exists(DATAFILE):
        raise gr.Error(
            f"Default dataset '{DATAFILE}' not found in repo root.\n"
            "Upload the CSV to your Space or set env var DEFAULT_DATAFILE."
        )
    df = pd.read_csv(DATAFILE)
    return _compute_metrics(df)

def load_uploaded(file):
    if file is None:
        return load_builtin()
    df = pd.read_csv(file.name if hasattr(file, "name") else file)
    return _compute_metrics(df)

# Plots
def hist(df, col, title): return px.histogram(df, x=col, nbins=20, title=title)
def bar(df, x, y, title): return px.bar(df, x=x, y=y, title=title)
def pie(df): return px.pie(df, values="total_revenue", names="channel", title="Total Revenue by Channel", hole=0.6)
def cltv_box(df): return px.box(df, x="channel", y="cltv", title="CLTV Distribution by Channel")

# Simulator
def simulate_reallocation(by_df: pd.DataFrame, allocations_json: str):
    try:
        alloc = json.loads(allocations_json or "{}")
    except Exception:
        return "❌ Invalid JSON.", None
    if by_df is None or by_df.empty:
        return "Load data first.", None

    # normalize to 1.0
    s = sum(alloc.values()) or 1.0
    weights = {k: v/s for k, v in alloc.items()}

    kpis = by_df.set_index("channel")[["avg_roi","avg_cltv"]]
    for ch in kpis.index:
        weights.setdefault(ch, 0.0)

    new_roi  = sum(kpis.loc[ch, "avg_roi"]  * w for ch, w in weights.items())
    new_cltv = sum(kpis.loc[ch, "avg_cltv"] * w for ch, w in weights.items())

    cur_w = (by_df.set_index("channel")["revenue_share_%"] / 100.0).to_dict()
    cur_roi  = sum(kpis.loc[ch, "avg_roi"]  * cur_w.get(ch,0) for ch in kpis.index)
    cur_cltv = sum(kpis.loc[ch, "avg_cltv"] * cur_w.get(ch,0) for ch in kpis.index)

    delta = {
        "current_weighted_roi": round(cur_roi, 2),
        "new_weighted_roi": round(new_roi, 2),
        "roi_change_%": round(100 * (new_roi - cur_roi) / (cur_roi + 1e-9), 2),
        "current_weighted_cltv": round(cur_cltv, 2),
        "new_weighted_cltv": round(new_cltv, 2),
        "cltv_change_%": round(100 * (new_cltv - cur_cltv) / (cur_cltv + 1e-9), 2),
    }
    text = (
        f"**ROI**: {delta['current_weighted_roi']:.2f} ➜ {delta['new_weighted_roi']:.2f} "
        f"(**{delta['roi_change_%']:+.2f}%**)\n\n"
        f"**CLTV**: {delta['current_weighted_cltv']:.2f} ➜ {delta['new_weighted_cltv']:.2f} "
        f"(**{delta['cltv_change_%']:+.2f}%**)"
    )
    return text, delta

# Gradio UI
with gr.Blocks(title="CLTV & ROI Dashboard", css="footer {visibility: hidden;}") as demo:
    gr.Markdown(
        "## Customer Lifetime Value (CLTV) & ROI Dashboard\n"
        f"- Default dataset: **{DATAFILE}** (auto-loaded)\n"
        "- Or upload your own CSV with columns: "
        "`customer_id, channel, cost, conversion_rate, revenue`."
    )

    state_df = gr.State()
    state_by = gr.State()
    state_meta = gr.State()

    with gr.Row():
        file_in = gr.File(file_count="single", label="Upload CSV")
        load_btn = gr.Button("Load Uploaded CSV", variant="primary")
        builtin_btn = gr.Button("Use Built-in Dataset")

    status = gr.Markdown()

    # Overview tab
    with gr.Tab("Overview"):
        meta_json = gr.JSON(label="Overall Stats")
        with gr.Row():
            fig_cost_hist = gr.Plot(label="Cost Histogram")
            fig_rev_hist  = gr.Plot(label="Revenue Histogram")

    # Channel tab
    with gr.Tab("Channel Analysis"):
        by_table = gr.Dataframe(label="Channel Summary", interactive=False, wrap=True)
        with gr.Row():
            fig_cost_ch = gr.Plot()
            fig_conv_ch = gr.Plot()
        with gr.Row():
            fig_roi_ch  = gr.Plot()
            fig_cltv_ch = gr.Plot()
        fig_rev_share = gr.Plot()
        fig_cltv_boxp = gr.Plot()

    # Simulator tab
    with gr.Tab("Reallocation Simulator"):
        gr.Markdown("Provide target allocation JSON (must sum ~100). "
                    "Example: `{ \"email marketing\": 20, \"paid advertising\": 10, \"referral\": 35, \"social media\": 35 }`")
        alloc_in = gr.Textbox(
            value='{ "email marketing": 20, "paid advertising": 10, "referral": 35, "social media": 35 }',
            lines=3, label="Allocation JSON"
        )
        sim_btn = gr.Button("Simulate", variant="primary")
        sim_out = gr.Markdown()
        sim_json = gr.JSON(label="Simulation Details")

    # Download tab
    with gr.Tab("Download"):
        dl1 = gr.File(label="Customer-level CSV (with metrics)", interactive=False)
        dl2 = gr.File(label="Channel summary CSV", interactive=False)
        gen_btn = gr.Button("Generate CSVs")

    # Callbacks
    def _render(df, by, meta, msg_prefix):
        state = (df, by, meta)
        status_text = f"{msg_prefix} • rows: {meta['rows']} • channels: {meta['channels']}"
        return (
            *state, status_text, meta,
            hist(df, "cost", "Distribution of Acquisition Cost"),
            hist(df, "revenue", "Distribution of Revenue"),
            bar(by, "channel", "avg_cost", "Customer Acquisition Cost by Channel"),
            bar(by, "channel", "avg_conv_rate", "Conversion Rate by Channel"),
            bar(by, "channel", "avg_roi", "Return on Investment (ROI) by Channel"),
            bar(by, "channel", "avg_cltv", "Customer Lifetime Value (CLTV) by Channel"),
            pie(by), cltv_box(df),
            by  # for table
        )

    def on_load_uploaded(file):
        df, by, meta = load_uploaded(file)
        return _render(df, by, meta, "Uploaded CSV loaded")

    def on_load_builtin():
        df, by, meta = load_builtin()
        return _render(df, by, meta, "Built-in dataset loaded")

    load_btn.click(on_load_uploaded, inputs=[file_in],
                   outputs=[state_df, state_by, state_meta, status,
                            meta_json, fig_cost_hist, fig_rev_hist,
                            fig_cost_ch, fig_conv_ch, fig_roi_ch, fig_cltv_ch,
                            fig_rev_share, fig_cltv_boxp, by_table])

    builtin_btn.click(on_load_builtin, inputs=[],
                      outputs=[state_df, state_by, state_meta, status,
                               meta_json, fig_cost_hist, fig_rev_hist,
                               fig_cost_ch, fig_conv_ch, fig_roi_ch, fig_cltv_ch,
                               fig_rev_share, fig_cltv_boxp, by_table])

    def _simulate(alloc_json, by):
        txt, delta = simulate_reallocation(by, alloc_json)
        return txt, delta

    sim_btn.click(_simulate, inputs=[alloc_in, state_by], outputs=[sim_out, sim_json])

    def _downloads(df, by):
        if df is None or by is None:
            raise gr.Error("Load data first.")
        p1, p2 = "/tmp/customer_with_metrics.csv", "/tmp/channel_summary.csv"
        pd.DataFrame(df).to_csv(p1, index=False)
        pd.DataFrame(by).to_csv(p2, index=False)
        return p1, p2

    gen_btn.click(_downloads, inputs=[state_df, state_by], outputs=[dl1, dl2])

# Auto-load built-in on startup in Spaces
if __name__ == "__main__":
    # If running locally: open and load default dataset
    demo.queue().launch()
