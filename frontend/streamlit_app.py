from html import escape
from pathlib import Path
import subprocess
import sys

import altair as alt
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = ROOT / "data" / "production_log.csv"
REPORT_FILE = ROOT / "reports" / "drift_report.html"

# Design tokens: deep teal ink, brass accent, mint / coral for decisions
INK, PANEL, LINE = "#081214", "#0f1c1f", "#ffffff14"
TEXT, MUTED = "#eef3f1", "#8fa5a3"
BRASS, MINT, CORAL = "#d9b36a", "#4ade9c", "#ff7a6b"

st.set_page_config(page_title="LoanLens", page_icon="🏦", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Instrument+Sans:wght@400;500;600&display=swap');
    :root {{ --ink:{INK}; --panel:{PANEL}; --line:{LINE}; --text:{TEXT};
             --muted:{MUTED}; --brass:{BRASS}; --mint:{MINT}; --coral:{CORAL}; }}
    html, body, .stApp {{ font-family:'Instrument Sans',sans-serif; color:var(--text); }}
    .stApp {{ background: radial-gradient(900px 500px at 85% -10%, #12383a 0%, transparent 60%), var(--ink); }}
    #MainMenu, footer, [data-testid="stToolbar"] {{ visibility:hidden; }}
    .block-container {{ padding-top:2rem; max-width:1280px; }}
    h1,h2,h3,h4 {{ font-family:'Fraunces',serif !important; color:var(--text) !important; letter-spacing:-.01em; }}
    p, label, span, li {{ color:var(--text); }}
    [data-testid="stCaptionContainer"], .stCaption {{ color:var(--muted) !important; }}

    /* sidebar */
    [data-testid="stSidebar"] {{ background:#050c0d; border-right:1px solid var(--line); }}
    [data-testid="stSidebar"] button {{ background:var(--panel) !important; color:var(--text) !important;
        border:1px solid var(--line) !important; border-radius:10px !important; text-align:left;
        transition:border-color .15s, transform .15s; }}
    [data-testid="stSidebar"] button:hover {{ border-color:var(--brass) !important; }}

    /* inputs */
    [data-baseweb="input"], [data-baseweb="select"] > div, .stNumberInput input {{
        background:var(--panel) !important; border-color:var(--line) !important; color:var(--text) !important;
        border-radius:10px !important; }}
    [data-baseweb="input"]:focus-within, [data-baseweb="select"]:focus-within > div {{
        border-color:var(--brass) !important; box-shadow:0 0 0 1px var(--brass); }}
    [data-baseweb="popover"] ul {{ background:var(--panel) !important; }}
    [data-testid="stForm"] {{ background:#ffffff08; border:1px solid var(--line); border-radius:18px; padding:1.4rem; }}
    div[data-testid="stFormSubmitButton"] button {{ background:var(--brass) !important; color:#1a1305 !important;
        font-weight:600 !important; border:none !important; border-radius:12px !important; padding:.7rem 1rem; }}
    div[data-testid="stFormSubmitButton"] button:hover {{ filter:brightness(1.08); }}

    /* tabs */
    [data-baseweb="tab-list"] {{ gap:1.6rem; border-bottom:1px solid var(--line); }}
    [data-baseweb="tab"] {{ background:transparent !important; color:var(--muted) !important; font-weight:500; padding:.6rem 0; }}
    [aria-selected="true"] {{ color:var(--brass) !important; }}
    [data-baseweb="tab-highlight"] {{ background:var(--brass) !important; }}

    /* masthead */
    .mast {{ display:flex; justify-content:space-between; align-items:flex-end; gap:1rem;
        padding-bottom:1.4rem; margin-bottom:1.4rem; border-bottom:1px solid var(--line); flex-wrap:wrap; }}
    .mast h1 {{ font-size:2.8rem; margin:0; line-height:1; }}
    .mast p {{ margin:.5rem 0 0; color:var(--muted); max-width:52ch; }}
    .pill {{ display:inline-flex; align-items:center; gap:.5rem; padding:.45rem .9rem; border-radius:999px;
        border:1px solid var(--line); background:#ffffff08; font-size:.9rem; }}
    .dot {{ width:8px; height:8px; border-radius:50%; }}
    .dot.on {{ background:var(--mint); box-shadow:0 0 0 0 #4ade9c99; animation:ping 2s infinite; }}
    .dot.off {{ background:var(--coral); }}
    @keyframes ping {{ 70% {{ box-shadow:0 0 0 8px #4ade9c00; }} 100% {{ box-shadow:0 0 0 0 #4ade9c00; }} }}

    /* kpis */
    .kpi {{ background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:1.1rem 1.3rem;
        border-top:2px solid var(--accent, var(--brass)); height:100%; }}
    .kpi .l {{ color:var(--muted); font-size:.9rem; }}
    .kpi .v {{ font-family:'Fraunces',serif; font-size:2.1rem; font-weight:700; margin-top:.2rem; }}
    .kpi .s {{ color:var(--muted); font-size:.82rem; margin-top:.15rem; }}

    /* verdict (the one memorable element) */
    .verdict {{ display:flex; gap:1.8rem; align-items:center; background:var(--panel);
        border:1px solid var(--line); border-radius:22px; padding:1.6rem; }}
    .dial {{ --p:0; --c:var(--mint); width:170px; height:170px; flex:none; border-radius:50%;
        background:conic-gradient(var(--c) calc(var(--p) * 1%), #ffffff12 0);
        display:grid; place-items:center; transition:all .6s; }}
    .dial > div {{ width:136px; height:136px; border-radius:50%; background:var(--panel); display:grid;
        place-content:center; text-align:center; }}
    .dial b {{ font-family:'Fraunces',serif; font-size:2.1rem; }}
    .dial small {{ color:var(--muted); }}
    .verdict h2 {{ margin:0; font-size:2.2rem; }}
    .verdict.ok h2 {{ color:var(--mint) !important; }}
    .verdict.no h2 {{ color:var(--coral) !important; }}
    .verdict p {{ margin:.4rem 0 0; color:var(--muted); }}
    .driver {{ background:#ffffff08; border:1px solid var(--line); border-left:3px solid var(--brass);
        border-radius:12px; padding:.8rem 1rem; margin-bottom:.6rem; }}
    .chip {{ background:#ffffff0a; border:1px solid var(--line); border-radius:12px; padding:.7rem 1rem; }}
    .chip .l {{ color:var(--muted); font-size:.82rem; }} .chip .v {{ font-weight:600; font-size:1.15rem; }}
    .step {{ background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:1rem 1.2rem; height:100%; }}
    .step b {{ color:var(--brass); }} .step p {{ color:var(--muted); margin:.3rem 0 0; font-size:.9rem; }}
    @media (max-width:700px) {{ .verdict {{ flex-direction:column; text-align:center; }} .mast h1 {{ font-size:2.1rem; }} }}
    @media (prefers-reduced-motion:reduce) {{ .dot.on, .dial {{ animation:none; transition:none; }} }}
    </style>
    """,
    unsafe_allow_html=True,
)


def api_call(api_url, endpoint, method="get", payload=None):
    try:
        response = requests.request(method, f"{api_url.rstrip('/')}{endpoint}",
                                    json=payload, timeout=15)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, f"API unavailable: {exc}"


def read_log():
    if not LOG_FILE.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(LOG_FILE)
    except pd.errors.ParserError:
        return pd.DataFrame()


def run_local_command(args):
    result = subprocess.run(
        [sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=180
    )
    return result.returncode, result.stdout + result.stderr


def kpi(label, value, sub="", accent=BRASS):
    return (f'<div class="kpi" style="--accent:{accent}"><div class="l">{label}</div>'
            f'<div class="v">{value}</div><div class="s">{sub}</div></div>')


def chip(label, value):
    return f'<div class="chip"><div class="l">{label}</div><div class="v">{value}</div></div>'


def chart_style(chart):
    return (chart.configure(background="transparent")
            .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor="#ffffff10",
                            domainColor="#ffffff20", tickColor="#ffffff20")
            .configure_legend(labelColor=TEXT, titleColor=MUTED)
            .configure_view(strokeWidth=0))


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Control room")
    api_url = st.text_input("API address", "http://localhost:8000")
    st.caption("Start the API with: uvicorn app.main:app --reload")
    health, health_error = api_call(api_url, "/health")
    if health:
        st.success(f"API {health.get('status', 'unknown')}")
        st.caption(f"Alias: {health.get('alias', 'unknown')} · "
                   f"Version: {health.get('model_version', 'unknown')}")
    else:
        st.error("API offline")
        st.caption(health_error)
    st.divider()
    st.markdown("### Actions")
    if st.button("🔄 Refresh data", use_container_width=True):
        st.rerun()
    if st.button("🧪 Simulate drift", use_container_width=True):
        with st.spinner("Generating drifted traffic…"):
            code, output = run_local_command(["scripts/simulate_drift.py"])
        st.session_state.update(command_output=output, command_ok=code == 0)
    if st.button("📊 Check for drift", use_container_width=True):
        with st.spinner("Running Evidently…"):
            code, output = run_local_command(["-m", "src.drift"])
        st.session_state.update(command_output=output, command_ok=code in (0, 2))
    if st.button("🚀 Retrain model", use_container_width=True):
        with st.spinner("Retraining and registering…"):
            code, output = run_local_command(["-m", "src.retrain"])
        st.session_state.update(command_output=output, command_ok=code == 0)

# ---------------------------------------------------------------- data
log = read_log()
model_info, model_error = api_call(api_url, "/model-info")
count = len(log)
approved_count = int((log["prediction"] == "Approved").sum()) if "prediction" in log else 0
rejected_count = count - approved_count
approval_rate = approved_count / count if count else 0
status_html = ('<span class="dot on"></span>API online' if health
               else '<span class="dot off"></span>API offline')

st.markdown(
    f'<div class="mast"><div><h1>LoanLens</h1>'
    f'<p>Check a loan application, see why the model decided, and watch for drift in live traffic.</p></div>'
    f'<span class="pill">{status_html}</span></div>',
    unsafe_allow_html=True,
)

cols = st.columns(4)
cols[0].markdown(kpi("Predictions made", f"{count:,}", "all time"), unsafe_allow_html=True)
cols[1].markdown(kpi("Approved", f"{approved_count:,}", f"{rejected_count:,} rejected", MINT),
                 unsafe_allow_html=True)
cols[2].markdown(kpi("Approval rate", f"{approval_rate:.1%}", "share of approved loans", BRASS),
                 unsafe_allow_html=True)
cols[3].markdown(kpi("Live model", model_info.get("model", "XGBoost") if model_info else "XGBoost",
                     f"version {health.get('model_version', '–')}" if health else "API offline", CORAL),
                 unsafe_allow_html=True)
st.write("")

predict_tab, history_tab, monitor_tab, about_tab = st.tabs(
    ["Check an applicant", "Prediction history", "Drift & health", "Model details"]
)

# ---------------------------------------------------------------- predict
with predict_tab:
    with st.form("loan_form", clear_on_submit=False):
        left, middle, right = st.columns(3)
        with left:
            applicant_income = st.number_input("Applicant income ($/month)", min_value=1.0,
                                               value=6500.0, step=500.0)
            coapplicant_income = st.number_input("Co-applicant income ($/month)", min_value=0.0,
                                                 value=2000.0, step=500.0)
            loan_amount = st.number_input("Loan amount ($ thousands)", min_value=1.0,
                                          value=145.0, step=10.0)
            term = st.selectbox("Loan term (months)", [180.0, 240.0, 360.0, 480.0], index=2)
        with middle:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        with right:
            credit_history = st.selectbox(
                "Credit history", [1.0, 0.0],
                format_func=lambda v: "Good (1.0)" if v else "Poor or defaulted (0.0)")
            self_employed = st.selectbox("Self-employed", ["No", "Yes"])
            property_area = st.selectbox("Property area", ["Semiurban", "Urban", "Rural"])
            submitted = st.form_submit_button("Get decision", use_container_width=True)

    if submitted:
        payload = {
            "Gender": gender, "Married": married, "Dependents": dependents,
            "Education": education, "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income, "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount, "Loan_Amount_Term": term,
            "Credit_History": credit_history, "Property_Area": property_area,
        }
        with st.spinner("Scoring application…"):
            result, error = api_call(api_url, "/predict", "post", payload)
        if error:
            st.error(f"Couldn't get a decision. {error}")
        else:
            st.session_state["last_result"] = result
            st.session_state["last_inputs"] = (applicant_income, coapplicant_income,
                                               loan_amount, term)

    result = st.session_state.get("last_result")
    if not result:
        st.info("Fill in the applicant details above and select Get decision.")
    else:
        probability = float(result["probability"])
        approved = result["prediction"] == "Approved"
        color = MINT if approved else CORAL
        st.write("")
        v_col, d_col = st.columns([1.15, 1], gap="large")
        with v_col:
            st.markdown(
                f'<div class="verdict {"ok" if approved else "no"}">'
                f'<div class="dial" style="--p:{probability * 100:.1f};--c:{color}">'
                f'<div><b>{probability:.0%}</b><small>approval chance</small></div></div>'
                f'<div><h2>{escape(str(result["prediction"]))}</h2>'
                f'<p>The model puts the chance of approval at {probability:.1%}.</p></div></div>',
                unsafe_allow_html=True,
            )
            inc, co, amt, months = st.session_state["last_inputs"]
            total = inc + co
            st.write("")
            c1, c2, c3 = st.columns(3)
            c1.markdown(chip("Household income", f"${total:,.0f}/mo"), unsafe_allow_html=True)
            c2.markdown(chip("Loan vs. yearly income", f"{amt * 1000 / (total * 12):.0%}"),
                        unsafe_allow_html=True)
            c3.markdown(chip("Repayment period", f"{months / 12:.0f} years"), unsafe_allow_html=True)
        with d_col:
            st.markdown("#### What drove this decision")
            reasons = result.get("reasons", [])
            if reasons:
                for reason in reasons:
                    st.markdown(f'<div class="driver">{escape(str(reason))}</div>',
                                unsafe_allow_html=True)
            else:
                st.caption("The API returned no explanation for this decision.")

# ---------------------------------------------------------------- history
with history_tab:
    if log.empty:
        st.info("No predictions yet. Get a decision in the first tab and it will show up here.")
    else:
        left, right = st.columns([1.6, 1], gap="large")
        with left:
            st.markdown("#### Recent predictions")
            display_columns = [c for c in ["timestamp", "prediction", "probability",
                                           "ApplicantIncome", "LoanAmount", "Credit_History",
                                           "Property_Area"] if c in log.columns]
            view = log[display_columns].tail(100).iloc[::-1]
            if "prediction" in view:
                choice = st.radio("Show", ["All", "Approved", "Rejected"], horizontal=True,
                                  label_visibility="collapsed")
                if choice != "All":
                    view = view[view["prediction"] == choice]
            column_config = {}
            if "probability" in view:
                column_config["probability"] = st.column_config.ProgressColumn(
                    "Approval chance", min_value=0.0, max_value=1.0, format="%.0f%%")
            st.dataframe(view, use_container_width=True, hide_index=True,
                         column_config=column_config, height=380)
            st.download_button("Download full log (CSV)", log.to_csv(index=False).encode(),
                               "production_log.csv", "text/csv")
        with right:
            if "prediction" in log:
                st.markdown("#### Decision split")
                split = log["prediction"].value_counts().rename_axis("decision").reset_index(name="count")
                donut = alt.Chart(split).mark_arc(innerRadius=62, stroke=INK, strokeWidth=3).encode(
                    theta="count:Q",
                    color=alt.Color("decision:N", legend=alt.Legend(orient="bottom", title=None),
                                    scale=alt.Scale(domain=["Approved", "Rejected"],
                                                    range=[MINT, CORAL])),
                    tooltip=["decision", "count"],
                ).properties(height=300)
                st.altair_chart(chart_style(donut), use_container_width=True)
        if {"probability"}.issubset(log.columns):
            st.markdown("#### Approval chance over the last 100 predictions")
            trend = log.tail(100).reset_index(drop=True).reset_index(names="n")
            line = alt.Chart(trend).mark_area(
                line={"color": BRASS}, color=alt.Gradient(
                    gradient="linear", stops=[alt.GradientStop(color="#d9b36a55", offset=0),
                                              alt.GradientStop(color="#d9b36a00", offset=1)],
                    x1=1, x2=1, y1=1, y2=0)
            ).encode(x=alt.X("n:Q", title="Prediction"),
                     y=alt.Y("probability:Q", title="Approval chance",
                             scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                     tooltip=["n", alt.Tooltip("probability:Q", format=".1%")]).properties(height=220)
            st.altair_chart(chart_style(line), use_container_width=True)

# ---------------------------------------------------------------- monitor
with monitor_tab:
    st.markdown("#### Drift & health")
    st.write("Compares the data the model was trained on with what it sees in production.")
    ready = count >= 20
    m = st.columns(3)
    m[0].markdown(kpi("Production batch", f"{count:,}",
                      "ready to check" if ready else f"{20 - count} more needed",
                      MINT if ready else CORAL), unsafe_allow_html=True)
    m[1].markdown(kpi("Minimum batch size", "20", "predictions needed for a check"),
                  unsafe_allow_html=True)
    m[2].markdown(kpi("Drift sensitivity", "30%", "share of drifted features that triggers an alert"),
                  unsafe_allow_html=True)
    st.write("")
    if st.session_state.get("command_output"):
        if st.session_state.get("command_ok"):
            st.success("Last action finished")
        else:
            st.error("Last action failed. Check the output below.")
        with st.expander("Action output", expanded=not st.session_state.get("command_ok")):
            st.code(st.session_state["command_output"])
    if REPORT_FILE.exists():
        st.download_button("Download drift report (HTML)", REPORT_FILE.read_bytes(),
                           "drift_report.html", "text/html")
    else:
        st.caption("No drift report yet.")
    st.caption("To try it: select Simulate drift in the sidebar, then Check for drift.")

# ---------------------------------------------------------------- about
with about_tab:
    st.markdown("#### How it works")
    steps = [("Validate", "Incoming data is checked before use."),
             ("Preprocess", "Features are cleaned and encoded."),
             ("Train", "The model is trained and tracked."),
             ("Register", "MLflow stores the champion version."),
             ("Serve", "FastAPI returns a decision and its top reasons.")]
    for col, (name, desc) in zip(st.columns(len(steps)), steps):
        col.markdown(f'<div class="step"><b>{name}</b><p>{desc}</p></div>', unsafe_allow_html=True)
    st.write("")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**Explainability.** The three biggest SHAP feature contributions are shown with each decision.")
        st.markdown("**Monitoring.** KS-test and chi-square checks run against training data and produce an HTML report.")
        st.markdown("**Fallback.** If the registry is unreachable, `models/best_model.joblib` keeps the API working.")
    with b:
        if model_info:
            with st.expander("Registry metadata", expanded=True):
                st.json(model_info)
        elif model_error:
            st.error(f"Couldn't load model details. {model_error}")