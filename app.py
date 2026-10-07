import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import pipeline as P

st.set_page_config(page_title="Employee Salary Intelligence", page_icon="💼", layout="wide")
BLUE, ORANGE = "#2F6FDE", "#F28E2B"
COLORS = {"<=50K": BLUE, ">50K": ORANGE}


# ----------------------------------------------------------------------------- loading
@st.cache_resource(show_spinner="First run: training models on the real dataset...")
def load_artifacts():
    if not (P.ART / "meta.json").exists() or not (P.ART / "best_model.joblib").exists():
        P.train()
    return json.loads((P.ART / "meta.json").read_text()), joblib.load(P.ART / "best_model.joblib")


@st.cache_data
def load_data():
    if not P.CLEAN.exists():
        load_artifacts()
    return pd.read_csv(P.CLEAN)


meta, model = load_artifacts()
df = load_data()
df["high"] = (df["income"] == ">50K").astype(int)
BEST, RES = meta["best_model"], meta["results"]
AUD = meta["audit"]


def rate_by(col, order=None, min_n=30):
    g = df.groupby(col).agg(n=("high", "size"), rate=("high", "mean")).reset_index()
    g = g[g.n >= min_n]
    g["rate"] *= 100
    return g.sort_values("rate") if order is None else g.set_index(col).loc[order].reset_index()


def rate_bar(col, title, order=None, horizontal=True):
    g = rate_by(col, order)
    fig = px.bar(g, x="rate" if horizontal else col, y=col if horizontal else "rate", orientation="h" if horizontal else "v",
                 hover_data={"n": True, "rate": ":.1f"}, color="rate", color_continuous_scale="Blues",
                 labels={"rate": "% earning >50K"}, title=title)
    fig.update_layout(coloraxis_showscale=False, height=460)
    return fig


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("💼 Salary Intelligence")
page = st.sidebar.radio("Navigate", ["Overview & Data Audit", "Exploratory Analysis", "Model Comparison",
                                     "Live Prediction", "Insights"])
st.sidebar.caption(f"Dataset: UCI Adult Census Income\n\nSelected model: **{BEST}**")

# ----------------------------------------------------------------------------- overview
if page == "Overview & Data Audit":
    st.title("Employee Salary Intelligence & Classification System")
    st.caption("Classifying whether an employee earns **<=50K** or **>50K** from demographic, education and work data.")
    c = st.columns(5)
    c[0].metric("Raw records", f"{AUD['rows']:,}")
    c[1].metric("Columns", AUD["columns"])
    c[2].metric("Duplicates removed", AUD["duplicates"])
    c[3].metric("Unknown ('?') cells", f"{sum(AUD['unknown_values'].values()):,}")
    c[4].metric("Clean records", f"{meta['clean_rows']:,}")
    st.markdown("**Workflow:** Collection → Understanding → Cleaning → EDA → Preprocessing → Encoding → Split → "
                "Training → Evaluation → Selection → Prediction → Explainability → Dashboard → Insights")
    a, b = st.columns(2)
    with a:
        st.subheader("Structure & data types")
        st.dataframe(pd.DataFrame({"dtype": AUD["dtypes"],
                                   "unknown '?'": {k: AUD["unknown_values"].get(k, 0) for k in AUD["dtypes"]}}),
                     width="stretch", height=360)
    with b:
        st.subheader("Target class distribution (raw)")
        cd = pd.Series(AUD["class_distribution"])
        fig = px.pie(values=cd.values, names=cd.index, hole=.5, color=cd.index, color_discrete_map=COLORS)
        fig.update_layout(height=360)
        st.plotly_chart(fig, width="stretch")
    st.subheader("Numerical statistics (raw)")
    st.dataframe(pd.DataFrame(AUD["describe"]).T, width="stretch")
    with st.expander("Unique values of every categorical column"):
        for k, v in AUD["unique_categories"].items():
            st.markdown(f"**{k}** ({len(v)}): " + ", ".join(v))
    st.subheader("Preprocessing decisions")
    for line in meta["preprocessing_log"]:
        st.markdown(f"- {line}")
    st.info("The data are a 1994 US Census extract: salaries are in 1994 dollars and patterns reflect that era.")

# ----------------------------------------------------------------------------- EDA
elif page == "Exploratory Analysis":
    st.title("Exploratory Data Analysis")
    st.caption(f"Based on the {len(df):,} cleaned records. Hover over charts for exact values.")
    tabs = st.tabs(["Income & Age", "Education", "Occupation & Workclass", "Hours & Gender", "Correlations", "Group explorer"])
    with tabs[0]:
        a, b = st.columns(2)
        vc = df["income"].value_counts().reset_index()
        a.plotly_chart(px.bar(vc, x="income", y="count", color="income", color_discrete_map=COLORS,
                              title="Income distribution", text_auto=True), width="stretch")
        b.plotly_chart(px.histogram(df, x="age", color="income", nbins=40, barmode="overlay", opacity=.7,
                                    color_discrete_map=COLORS, title="Age distribution by income"), width="stretch")
        a, b = st.columns(2)
        a.plotly_chart(px.box(df, x="income", y="age", color="income", color_discrete_map=COLORS,
                              title="Age vs income"), width="stretch")
        df["age band"] = pd.cut(df["age"], [16, 24, 34, 44, 54, 64, 100],
                                labels=["17-24", "25-34", "35-44", "45-54", "55-64", "65+"])
        g = df.groupby("age band", observed=True)["high"].mean().mul(100).reset_index()
        b.plotly_chart(px.line(g, x="age band", y="high", markers=True, title="% earning >50K by age band",
                               labels={"high": "% >50K"}), width="stretch")
    with tabs[1]:
        order = df.groupby("education")["education-num"].first().sort_values().index.tolist()
        st.plotly_chart(rate_bar("education", "% earning >50K by education level (lowest → highest)",
                                 order=order, horizontal=False), width="stretch")
        ct = pd.crosstab(df["education"], df["income"]).loc[order].reset_index().melt("education", var_name="income", value_name="count")
        st.plotly_chart(px.bar(ct, x="education", y="count", color="income", barmode="group",
                               color_discrete_map=COLORS, title="Headcount by education and income"), width="stretch")
    with tabs[2]:
        a, b = st.columns(2)
        a.plotly_chart(rate_bar("occupation", "% earning >50K by occupation"), width="stretch")
        b.plotly_chart(rate_bar("workclass", "% earning >50K by workclass"), width="stretch")
    with tabs[3]:
        a, b = st.columns(2)
        a.plotly_chart(px.histogram(df, x="hours-per-week", color="income", nbins=50, barmode="overlay", opacity=.7,
                                    color_discrete_map=COLORS, title="Hours per week by income"), width="stretch")
        df["hours band"] = pd.cut(df["hours-per-week"], [0, 20, 39, 40, 50, 60, 100],
                                  labels=["≤20", "21-39", "40", "41-50", "51-60", "60+"])
        g = df.groupby("hours band", observed=True)["high"].mean().mul(100).reset_index()
        b.plotly_chart(px.bar(g, x="hours band", y="high", title="% earning >50K by weekly hours",
                              labels={"high": "% >50K"}, color="high", color_continuous_scale="Blues"), width="stretch")
        a, b = st.columns(2)
        sx = pd.crosstab(df["sex"], df["income"], normalize="index").mul(100).reset_index().melt("sex", value_name="pct")
        a.plotly_chart(px.bar(sx, x="sex", y="pct", color="income", color_discrete_map=COLORS, text_auto=".1f",
                              title="Gender vs income (% within gender)"), width="stretch")
        g = df.groupby(["sex", "education"])["high"].mean().mul(100).reset_index()
        b.plotly_chart(px.bar(df.groupby(["sex", "marital-status"])["high"].mean().mul(100).reset_index(),
                              x="marital-status", y="high", color="sex", barmode="group",
                              title="% >50K by marital status and gender", labels={"high": "% >50K"}), width="stretch")
    with tabs[4]:
        num = df[["age", "education-num", "capital-gain", "capital-loss", "hours-per-week", "high"]].rename(columns={"high": "income >50K (1/0)"})
        corr = num.corr().round(2)
        st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                  title="Correlation matrix of numerical variables"), width="stretch")
        top = corr["income >50K (1/0)"].drop("income >50K (1/0)").abs().idxmax()
        st.caption(f"Strongest linear relationship with high income: **{top}** "
                   f"(r = {corr.loc[top, 'income >50K (1/0)']:.2f}). Capital gain/loss are highly skewed, so Pearson r understates their effect.")
    with tabs[5]:
        cats = P.CAT
        c1, c2 = st.columns(2)
        r = c1.selectbox("Rows", cats, index=cats.index("occupation"))
        cl = c2.selectbox("Columns", [c for c in cats if c != r], index=0)
        top_n = lambda s: s[s.isin(s.value_counts().head(12).index)]
        d = df[df[r].isin(top_n(df[r])) & df[cl].isin(top_n(df[cl]))]
        pv = d.pivot_table(index=r, columns=cl, values="high", aggfunc="mean").mul(100).round(1)
        st.plotly_chart(px.imshow(pv, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                                  title=f"% earning >50K by {r} × {cl}"), width="stretch")

# ----------------------------------------------------------------------------- models
elif page == "Model Comparison":
    st.title("Model Training & Evaluation")
    st.caption(f"Stratified 80/20 split (seed {meta['split']['seed']}): {meta['split']['train']:,} training / "
               f"{meta['split']['test']:,} test rows. Metrics below are on the held-out test set.")
    t = pd.DataFrame(RES).T[["accuracy", "precision", "recall", "f1", "roc_auc", "cv_f1", "cv_roc_auc", "selection_score"]]
    t.columns = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV F1 (train)", "CV ROC-AUC (train)", "Selection score"]
    st.dataframe(t.style.format("{:.4f}").highlight_max(axis=0, color="#cfe8cf"), width="stretch")
    others = [m for m in RES if m != BEST]
    st.success(f"**Selected model: {BEST}.** It has the highest selection score (mean of cross-validated F1 and ROC-AUC on the "
               f"training data = {RES[BEST]['selection_score']:.4f}). On the test set it reached F1 {RES[BEST]['f1']:.3f} and "
               f"ROC-AUC {RES[BEST]['roc_auc']:.3f}, versus " +
               "; ".join(f"{m}: F1 {RES[m]['f1']:.3f}, AUC {RES[m]['roc_auc']:.3f}" for m in others) +
               ". Accuracy alone was not used for selection.")
    m = t[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]].reset_index(names="Model").melt("Model", var_name="Metric")
    st.plotly_chart(px.bar(m, x="Metric", y="value", color="Model", barmode="group", text_auto=".3f",
                           title="Test-set metrics by model"), width="stretch")
    fig = go.Figure()
    for n, r in RES.items():
        fig.add_trace(go.Scatter(x=r["roc"]["fpr"], y=r["roc"]["tpr"], name=f"{n} (AUC {r['roc_auc']:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="grey"), name="Chance"))
    fig.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate", height=480)
    st.plotly_chart(fig, width="stretch")
    st.subheader("Confusion matrices")
    cols = st.columns(3)
    for col, (n, r) in zip(cols, RES.items()):
        cm = np.array(r["confusion_matrix"])
        f = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", x=["Pred <=50K", "Pred >50K"],
                      y=["Actual <=50K", "Actual >50K"], title=n)
        f.update_layout(coloraxis_showscale=False, height=330)
        col.plotly_chart(f, width="stretch")
    st.subheader(f"Explainability: what drives the {BEST} model")
    imp = pd.Series(meta["importance"]).sort_values().reset_index()
    imp.columns = ["feature", "AUC drop when shuffled"]
    st.plotly_chart(px.bar(imp, x="AUC drop when shuffled", y="feature", orientation="h",
                           title="Permutation importance (test sample; larger = more important)"), width="stretch")
    st.caption("Hyper-parameters are fixed, sensible defaults (not exhaustively tuned): Decision Tree depth 8 / leaf 25; "
               "Random Forest 200 trees / leaf 3; class weights balanced for all models.")

# ----------------------------------------------------------------------------- predict
elif page == "Live Prediction":
    st.title("Live Income Prediction")
    st.caption(f"Model in use: **{BEST}** (chosen automatically from the evaluation results).")
    o, rg, base = meta["options"], meta["ranges"], meta["baseline"]
    with st.form("f"):
        a, b, c = st.columns(3)
        age = a.slider("Age", rg["age"][0], rg["age"][1], 38)
        hours = a.slider("Hours per week", 1, 99, 40)
        gain = a.number_input("Capital gain ($)", 0, 99999, 0, step=500)
        loss = a.number_input("Capital loss ($)", 0, 4356, 0, step=100)
        edu = b.selectbox("Education", o["education"], index=o["education"].index("Bachelors"))
        work = b.selectbox("Workclass", o["workclass"], index=o["workclass"].index("Private"))
        occ = b.selectbox("Occupation", o["occupation"], index=o["occupation"].index("Prof-specialty"))
        ctry = b.selectbox("Native country", o["native-country"], index=o["native-country"].index("United-States"))
        mar = c.selectbox("Marital status", o["marital-status"], index=o["marital-status"].index("Married-civ-spouse"))
        rel = c.selectbox("Relationship", o["relationship"], index=o["relationship"].index("Husband"))
        race = c.selectbox("Race", o["race"], index=o["race"].index("White"))
        sex = c.selectbox("Sex", o["sex"])
        go_ = st.form_submit_button("Predict income category", type="primary", width="stretch")
    if go_:
        row = pd.DataFrame([{"age": age, "workclass": work, "education": edu, "marital-status": mar,
                             "occupation": occ, "relationship": rel, "race": race, "sex": sex,
                             "capital-gain": gain, "capital-loss": loss, "hours-per-week": hours,
                             "native-country": ctry}])[P.FEATURES]
        p = float(model.predict_proba(row)[0, 1])
        label = P.LABELS[int(p >= 0.5)]
        conf = p if label == ">50K" else 1 - p
        r1, r2 = st.columns([1, 1])
        with r1:
            st.markdown(f"### Predicted income category: **{label}**")
            st.metric("Prediction probability", f"{conf:.1%}", help="Probability of the predicted class")
            st.caption(f"P(>50K) = {p:.1%} · P(<=50K) = {1 - p:.1%} · Model: {BEST}")
        with r2:
            g = go.Figure(go.Indicator(mode="gauge+number", value=p * 100, number={"suffix": "%"},
                          title={"text": "Probability of earning >50K"},
                          gauge={"axis": {"range": [0, 100]}, "bar": {"color": ORANGE},
                                 "threshold": {"line": {"color": "black", "width": 3}, "value": 50}}))
            g.update_layout(height=240, margin=dict(t=60, b=0))
            st.plotly_chart(g, width="stretch")
        ex = P.explain_instance(model, row, base)
        pos, neg = ex[ex.effect > 0.005].head(2), ex[ex.effect < -0.005].head(2)
        fmt = lambda d: ", ".join(f"{r.feature} = {r.value}" for r in d.itertuples()) or "none notable"
        st.info(f"**Why?** Compared with a typical profile in the data, the factors pushing towards **>50K** are: {fmt(pos)}. "
                f"Factors pushing towards **<=50K**: {fmt(neg)}.")
        ex["direction"] = np.where(ex.effect >= 0, "towards >50K", "towards <=50K")
        ex["label"] = ex.feature + " = " + ex.value.astype(str)
        fig = px.bar(ex.head(8).iloc[::-1], x=ex.head(8).iloc[::-1].effect * 100, y="label", orientation="h",
                     color="direction", color_discrete_map={"towards >50K": ORANGE, "towards <=50K": BLUE},
                     labels={"x": "Change in P(>50K), percentage points", "label": ""},
                     title="Top factors for this prediction (what-if vs typical profile)")
        st.plotly_chart(fig, width="stretch")
    st.caption("For educational use only. The model learned historical (1994) US census patterns, which include "
               "social inequalities (e.g. by sex and race); it must not be used for real hiring or pay decisions.")

# ----------------------------------------------------------------------------- insights
else:
    st.title("Key Insights")
    base = df.high.mean() * 100
    def best_worst(col, n=30):
        g = rate_by(col, min_n=n)
        return g.iloc[-1], g.iloc[0]
    eb, ew = best_worst("education"); ob, ow = best_worst("occupation"); wb, ww = best_worst("workclass")
    mb, mw = best_worst("marital-status")
    sx = rate_by("sex").set_index("sex")["rate"]
    hrs = df.groupby(pd.cut(df["hours-per-week"], [0, 39, 40, 100], labels=["<40", "40", ">40"]), observed=True)["high"].mean() * 100
    cg = df[df["capital-gain"] > 0].high.mean() * 100
    top3 = list(pd.Series(meta["importance"]).sort_values(ascending=False).index[:3])
    items = [
        f"**Class imbalance:** only {base:.1f}% of employees earn >50K, so accuracy is misleading; a model always predicting <=50K would already be ~{100 - base:.0f}% 'accurate'. F1 and ROC-AUC were used for selection.",
        f"**Education:** {eb['education']} holders earn >50K {eb['rate']:.0f}% of the time vs {ew['education']} at {ew['rate']:.1f}%.",
        f"**Occupation:** {ob['occupation']} is highest ({ob['rate']:.0f}% >50K) and {ow['occupation']} lowest ({ow['rate']:.1f}%).",
        f"**Workclass:** {wb['workclass']} leads ({wb['rate']:.0f}%); {ww['workclass']} is lowest ({ww['rate']:.1f}%).",
        f"**Marital status:** {mb['marital-status']} shows {mb['rate']:.0f}% >50K vs {mw['marital-status']} at {mw['rate']:.1f}%, partly an age and household-income effect.",
        f"**Hours:** {hrs['>40']:.0f}% of people working over 40 h/week earn >50K, compared with {hrs['40']:.0f}% at exactly 40 h and {hrs['<40']:.0f}% below 40 h.",
        f"**Gender gap:** {sx.get('Male', float('nan')):.1f}% of men vs {sx.get('Female', float('nan')):.1f}% of women earn >50K in this data. This describes the dataset, not a causal effect, and is partly confounded by hours, occupation and marital status.",
        f"**Capital gains:** {cg:.0f}% of people with any capital gain earn >50K (vs {base:.0f}% overall).",
        f"**Model:** {BEST} won (test F1 {RES[BEST]['f1']:.3f}, ROC-AUC {RES[BEST]['roc_auc']:.3f}). Its most influential inputs are {', '.join(top3)}.",
    ]
    for i in items:
        st.markdown(f"- {i}")
    st.subheader("Limitations")
    st.markdown("- 1994 US Census data; salaries and job markets have changed.\n"
                "- Income is only a binary threshold (50K), so the model cannot estimate an actual salary.\n"
                "- Sensitive attributes (sex, race, native country) were included as requested; predictions reflect historical bias and should not be used for decisions about real people.\n"
                "- Hyper-parameters were not exhaustively tuned; the 'Unknown' category is kept rather than imputed.")
