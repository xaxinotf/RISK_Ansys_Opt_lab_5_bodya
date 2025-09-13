# app.py
# --------------------------------------------
# Ординальний підхід до оцінювання ризику банківської/платіжної системи
# ДОДАНО:
#  - Агрегації: Борда, Середнє норм. рангів, Медіана рангу, Copeland (з вагами)
#  - Більше графіків: радар (поляр), heatmap кореляцій, scatter-matrix,
#    діаграма ваг, індикатор Kendall's W (gauge)
#  - Вкладки і приємніші стилі
# --------------------------------------------

import io
import base64
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, dash_table, Input, Output, State, ctx, no_update
import plotly.graph_objs as go
import plotly.express as px

# ---------- Прикладні дані ----------
BANKS = ["Bank A", "Bank B", "Bank C", "Bank D", "Bank E", "Bank F"]

INDICATORS_META = pd.DataFrame([
    {"indicator": "NPL_ratio",           "direction": +1, "weight": 1.0, "desc": "Частка проблемних кредитів (NPL)"},
    {"indicator": "Liquidity_ratio",     "direction": -1, "weight": 1.0, "desc": "Коеф. миттєвої/поточної ліквідності"},
    {"indicator": "CAR",                 "direction": -1, "weight": 1.0, "desc": "Капітальна адекватність (CAR)"},
    {"indicator": "Leverage",            "direction": +1, "weight": 1.0, "desc": "Фінансовий левередж"},
    {"indicator": "ROA",                 "direction": -1, "weight": 1.0, "desc": "Рентабельність активів"},
    {"indicator": "Payment_fail_rate",   "direction": +1, "weight": 1.0, "desc": "Частка відмов/збоїв у платежах"},
    {"indicator": "Settlement_delay",    "direction": +1, "weight": 1.0, "desc": "Затримки розрахунків (середнє, год)"},
    {"indicator": "Fraud_incidents",     "direction": +1, "weight": 1.0, "desc": "Зареєстровані випадки шахрайства"},
], columns=["indicator", "direction", "weight", "desc"])

np.random.seed(42)
VALUES = pd.DataFrame({
    "NPL_ratio":         [5.2, 7.9, 3.1, 6.0, 4.4, 8.3],
    "Liquidity_ratio":   [0.95, 0.82, 1.15, 0.88, 1.05, 0.74],
    "CAR":               [14.2, 12.5, 16.8, 11.9, 15.0, 10.7],
    "Leverage":          [9.5, 11.2, 7.4, 10.1, 8.3, 12.0],
    "ROA":               [1.1, 0.6, 1.5, 0.9, 1.2, 0.4],
    "Payment_fail_rate": [0.8, 1.7, 0.5, 1.1, 0.6, 2.0],
    "Settlement_delay":  [1.2, 2.5, 0.8, 1.6, 1.1, 3.1],
    "Fraud_incidents":   [12, 16,  7, 10,  9, 19],
}, index=BANKS)

# ---------- Утиліти ----------
def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.array([safe_float(i, 0.0) for i in w], dtype=float)
    w = np.where(np.isfinite(w) & (w >= 0), w, 0.0)
    s = w.sum()
    if s <= 0:
        return np.ones_like(w) / len(w)
    return w / s

def compute_ranks(df_vals: pd.DataFrame, meta: pd.DataFrame, method="average"):
    """
    Рахує ранги по кожному індикатору з урахуванням напряму.
    1 = найкращий (менш ризиковий).
    """
    ranks = pd.DataFrame(index=df_vals.index)
    for _, row in meta.iterrows():
        col = row["indicator"]
        direction = int(row["direction"])
        ascending = True if direction == +1 else False
        ranks[col] = df_vals[col].rank(method=method, ascending=ascending)
    return ranks

def weighted_median(values, weights):
    """Зважена медіана (стійкий спосіб агрегування рангів)."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    c = np.cumsum(w) / np.sum(w)
    idx = np.searchsorted(c, 0.5)
    idx = np.clip(idx, 0, len(v)-1)
    return float(v[idx])

def copeland_score(df_vals: pd.DataFrame, meta: pd.DataFrame):
    """
    Copeland з вагами: індикатори як "виборці".
    Для кожної пари банків i,j рахуємо зважену перевагу i над j:
      sign(кращий_за_індикатором) * weight
    Потім: Copeland_i = (#перемог_i - #поразок_i).
    Нормуємо у [0,1], де 1 = найкращий, 0 = найгірший.
    """
    banks = df_vals.index.tolist()
    w = normalize_weights(meta["weight"].values)
    W = {b: 0.0 for b in banks}
    for a_idx in range(len(banks)):
        for b_idx in range(a_idx + 1, len(banks)):
            a = banks[a_idx]
            b = banks[b_idx]
            advantage = 0.0
            for k, row in meta.iterrows():
                ind = row["indicator"]
                direction = int(row["direction"])
                wa = w[k]
                va, vb = df_vals.loc[a, ind], df_vals.loc[b, ind]
                if direction == +1:      # менше краще
                    better = 1 if va < vb else (-1 if va > vb else 0)
                else:                    # більше краще
                    better = 1 if va > vb else (-1 if va < vb else 0)
                advantage += wa * better
            if advantage > 0:
                W[a] += 1; W[b] -= 1
            elif advantage < 0:
                W[a] -= 1; W[b] += 1
    n = len(banks)
    denom = 2 * (n - 1) if n > 1 else 1
    copeland_raw = pd.Series(W, name="CopelandRaw")
    copeland_norm = (copeland_raw + (n - 1)) / denom
    return copeland_raw, copeland_norm  # 1=краще

def aggregate_risk(vals: pd.DataFrame, ranks: pd.DataFrame, meta: pd.DataFrame, agg_mode="borda"):
    """
    Повертає (score_raw, score_norm, note), де score_norm у [0,1] і
    МЕНШЕ = КРАЩЕ (менш ризиково), щоб бути узгодженими з рештою UI.
    """
    w = normalize_weights(meta["weight"].values)
    if agg_mode == "borda":
        score = (ranks.values * w).sum(axis=1)
        score = pd.Series(score, index=ranks.index, name="RiskScore")
        mn, mx = score.min(), score.max()
        norm = (score - mn) / (mx - mn) if mx > mn else score - mn
        return score, norm, "Борда (зважена сума рангів)"

    if agg_mode == "avg":
        n = len(ranks.index)
        norm_ranks = (ranks - 1) / (n - 1) if n > 1 else ranks * 0.0
        score = (norm_ranks.values * w).sum(axis=1)
        score = pd.Series(score, index=ranks.index, name="RiskScore")
        return score, score, "Зважене середнє нормованих рангів"

    if agg_mode == "median":
        med = [weighted_median(ranks.loc[i].values, w) for i in ranks.index]
        score = pd.Series(med, index=ranks.index, name="RiskScore")
        n = len(ranks.index)
        norm = (score - 1) / (n - 1) if n > 1 else score * 0.0
        return score, norm, "Зважена медіана рангу"

    if agg_mode == "copeland":
        cop_raw, cop_norm_good = copeland_score(vals, meta)   # 1 = краще
        risk_norm = 1.0 - cop_norm_good
        inv_raw = -cop_raw
        inv_raw.name = "RiskScore"
        return inv_raw, pd.Series(risk_norm, name="RiskNorm"), "Copeland (ваговий, парні порівняння)"

    # дефолт
    return aggregate_risk(vals, ranks, meta, "borda")

def kendall_w(ranks: pd.DataFrame):
    R = ranks.values
    n, m = R.shape
    if n < 2 or m < 2:
        return np.nan
    Ri = R.sum(axis=1)
    Rbar = Ri.mean()
    S = ((Ri - Rbar) ** 2).sum()
    W = 12 * S / (m**2 * (n**3 - n))
    return float(W)

def risk_bucket(x, q_low=0.33, q_high=0.66):
    if x <= q_low: return "Low"
    if x <= q_high: return "Medium"
    return "High"

# ---------- Створюємо Dash ----------
app = Dash(__name__)
server = app.server

# Глобальний стиль/тема Plotly
PLOTLY_TEMPLATE = "plotly_white"

def card(children):
    return html.Div(
        children=children,
        style={"background":"white","borderRadius":"16px","padding":"14px","boxShadow":"0 6px 18px rgba(0,0,0,0.08)"}
    )

def section_title(text):
    return html.H3(text, style={"margin":"0 0 8px 0"})

def meta_table_data(meta: pd.DataFrame):
    return [{"indicator": r["indicator"], "direction": int(r["direction"]), "weight": float(r["weight"]), "desc": r["desc"]}
            for _, r in meta.iterrows()]

def values_table_data(vals: pd.DataFrame):
    return [{"Bank": idx, **{c: float(vals.loc[idx, c]) for c in vals.columns}} for idx in vals.index]

app.layout = html.Div(style={"maxWidth":"1300px","margin":"0 auto","padding":"16px","background":"#f7f8fa"}, children=[
    html.H2("Ординальна оцінка ризику банківської/платіжної системи", style={"marginBottom":"6px"}),
    html.P("Редагуйте індикатори, напрями впливу та ваги. Оберіть режим агрегування. Результат — інтегральний ризик та рейтинг."),

    card([
        section_title("Індикатори та ваги"),
        dash_table.DataTable(
            id="meta-table",
            columns=[
                {"name":"Індикатор","id":"indicator"},
                {"name":"Напрям (+1=більше гірше; -1=більше краще)","id":"direction","type":"numeric"},
                {"name":"Вага","id":"weight","type":"numeric"},
                {"name":"Опис","id":"desc"},
            ],
            data=meta_table_data(INDICATORS_META),
            editable=True,
            style_cell={"textAlign":"center","padding":"6px","whiteSpace":"normal"},
            style_header={"fontWeight":"700"},
            tooltip_header={"direction":"Напрям: +1 — більше = гірше; -1 — більше = краще",
                            "weight":"Вага індикатора у підсумковому ризику"},
            tooltip_delay=300, tooltip_duration=None,
        ),
    ]),

    html.Div(style={"display":"grid","gridTemplateColumns":"1fr","gap":"12px","marginTop":"12px"}, children=[
        card([
            section_title("Дані по банках/платіжних установах"),
            dash_table.DataTable(
                id="values-table",
                columns=[{"name":"Банк/Установа","id":"Bank"}] +
                        [{"name":c,"id":c,"type":"numeric"} for c in VALUES.columns],
                data=values_table_data(VALUES),
                editable=True, row_deletable=True,
                style_table={"overflowX":"auto"},
                style_cell={"textAlign":"center","padding":"6px"},
                style_header={"fontWeight":"700"},
            ),
            html.Div(style={"display":"flex","gap":"8px","marginTop":"8px"}, children=[
                html.Button("Додати банк", id="add-bank", n_clicks=0, className="btn"),
                html.Button("Експорт CSV", id="export", n_clicks=0, className="btn"),
                dcc.Download(id="dl"),
            ])
        ])
    ]),

    html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr 1fr","gap":"12px","marginTop":"12px"}, children=[
        card([
            html.Label("Режим агрегування"),
            dcc.Dropdown(
                id="agg-mode",
                options=[
                    {"label":"Борда (зважена сума рангів)","value":"borda"},
                    {"label":"Зважене середнє нормованих рангів","value":"avg"},
                    {"label":"Медіана рангу (зваж.)","value":"median"},
                    {"label":"Copeland (парні порівняння, з вагами)","value":"copeland"},
                ],
                value="borda", clearable=False
            ),
        ]),
        card([
            html.Label("Банк для деталізації"),
            dcc.Dropdown(id="focus-bank", options=[{"label":b,"value":b} for b in BANKS], value=BANKS[0], clearable=False)
        ]),
        card([
            html.Div(id="summary", style={"fontWeight":600,"minHeight":"40px"}),
            dcc.Graph(id="kendall-gauge", config={"displayModeBar": False})
        ])
    ]),

    dcc.Tabs(id="tabs", value="tab_overview", children=[
        dcc.Tab(label="Огляд", value="tab_overview", children=[
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"12px","marginTop":"12px"}, children=[
                card([dcc.Graph(id="ranks-heatmap")]),
                card([dcc.Graph(id="risk-bars")]),
            ]),
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"12px","marginTop":"12px"}, children=[
                card([dcc.Graph(id="weights-bar")]),
                card([dcc.Graph(id="corr-heatmap")]),
            ]),
        ]),
        dcc.Tab(label="Деталізація банку", value="tab_focus", children=[
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr","gap":"12px","marginTop":"12px"}, children=[
                card([dcc.Graph(id="focus-indicators")]),
                card([dcc.Graph(id="focus-radar")]),
            ])
        ]),
        dcc.Tab(label="Розкид/зв'язки", value="tab_scatter", children=[
            html.Div(style={"marginTop":"12px"}, children=[ card([dcc.Graph(id="scatter-matrix")]) ])
        ]),
    ]),
])

# ---------- Додати банк ----------
@app.callback(
    Output("values-table", "data"),
    Input("add-bank", "n_clicks"),
    State("values-table", "data"),
    prevent_initial_call=True
)
def add_bank(n, data):
    if not n:
        return no_update
    existing = [row.get("Bank","") for row in data]
    i = 1
    cand = f"Bank X{i}"
    while cand in existing:
        i += 1; cand = f"Bank X{i}"
    new_row = {"Bank": cand}
    for c in VALUES.columns: new_row[c] = 0.0
    return data + [new_row]

# ---------- Експорт ----------
@app.callback(
    Output("dl", "data"),
    Input("export", "n_clicks"),
    State("meta-table", "data"),
    State("values-table", "data"),
    State("agg-mode", "value"),
    prevent_initial_call=True
)
def export_csv(n, meta_rows, val_rows, agg_mode):
    if not n:
        return no_update
    meta = pd.DataFrame(meta_rows)
    vals = pd.DataFrame(val_rows).set_index("Bank")
    vals = vals.apply(pd.to_numeric, errors="coerce")
    ranks = compute_ranks(vals, meta)
    score, norm, _ = aggregate_risk(vals, ranks, meta, agg_mode=agg_mode)

    out = io.StringIO()
    out.write("# META\n"); meta.to_csv(out, index=False)
    out.write("\n# VALUES\n"); vals.to_csv(out)
    out.write("\n# RANKS\n"); ranks.to_csv(out)
    out.write("\n# RISK (raw)\n"); score.to_csv(out, header=["RiskScore"])
    out.write("\n# RISK_NORM [0..1]\n"); norm.to_csv(out, header=["RiskNorm"])
    content = "data:text/csv;charset=utf-8," + base64.b64encode(out.getvalue().encode()).decode()
    return dict(content=content, filename="ordinal_risk_export.csv")

# ---------- Головний розрахунок / візуалізації ----------
@app.callback(
    Output("ranks-heatmap", "figure"),
    Output("risk-bars", "figure"),
    Output("focus-indicators", "figure"),
    Output("summary", "children"),
    Output("focus-bank", "options"),
    Output("kendall-gauge", "figure"),
    Output("weights-bar", "figure"),
    Output("corr-heatmap", "figure"),
    Output("scatter-matrix", "figure"),
    Output("focus-radar", "figure"),
    Input("meta-table", "data"),
    Input("values-table", "data"),
    Input("agg-mode", "value"),
    Input("focus-bank", "value"),
)
def update_all(meta_rows, val_rows, agg_mode, focus_bank):
    meta = pd.DataFrame(meta_rows)
    vals = pd.DataFrame(val_rows).set_index("Bank")
    vals = vals.apply(pd.to_numeric, errors="coerce")
    cols_present = [c for c in meta["indicator"] if c in vals.columns]
    meta = meta[meta["indicator"].isin(cols_present)].reset_index(drop=True)
    vals = vals[cols_present].copy()

    if vals.empty or meta.empty:
        empty = go.Figure(); empty.update_layout(template=PLOTLY_TEMPLATE)
        options = [{"label": b, "value": b} for b in BANKS]
        return empty, empty, empty, "Немає даних.", options, empty, empty, empty, empty, empty

    # ранги та агрегування
    ranks = compute_ranks(vals, meta)
    score, norm, agg_name = aggregate_risk(vals, ranks, meta, agg_mode=agg_mode)

    # категорії ризику
    q1, q2 = norm.quantile([0.33, 0.66])
    buckets = norm.apply(lambda x: risk_bucket(x, q1, q2))

    # Kendall's W
    W = kendall_w(ranks)

    # --- Heatmap рангів ---
    heat = go.Figure(data=go.Heatmap(
        z=ranks.values, x=ranks.columns.tolist(), y=ranks.index.tolist(),
        colorbar=dict(title="Ранг (1 = краще)"), colorscale="YlOrRd"
    ))
    heat.update_layout(title="Теплокарта рангів за індикаторами", template=PLOTLY_TEMPLATE,
                       margin=dict(t=50,l=40,r=10,b=40))

    # --- Барчарт інтегрального ризику ---
    risk_df = pd.DataFrame({"RiskNorm": norm, "Bucket": buckets}).sort_values("RiskNorm", ascending=True)
    bars = go.Figure(data=go.Bar(
        x=risk_df.index.tolist(), y=risk_df["RiskNorm"].values,
        text=risk_df["Bucket"].values, textposition="outside",
        name="Інтегральний ризик [0..1]"
    ))
    bars.update_layout(title="Інтегральний ординальний ризик (нижче = краще)",
                       xaxis_title="Банк/Установа", yaxis_title="Ризик [0..1]",
                       template=PLOTLY_TEMPLATE, margin=dict(t=50,l=50,r=20,b=50))

    # --- Деталізація по банку ---
    focus = focus_bank if focus_bank in vals.index else vals.index[0]
    focus_vals = vals.loc[focus]
    focus_ranks = ranks.loc[focus]
    n = len(ranks.index)
    norm_r = (focus_ranks - 1) / (n - 1) if n > 1 else focus_ranks * 0.0
    detail = go.Figure()
    detail.add_trace(go.Bar(name="Норм. ранг (нижче — краще)", x=norm_r.index.tolist(), y=norm_r.values))
    detail.add_trace(go.Scatter(name="Значення індикатора", x=focus_vals.index.tolist(), y=focus_vals.values,
                                mode="lines+markers", yaxis="y2"))
    detail.update_layout(
        title=f"Деталізація по {focus}",
        yaxis=dict(title="Норм. ранг [0..1]", rangemode="tozero"),
        yaxis2=dict(title="Значення", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h"),
        template=PLOTLY_TEMPLATE, margin=dict(t=50,l=50,r=20,b=50)
    )

    # --- Індикатор Kendall's W ---
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=0 if not np.isfinite(W) else W,
        title={"text":"Kendall's W"},
        gauge={"axis":{"range":[0,1]},"bar":{"thickness":0.3},
               "steps":[{"range":[0,0.33],"color":"#fde0dd"},
                        {"range":[0.33,0.66],"color":"#fa9fb5"},
                        {"range":[0.66,1.0],"color":"#c51b8a"}]}
    ))
    gauge.update_layout(height=220, template=PLOTLY_TEMPLATE, margin=dict(t=10,l=20,r=20,b=10))

    # --- Ваги індикаторів ---
    weights_bar = go.Figure(data=go.Bar(x=meta["indicator"], y=normalize_weights(meta["weight"].values)))
    weights_bar.update_layout(title="Нормовані ваги індикаторів",
                              xaxis_title="Індикатор", yaxis_title="Вага",
                              template=PLOTLY_TEMPLATE, margin=dict(t=50,l=50,r=20,b=50))

    # --- Heatmap кореляцій значень індикаторів ---
    if len(vals.index) >= 3:
        corr = vals.corr(numeric_only=True)
        corr_heat = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index, colorscale="RdBu", zmid=0,
            colorbar=dict(title="Кореляція")
        ))
        corr_heat.update_layout(title="Кореляції між індикаторами (за значеннями)",
                                template=PLOTLY_TEMPLATE, margin=dict(t=50,l=50,r=20,b=50))
    else:
        corr_heat = go.Figure(); corr_heat.update_layout(template=PLOTLY_TEMPLATE)

    # --- Scatter-matrix (розкид) — ВИПРАВЛЕНО ---
    dims_all = meta["indicator"].tolist()
    vals_num = vals.copy()

    dims_valid = []
    for c in dims_all:
        col = pd.to_numeric(vals_num[c], errors="coerce")
        if col.notna().sum() >= 2:
            dims_valid.append(c)
    vals_num = vals_num[dims_valid]

    if len(dims_valid) >= 2 and len(vals_num.index) >= 2:
        df_sm = vals_num.reset_index()  # перша колонка — це назви банків
        # гарантуємо, що колонка називається 'Bank' (для color):
        if df_sm.columns[0] != "Bank":
            df_sm = df_sm.rename(columns={df_sm.columns[0]: "Bank"})
        sm = px.scatter_matrix(
            df_sm,
            dimensions=dims_valid,
            color="Bank",  # 🔧 було 'index' — спричиняло ValueError
            title="Scatter-Matrix індикаторів"
        )
        sm.update_layout(template=PLOTLY_TEMPLATE, height=700,
                         margin=dict(t=50,l=50,r=20,b=50), legend_title_text="Банк")
    else:
        sm = go.Figure()
        msg = ("Недостатньо даних для scatter-matrix: потрібно ≥2 індикатори та ≥2 банки "
               "з числовими значеннями без суцільних пропусків.")
        sm.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                          showarrow=False, font=dict(size=14))
        sm.update_xaxes(visible=False); sm.update_yaxes(visible=False)
        sm.update_layout(template=PLOTLY_TEMPLATE, height=300, margin=dict(t=40,l=40,r=40,b=40))

    # --- Радар (поляр) нормованих рангів ---
    radar_categories = meta["indicator"].tolist()
    radar_vals = norm_r.reindex(radar_categories).values
    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(
        r=np.append(radar_vals, radar_vals[0]) if len(radar_vals)>0 else radar_vals,
        theta=np.append(radar_categories, radar_categories[0]) if len(radar_categories)>0 else radar_categories,
        fill='toself', name=f"{focus}"
    ))
    radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        title=f"Профіль нормованих рангів (0 — краще) • {focus}",
        template=PLOTLY_TEMPLATE, margin=dict(t=50,l=50,r=20,b=50)
    )

    # --- Summary ---
    risk_df["Bank"] = risk_df.index
    top_k = risk_df.head(3)["Bank"].tolist()
    worst_k = risk_df.tail(3)["Bank"].tolist()
    summary = (f"Агрегування: {agg_name}. Kendall's W = "
               f"{'%.3f' % W if np.isfinite(W) else 'н/д'}. "
               f"Топ-3 найменш ризикових: {', '.join(top_k)}. "
               f"Топ-3 з підвищеним ризиком: {', '.join(worst_k)}.")

    options = [{"label": b, "value": b} for b in vals.index.tolist()]
    return heat, bars, detail, summary, options, gauge, weights_bar, corr_heat, sm, radar

# ---------- Запуск ----------
if __name__ == "__main__":
    app.run(debug=True, port=8050, use_reloader=False)
