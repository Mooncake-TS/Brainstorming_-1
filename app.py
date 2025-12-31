# app.py
# AutoPlanIQ — YOLO + LSTM 컨셉: 재고 정합성 자동화 & 리드타임 기반 추천 생산량
#
# 실행:
#   pip install -r requirements.txt
#   streamlit run app.py
#
# 데이터(레포 ./data):
#   - data/sku_master.csv
#   - data/sales_history.csv
#   - data/erp_inventory.csv
#   - data/vision_count.csv
#
# 이미지(선택): real_shelf.png / yolo_sim.png

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import altair as alt
except Exception:
    alt = None

APP_NAME = "AutoPlanIQ"
TAGLINE = "YOLO + LSTM 기반 재고 정합성 자동화로 생산계획 정밀도 향상"

st.set_page_config(page_title=f"{APP_NAME} | {TAGLINE}", page_icon="📦", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
.hero { padding: 18px 20px; border: 1px solid rgba(0,0,0,0.08); border-radius: 16px; background: rgba(0,0,0,0.02); }
.hero-title { font-size: 34px; font-weight: 800; margin: 0 0 6px 0; }
.hero-sub { font-size: 15px; color: rgba(0,0,0,0.70); margin: 0; line-height: 1.45; }
.card { padding: 16px; border: 1px solid rgba(0,0,0,0.10); border-radius: 16px; background: white; height: 100%; }
.card h3 { margin: 0 0 6px 0; font-size: 18px; }
.card p { margin: 0; color: rgba(0,0,0,0.70); line-height: 1.45; }
.section { padding: 16px 18px; border: 1px solid rgba(0,0,0,0.08); border-radius: 16px; background: white; }
hr { border: none; border-top: 1px solid rgba(0,0,0,0.08); margin: 18px 0; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- Sidebar: Upload or ./data ---------------
st.sidebar.title("📁 Data")
use_sample = st.sidebar.checkbox("샘플 데이터 사용(권장)", value=False)

up_sku = st.sidebar.file_uploader("sku_master.csv", type=["csv"])
up_sales = st.sidebar.file_uploader("sales_history.csv", type=["csv"])
up_erp = st.sidebar.file_uploader("erp_inventory.csv", type=["csv"])
up_vision = st.sidebar.file_uploader("vision_count.csv", type=["csv"])

st.sidebar.divider()
st.sidebar.subheader("AutoPlan 설정")
forecast_months = st.sidebar.selectbox("LSTM 예측 기간(개월)", [1, 2, 3], index=0)
noise_pct = st.sidebar.slider("예측 변동성(±%)", 0, 20, 6, step=1)
service_level = st.sidebar.selectbox("서비스레벨(데모)", ["90%", "95%", "98%"], index=1)
sl_map = {"90%": 1.28, "95%": 1.65, "98%": 2.05}
z = sl_map[service_level]

DATA_DIR = Path("data")

def _read_csv(uploaded, fallback: Path, required_cols):
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    elif fallback.exists():
        df = pd.read_csv(fallback)
    else:
        return None
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"{fallback.name} 컬럼 누락: {missing}")
        return None
    return df

@st.cache_data
def load_sample():
    # 샘플은 레포의 ./data 를 사용
    sku = pd.read_csv(DATA_DIR / "sku_master.csv")
    sales = pd.read_csv(DATA_DIR / "sales_history.csv")
    erp = pd.read_csv(DATA_DIR / "erp_inventory.csv")
    vision = pd.read_csv(DATA_DIR / "vision_count.csv")
    return sku, sales, erp, vision

def load_data():
    if use_sample:
        return load_sample()

    sku = _read_csv(up_sku, DATA_DIR/"sku_master.csv",
                    ["sku_id","sku_name","lead_time_days","safety_stock_days","moq","pack_size"])
    sales = _read_csv(up_sales, DATA_DIR/"sales_history.csv", ["month","sku_id","qty_sold"])
    erp = _read_csv(up_erp, DATA_DIR/"erp_inventory.csv", ["bin_id","sku_id","erp_qty"])
    vision = _read_csv(up_vision, DATA_DIR/"vision_count.csv", ["bin_id","sku_id","vision_qty","timestamp"])
    return sku, sales, erp, vision

sku_master, sales_history, erp_inventory, vision_count = load_data()

st.markdown(
    f"""
<div class="hero">
  <div class="hero-title">{APP_NAME} — {TAGLINE}</div>
  <p class="hero-sub">
    <b>YOLO</b>로 실사 재고를 자동 집계하고, <b>ERP vs 실사</b> 격차를 자동 리포트합니다.
    (데모) <b>LSTM 수요예측</b>으로 리드타임 구간 수요를 예측해 <b>추천 생산량</b>과 <b>우선 생산 Top5</b>를 생성합니다.
  </p>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")

# Cards
c1, c2, c3 = st.columns(3, gap="large")
with c1:
    st.markdown("""<div class="card"><h3>1) Auto Scan (YOLO)</h3>
<p>bin(칸)별 카운팅 → 실사 재고(vision_qty)</p><p><b>Location = SKU</b> 가정이면 안정성이 올라갑니다.</p></div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""<div class="card"><h3>2) Auto Gap Report</h3>
<p><code>Gap = Vision - ERP</code> 자동 계산</p><p>정합성 이슈 SKU/bin을 우선순위로 표시</p></div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""<div class="card"><h3>3) AutoPlan (LSTM + Lead Time)</h3>
<p>(데모)LSTM 예측 × 리드타임 + 안전재고 + MOQ/포장단위</p><p><b>추천 생산량</b> & <b>Top5</b> 자동 생성</p></div>""", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# Optional images
real_path = Path("real_shelf.png")
yolo_path = Path("yolo_sim.png")
l, r = st.columns(2, gap="large")
with l:
    st.markdown("### [Real] 창고 선반 실물")
    if real_path.exists(): st.image(str(real_path), use_container_width=True)
    else: st.info("real_shelf.png 없으면 생략")
with r:
    st.markdown("### [Vision] YOLO 시뮬레이션")
    if yolo_path.exists(): st.image(str(yolo_path), use_container_width=True)
    else: st.info("yolo_sim.png 없으면 생략")

st.markdown("<hr/>", unsafe_allow_html=True)

if any(x is None for x in [sku_master, sales_history, erp_inventory, vision_count]):
    st.error("데이터가 부족해요. ./data에 CSV를 올리거나 Sidebar로 업로드해주세요.")
    st.stop()

# Normalize
sales_history["month"] = pd.to_datetime(sales_history["month"])
vision_count["timestamp"] = pd.to_datetime(vision_count["timestamp"])

# Aggregate per SKU
erp_sku = erp_inventory.groupby("sku_id", as_index=False)["erp_qty"].sum().rename(columns={"erp_qty":"erp_stock"})
vis_sku = vision_count.groupby("sku_id", as_index=False)["vision_qty"].sum().rename(columns={"vision_qty":"vision_stock"})

inv = sku_master.merge(erp_sku, on="sku_id", how="left").merge(vis_sku, on="sku_id", how="left").fillna(0)
inv["erp_stock"] = inv["erp_stock"].astype(int)
inv["vision_stock"] = inv["vision_stock"].astype(int)
inv["erp_stock"] = np.maximum(inv["erp_stock"], inv["vision_stock"] + 1)  # business rule

inv["gap_vision_minus_erp"] = inv["vision_stock"] - inv["erp_stock"]
inv["abs_gap"] = (inv["erp_stock"] - inv["vision_stock"]).abs()

# Forecast helpers
def seasonal_avg_month(df: pd.DataFrame, m: int) -> pd.DataFrame:
    tmp = df.copy()
    tmp["mm"] = tmp["month"].dt.month
    sub = tmp[tmp["mm"] == m]
    avg = sub.groupby("sku_id", as_index=False)["qty_sold"].mean().rename(columns={"qty_sold":"avg_sales"})
    avg["avg_sales"] = avg["avg_sales"].round().astype(int)
    return avg

def lstm_demo_forecast(avg_sales_df: pd.DataFrame, months_ahead: int, noise_pct: int, seed=123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = avg_sales_df.copy()
    baseline = out["avg_sales"].to_numpy() * months_ahead
    jitter = rng.uniform(-noise_pct, noise_pct, size=len(out)) / 100.0
    trend = rng.uniform(-0.02, 0.06, size=len(out))
    out["lstm_forecast"] = np.clip(np.round(baseline * (1+jitter+trend)), 0, None).astype(int)
    return out[["sku_id","lstm_forecast"]]

def round_up(x: int, step: int) -> int:
    step = max(int(step), 1)
    return int(((x + step - 1)//step)*step)

def calc_reco(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    daily = out["lstm_forecast"] / (30*forecast_months)
    out["daily_demand"] = daily

    out["lt_demand"] = (daily * out["lead_time_days"]).round().astype(int)
    out["safety_stock"] = (daily * out["safety_stock_days"] * z).round().astype(int)

    need_raw = out["lt_demand"] + out["safety_stock"] - out["vision_stock"]
    out["need_qty_raw"] = np.maximum(0, need_raw).astype(int)

    out["need_qty_moq"] = out.apply(lambda r: round_up(int(r["need_qty_raw"]), int(r["moq"])), axis=1)
    out["reco_prod_qty"] = out.apply(lambda r: round_up(int(r["need_qty_moq"]), int(r["pack_size"])), axis=1)
    out["shortage_lstm_vs_vision"] = np.maximum(0, out["lstm_forecast"] - out["vision_stock"]).astype(int)
    return out

tab1, tab2, tab3 = st.tabs(["Auto Scan", "Auto Gap Report", "AutoPlan"])

with tab1:
    st.markdown("### Auto Scan (YOLO 결과)")
    st.metric("최근 스캔", str(vision_count["timestamp"].max()))
    st.dataframe(vision_count.sort_values("timestamp", ascending=False), use_container_width=True, height=360)

with tab2:
    st.markdown("### Auto Gap Report (ERP vs Vision)")
    st.dataframe(
        inv[["sku_id","sku_name","erp_stock","vision_stock","gap_vision_minus_erp","abs_gap"]].sort_values("abs_gap", ascending=False),
        use_container_width=True, height=360
    )
    if alt is not None:
        long = inv.melt(id_vars=["sku_id","sku_name"], value_vars=["erp_stock","vision_stock"], var_name="metric", value_name="value")
        long["metric"] = long["metric"].map({"erp_stock":"ERP","vision_stock":"Vision(실사)"})
        chart = alt.Chart(long).mark_bar().encode(
            x=alt.X("sku_id:N", title="SKU"),
            xOffset=alt.XOffset("metric:N", sort=["ERP","Vision(실사)"]),
            y=alt.Y("value:Q", title="Units"),
            color=alt.Color("metric:N", title="지표"),
            tooltip=[alt.Tooltip("sku_id:N"), alt.Tooltip("metric:N"), alt.Tooltip("value:Q", format=",")]
        ).properties(height=360)
        st.altair_chart(chart, use_container_width=True)

with tab3:
    st.markdown("### AutoPlan (추천 생산량)")
    month_sel = st.selectbox("기준 월(3년 평균)", list(range(1,13)), index=11)
    avg = seasonal_avg_month(sales_history, month_sel)
    lstm = lstm_demo_forecast(avg, forecast_months, noise_pct)

    plan = inv.merge(avg, on="sku_id", how="left").merge(lstm, on="sku_id", how="left").fillna(0)
    plan["avg_sales"] = plan["avg_sales"].astype(int)
    plan["lstm_forecast"] = plan["lstm_forecast"].astype(int)

    plan = calc_reco(plan)

    if alt is not None:
        long = plan.melt(
            id_vars=["sku_id","sku_name"],
            value_vars=["avg_sales","lstm_forecast","erp_stock","vision_stock"],
            var_name="metric",
            value_name="value"
        )
        label = {
            "avg_sales": f"3년 평균({month_sel}월)",
            "lstm_forecast": f"LSTM 예측({forecast_months}개월)",
            "erp_stock": "ERP",
            "vision_stock": "Vision(실사)"
        }
        order = [label["avg_sales"], label["lstm_forecast"], "ERP", "Vision(실사)"]
        long["metric"] = long["metric"].map(label)

        chart = alt.Chart(long).mark_bar().encode(
            x=alt.X("sku_id:N", title="SKU"),
            xOffset=alt.XOffset("metric:N", sort=order),
            y=alt.Y("value:Q", title="Units"),
            color=alt.Color("metric:N", title="지표", sort=order),
            tooltip=[alt.Tooltip("sku_id:N"), alt.Tooltip("metric:N"), alt.Tooltip("value:Q", format=",")]
        ).properties(height=420)
        st.altair_chart(chart, use_container_width=True)

    st.markdown("#### Top5 우선 생산(추천 생산량 기준)")
    show_cols = ["sku_id","sku_name","lead_time_days","safety_stock_days","moq","pack_size",
                 "vision_stock","erp_stock","avg_sales","lstm_forecast","lt_demand","safety_stock","need_qty_raw","reco_prod_qty"]
    top5 = plan.sort_values(["reco_prod_qty","need_qty_raw"], ascending=[False, False]).head(5)
    st.dataframe(top5[show_cols], use_container_width=True, height=260)

    st.markdown("#### 전체 SKU 추천 생산 테이블")
    full = plan.sort_values(["reco_prod_qty","need_qty_raw"], ascending=[False, False])[show_cols]
    st.dataframe(full, use_container_width=True, height=420)

st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("### 발표용 한 줄 요약")
st.write("**YOLO로 실사 재고를 자동 집계 → ERP와 격차를 자동 리포트 → (데모)LSTM 예측 + 리드타임/안전재고/MOQ로 추천 생산량 자동화**")
