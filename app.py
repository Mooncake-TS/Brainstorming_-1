# app.py
# Streamlit demo: YOLO 기반 실사 재고 자동 집계 + ERP 격차 리포트 + AutoPlan(3년 평균 월 판매량 vs 현재 재고)
# 실행: streamlit run app.py
#
# 이미지 준비(같은 폴더):
#   - real_shelf.png : 실물사진
#   - yolo_sim.png   : YOLO 시뮬레이션(박스 네모 쳐진) 사진

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Altair(차트)
try:
    import altair as alt
except Exception:
    alt = None

st.set_page_config(
    page_title="YOLO + AutoPlan 재고 정합성 자동화",
    page_icon="📦",
    layout="wide",
)

st.markdown(
    """
<style>
.block-container { padding-top: 1.3rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: -0.2px; }

.hero {
  padding: 18px 20px;
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 16px;
  background: rgba(0,0,0,0.02);
}
.hero-title { font-size: 34px; font-weight: 800; margin: 0 0 6px 0; }
.hero-sub { font-size: 16px; color: rgba(0,0,0,0.70); margin: 0; }

.card {
  padding: 16px 16px;
  border: 1px solid rgba(0,0,0,0.10);
  border-radius: 16px;
  background: white;
  height: 100%;
}
.card h3 { margin: 0 0 6px 0; font-size: 18px; }
.card p { margin: 0; color: rgba(0,0,0,0.70); line-height: 1.45; }

.section {
  padding: 16px 18px;
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 16px;
  background: white;
}
.kpi {
  border-radius: 14px;
  padding: 14px;
  border: 1px solid rgba(0,0,0,0.08);
  background: rgba(0,0,0,0.015);
}
.small-muted { color: rgba(0,0,0,0.65); font-size: 13px; }
hr { border: none; border-top: 1px solid rgba(0,0,0,0.08); margin: 18px 0; }
code { font-size: 0.95em; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <div class="hero-title">YOLO 기반 실사 재고 자동화 + ERP 정합성 리포트 + AutoPlan</div>
  <p class="hero-sub">
    카메라(YOLO)로 실물 재고를 <b>자동 집계</b>하고, ERP와의 격차를 <b>자동 탐지·알림</b>하며,
    <b>3년 평균 월 판매량</b>과 비교해 <b>우선 생산(Top5)</b>을 자동 생성하는 컨셉 데모
  </p>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")

c1, c2, c3 = st.columns(3, gap="large")
with c1:
    st.markdown(
        """
<div class="card">
  <h3>1) Auto Scan (YOLO)</h3>
  <p>선반 이미지를 촬영해 <b>bin(칸)별 박스 검출/카운팅</b>을 자동화합니다.</p>
  <p>운영 룰: <b>Location = SKU</b> (한 칸에 한 상품)로 식별을 단순화합니다.</p>
</div>
""",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
<div class="card">
  <h3>2) Auto Gap Report</h3>
  <p><code>Gap = Vision(실사) - ERP</code>를 계산해</p>
  <p>정합성이 깨진 SKU/구역을 <b>우선순위로 리포트</b>합니다.</p>
</div>
""",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        """
<div class="card">
  <h3>3) Auto Plan</h3>
  <p><b>3년 평균 월 판매량</b> vs <b>현재 재고(실사/ERP)</b>를 한 화면에서 비교하고</p>
  <p>부족이 큰 SKU를 <b>Top5 우선 생산</b>으로 자동 정리합니다.</p>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("<hr/>", unsafe_allow_html=True)

# ----------------------------
# Images
# ----------------------------
left, right = st.columns([1, 1], gap="large")
real_path = Path("real_shelf.png")
yolo_path = Path("yolo_sim.png")

with left:
    st.markdown("### [Real World] 창고 선반 실물 이미지")
    if real_path.exists():
        st.image(str(real_path), use_container_width=True)
    else:
        st.warning("이미지 파일을 찾을 수 없어요: `real_shelf.png`")
    st.caption("사람이 직접 세기 어려운 규모 / 로스·불량·샘플 출고 등으로 ERP와 괴리 발생")

with right:
    st.markdown("### [Vision Output] YOLO 기반 박스 인식(시뮬레이션)")
    if yolo_path.exists():
        st.image(str(yolo_path), use_container_width=True)
    else:
        st.warning("이미지 파일을 찾을 수 없어요: `yolo_sim.png`")
    st.caption("박스 위치·수량 추출 → bin 기준 집계 → ERP와 비교")

st.markdown("<hr/>", unsafe_allow_html=True)

# ----------------------------
# Problem -> Concept -> Flow
# ----------------------------
pcol, mcol, scol = st.columns(3, gap="large")
with pcol:
    st.markdown("### 문제 제기 (Problem)")
    st.markdown(
        """
<div class="section">
- ERP 재고는 기록 기반이라 현장 이슈(포장 손실/불량 폐기/샘플 출고/누락)로 <b>과대 계상</b>되기 쉬움<br/>
- 실사 재고조사는 비용이 커서 <b>상시 점검이 어려움</b><br/>
- 결과적으로 생산/출하 계획 오차가 누적되고 <b>결품·과잉재고 리스크</b>가 증가
</div>
""",
        unsafe_allow_html=True,
    )

with mcol:
    st.markdown("### 적용 개념 (Concept)")
    st.markdown(
        """
<div class="section">
<b>YOLO</b>: 이미지에서 박스 검출 → <b>bin별 카운팅 자동화</b><br/><br/>
<b>AutoPlan</b>: 3년 평균 월 판매량 vs 현재 재고(실사/ERP) 비교 → <b>우선 생산 리스트 자동 생성</b><br/><br/>
<b>운영 가정</b>: Location=SKU(한 칸=한 상품) + 안쪽 선적재 룰(밖이 차면 안쪽은 이미 찼다고 추론)
</div>
""",
        unsafe_allow_html=True,
    )

with scol:
    st.markdown("### 자동화 흐름 (Flow)")
    st.markdown(
        """
<div class="section">
1) <b>Auto Scan</b>: 실사 재고(vision) 자동 집계<br/>
2) <b>Auto Gap</b>: ERP vs 실사 격차 자동 리포트<br/>
3) <b>Auto Plan</b>: 평균 판매량 대비 부족 Top5 자동 추출
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("<hr/>", unsafe_allow_html=True)

# ----------------------------
# Controls
# ----------------------------
st.markdown("## 데모 컨트롤")
control_left, control_right = st.columns([1.2, 1], gap="large")

with control_left:
    st.markdown(
        """
<div class="section">
<b>더미 데이터 생성 원칙</b><br/>
- <b>실사 재고 ≤ ERP 재고</b> (손실/불량/샘플 출고로 실사가 더 적은 상황)<br/>
- 판매량과 재고량 차이는 <b>생산으로 커버 가능한 수준</b>으로 설정<br/>
- (가시성) ERP가 실사보다 <b>조금 더 크게</b> 보이도록 갭을 완만하게 부여
</div>
""",
        unsafe_allow_html=True,
    )
    min_threshold = st.slider("부족 임계치(min) — bin 총 재고가 이 값 미만이면 알림", 350, 750, 520, step=10)
    use_inner_rule = st.checkbox("안쪽 선적재 룰 적용 (front>0이면 inner는 FULL로 추정)", value=True)

with control_right:
    st.markdown('<div class="kpi">', unsafe_allow_html=True)
    st.metric("자동 스캔 주기(가정)", "3회/일", "고정 카메라")
    st.metric("대상 구역(가정)", "Rack A", "bin 단위")
    st.metric("핵심 KPI", "재고 정합성", "계획 정밀도")
    st.markdown('<div class="small-muted">※ 아래 테이블/차트는 브레인스토밍용 더미 데이터입니다.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Dummy data (현실 제약 반영)
# ----------------------------
@st.cache_data
def make_dummy_gap_data(n_bins=24, seed=7):
    """
    현실 반영:
    - ERP >= 실사 (손실/불량/샘플 등 반영)
    - Gap 과도하지 않게: ERP가 실사보다 +40~+140 정도 크도록(가시성 확보)
    - SOP 컨셉: front>0인 bin 위주로 생성 (inner full 추론이 잘 먹는 상황)
    """
    rng = np.random.default_rng(seed)

    bins = [f"BIN-{i:02d}" for i in range(1, n_bins + 1)]
    sku_pool = [f"SKU-{i:02d}" for i in range(1, 7)]  # 6 SKUs
    skus = [sku_pool[i % len(sku_pool)] for i in range(n_bins)]

    inner_capacity = 400
    vision_front = rng.integers(60, 351, size=n_bins)  # 60~350
    vision_total_true = inner_capacity + vision_front   # 460~750

    erp_over = rng.integers(40, 141, size=n_bins)       # 40~140 (ERP가 더 크게 보이도록)
    erp_stock = vision_total_true + erp_over            # 항상 ERP > 실사

    return pd.DataFrame({
        "bin_id": bins,
        "sku_id": skus,
        "erp_stock": erp_stock,
        "vision_front": vision_front,
        "inner_capacity": inner_capacity,
    })


def apply_vision_rule(df: pd.DataFrame, use_inner_rule: bool):
    """front 관측 + (선택) SOP 기반 inner 추정으로 vision_total 계산.
    안전장치: 어떤 경우에도 실사(vision_total) <= ERP(erp_stock).
    """
    df = df.copy()

    if use_inner_rule:
        df["vision_inner"] = np.where(df["vision_front"] > 0, df["inner_capacity"], 0)
    else:
        df["vision_inner"] = 0

    df["vision_total"] = df["vision_front"] + df["vision_inner"]
    df["vision_total"] = np.minimum(df["vision_total"], df["erp_stock"])  # 안전 클램프

    df["gap"] = df["vision_total"] - df["erp_stock"]   # 음수(실사가 작음) 정상
    df["confidence"] = np.where(df["vision_front"] > 0, "HIGH", "MED")
    return df


@st.cache_data
def make_dummy_sales_history_3y(seed=21):
    """3년치 월별 판매량 더미: SKU별 36개월 시계열
    - SKU별 월 판매량을 2,000~2,800 정도로 생성 (현재 재고 합과 비교 가능)
    - 변동성은 완만하게 설정 (극단적 gap 방지)
    """
    rng = np.random.default_rng(seed)
    sku_pool = [f"SKU-{i:02d}" for i in range(1, 7)]
    months = pd.date_range(end=pd.Timestamp.today().normalize(), periods=36, freq="MS")

    rows = []
    for sku in sku_pool:
        base = rng.integers(2000, 2801)           # 월 평균
        season = rng.normal(0, 160, size=36)      # 완만한 계절성
        trend = np.linspace(-80, 120, 36)         # 완만한 추세
        qty = np.clip(base + season + trend, 1400, None).astype(int)
        for m, q in zip(months, qty):
            rows.append({"month": m, "sku_id": sku, "qty_sold": int(q)})
    return pd.DataFrame(rows)


def month_name_kr(m: int) -> str:
    return f"{m}월"


# ----------------------------
# Buttons
# ----------------------------
b1, b2, b3 = st.columns(3, gap="large")
_ = b1.button("▶ Auto Scan 실행", use_container_width=True)
_ = b2.button("▶ Auto Gap Report 생성", use_container_width=True)
_ = b3.button("▶ Auto Plan 생성", use_container_width=True)

df_base = make_dummy_gap_data()
df = apply_vision_rule(df_base, use_inner_rule=use_inner_rule)
sales_3y = make_dummy_sales_history_3y()

st.write("")
tab1, tab2, tab3 = st.tabs(["Auto Scan 결과", "Auto Gap Report", "Auto Plan"])

# ----------------------------
# Tab 1
# ----------------------------
with tab1:
    st.markdown("### Auto Scan (YOLO → bin별 실사 재고 추정)")
    st.caption("실사 재고(vision_total)는 카메라 결과(vision_front) + (옵션) 규칙 기반 inner 추정으로 계산됩니다.")
    show = df[["bin_id", "sku_id", "vision_front", "vision_inner", "vision_total", "confidence"]].copy()
    st.dataframe(show, use_container_width=True, height=360)

# ----------------------------
# Tab 2
# ----------------------------
with tab2:
    st.markdown("### Auto Gap Report (ERP vs 실사 격차 자동 분석)")
    st.caption("격차(|Gap|)가 크거나, 실사 총량이 임계치(min) 미만인 bin을 우선순위로 표시합니다.")

    report = df.copy()
    report["low_stock"] = report["vision_total"] < min_threshold
    report["abs_gap"] = report["gap"].abs()

    report = report[["bin_id", "sku_id", "erp_stock", "vision_total", "gap", "low_stock", "confidence", "abs_gap"]]
    report = report.sort_values(by=["low_stock", "abs_gap"], ascending=[False, False])

    st.dataframe(report.drop(columns=["abs_gap"]), use_container_width=True, height=360)

    low_cnt = int(report["low_stock"].sum())
    big_gap = report.sort_values("abs_gap", ascending=False).head(5)
    k1, k2 = st.columns(2, gap="large")
    with k1:
        st.metric("부족(bin) 개수", f"{low_cnt}개", f"min<{min_threshold}")
    with k2:
        st.metric("|Gap| 상위(Top5)", f"{len(big_gap)}개", "ERP vs Vision")

# ----------------------------
# Tab 3 (차트 개선: 면적 -> SKU별 3개 막대)
# ----------------------------
with tab3:
    st.markdown("### Auto Plan (3년 평균 월 판매량 vs 현재 재고: 실사/ERP)")
    st.caption("선택한 월의 ‘3년 평균 판매량’과 현재 재고(실사/ERP)를 비교해, 부족이 큰 SKU Top5를 우선생산으로 정리합니다.")

    month_sel = st.selectbox("비교할 월 선택", list(range(1, 13)), index=11)
    st.caption(f"선택 기준: **3년 평균 {month_name_kr(month_sel)} 판매량**")

    s = sales_3y.copy()
    s["m"] = s["month"].dt.month
    month_sales = s[s["m"] == month_sel]

    avg_month_sales = (month_sales.groupby("sku_id", as_index=False)["qty_sold"]
                       .mean()
                       .rename(columns={"qty_sold": "avg_sales"}))

    sku_stock = (df.groupby("sku_id", as_index=False)
                 .agg(vision_stock=("vision_total", "sum"),
                      erp_stock=("erp_stock", "sum")))

    plan = (avg_month_sales.merge(sku_stock, on="sku_id", how="left").fillna(0))

    # 안전장치(요청사항): ERP는 항상 실사보다 크게 보이도록
    # (모든 bin에서 이미 ERP>실사지만, 혹시 설정 변경/확장 시에도 깨지지 않게 한 번 더 보정)
    plan["erp_stock"] = np.maximum(plan["erp_stock"], plan["vision_stock"] + 1)

    # 부족(양수일 때만)
    plan["shortage"] = np.maximum(0, plan["avg_sales"] - plan["vision_stock"])
    plan["gap_abs_erp_vs_vision"] = (plan["erp_stock"] - plan["vision_stock"]).abs()

    # 차트용 long-form
    long = plan.melt(
        id_vars=["sku_id"],
        value_vars=["avg_sales", "erp_stock", "vision_stock"],
        var_name="metric",
        value_name="value"
    )

    labels = {
        "avg_sales": f"3년 평균 {month_name_kr(month_sel)} 판매량",
        "erp_stock": "현재 재고(ERP)",
        "vision_stock": "현재 실재고(실사, Vision)",
    }
    long["metric_label"] = long["metric"].map(labels)

    st.markdown("#### 1) SKU별 막대그래프(3개): 판매량 / ERP / 실사")
    if alt is None:
        st.warning("Altair가 설치되어 있지 않아 표로만 보여줘. (Streamlit Cloud에서는 requirements.txt로 설치됨)")
        st.dataframe(long, use_container_width=True, height=320)
    else:
        # grouped bar chart (xOffset으로 SKU당 3개 막대)
        order = ["3년 평균 " + month_name_kr(month_sel) + " 판매량", "현재 재고(ERP)", "현재 실재고(실사, Vision)"]
        chart = (
            alt.Chart(long)
            .mark_bar()
            .encode(
                x=alt.X("sku_id:N", title="SKU", sort=plan["sku_id"].tolist()),
                xOffset=alt.XOffset("metric_label:N", sort=order),
                y=alt.Y("value:Q", title="Units"),
                color=alt.Color("metric_label:N", title="지표", sort=order),
                tooltip=[
                    alt.Tooltip("sku_id:N", title="SKU"),
                    alt.Tooltip("metric_label:N", title="지표"),
                    alt.Tooltip("value:Q", title="수량", format=",")
                ],
            )
            .properties(height=380)
        )
        st.altair_chart(chart, use_container_width=True)

    st.write("")
    st.markdown("#### 2) Top5 우선 생산(부족 큰 순서)")
    top5 = plan.sort_values(["shortage", "gap_abs_erp_vs_vision"], ascending=[False, False]).head(5).copy()

    view = top5[["sku_id", "avg_sales", "vision_stock", "erp_stock", "shortage"]].copy()
    view = view.rename(columns={
        "avg_sales": f"3년 평균 {month_name_kr(month_sel)} 판매량",
        "vision_stock": "현재 실재고(실사)",
        "erp_stock": "현재 재고(ERP)",
        "shortage": "부족(평균판매-실사)"
    })

    for c in view.columns:
        if c != "sku_id":
            view[c] = view[c].map(lambda x: f"{int(round(x)):,}")

    st.dataframe(view, use_container_width=True, height=250)

    st.markdown("#### 3) 전체 SKU 테이블(부족/갭 큰 순서)")
    all_view = plan.rename(columns={
        "avg_sales": f"3년 평균 {month_name_kr(month_sel)} 판매량",
        "vision_stock": "현재 실재고(실사)",
        "erp_stock": "현재 재고(ERP)",
        "shortage": "부족(평균판매-실사)",
        "gap_abs_erp_vs_vision": "|ERP-실사|"
    }).sort_values(by=["부족(평균판매-실사)", "|ERP-실사|"], ascending=[False, False])

    for c in all_view.columns:
        if c != "sku_id":
            all_view[c] = all_view[c].map(lambda x: f"{int(round(x)):,}")

    st.dataframe(all_view, use_container_width=True, height=340)

st.markdown("<hr/>", unsafe_allow_html=True)

st.markdown("## Expected Impact (기대 효과)")
st.markdown(
    """
<div class="section">
- <b>재고 신뢰도 개선</b>: ERP–실사 격차를 자동 탐지해 정합성 유지<br/>
- <b>계획 정밀도 향상</b>: 평균 판매량 대비 부족 SKU를 빠르게 파악해 생산 우선순위를 명확화<br/>
- <b>운영 자동화</b>: 재고 점검/리포트/우선생산 리스트의 상시 자동 생성으로 업무 부담 절감<br/>
- <b>리스크 감소</b>: 결품·라인스톱·과잉재고 가능성 사전 완화
</div>
""",
    unsafe_allow_html=True,
)

st.caption("※ 이 페이지는 브레인스토밍/컨셉 검증용이며, 실제 모델 추론 대신 더미 데이터로 흐름을 시각화합니다.")
