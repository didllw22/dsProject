# ============================================================
#  세대 간 자산 형성 구조 변화: 근로소득 vs 부동산 디커플링 분석
#  Streamlit Dashboard - app.py (Monthly Resolution + Full UI/UX)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_percentage_error
import altair as alt
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="나홀로 집값 분석",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }

.main-title {
    font-size:2.2rem; font-weight:800; color:#1a1a2e;
    border-bottom:3px solid #e94560; padding-bottom:0.4rem; margin-bottom:1rem;
}
.sub-title { font-size:1rem; color:#555; margin-bottom:3rem; }
.metric-card {
    background:linear-gradient(135deg,#1a1a2e,#16213e);
    border-radius:12px; padding:1.5rem 1rem;
    text-align:center; border-left:4px solid #e94560;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    color:white;
    width:100%;
    height: 140px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-sizing: border-box;
    overflow: hidden;
}
.metric-value { font-size:1.8rem; font-weight:800; color:#e94560; margin-bottom:0.4rem;}
.metric-label { font-size:0.95rem; font-weight:600; color:#a0a0b0; }
.metric-sub   { font-size:0.8rem; color:#666; margin-top:0.4rem; }
.section-header {
    font-size:1.25rem; font-weight:700; color:#16213e;
    margin:2.5rem 0 1rem 0; padding-left:0.8rem;
    border-left:5px solid #e94560;
}
.info-box {
    background:#f0f4ff; border-radius:8px;
    padding:1rem 1.2rem; font-size:0.95rem; color:#333; margin-bottom:1.5rem;
}
.formula-box {
    background:#1a1a2e; border-radius:8px; padding:1.5rem;
    color:#e94560; font-size:1.1rem; text-align:center;
    font-family:monospace; font-weight:600; margin-bottom:1.5rem;
}
.scenario-card {
    border-radius:12px; padding:1.5rem; text-align:center; color:white;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
/* ── 용어 툴팁 ── */
.term {
    border-bottom: 1.5px dashed #e94560;
    cursor: help;
    position: relative;
    display: inline-block;
    font-weight: 600;
    color: #e94560;
}
.term .tip {
    visibility: hidden;
    opacity: 0;
    width: 280px;
    background: #1a1a2e;
    color: #e0e0f0;
    font-size: 0.82rem;
    font-weight: 400;
    line-height: 1.55;
    text-align: left;
    border-radius: 8px;
    padding: 10px 14px;
    position: absolute;
    z-index: 9999;
    bottom: 130%;
    left: 50%;
    transform: translateX(-50%);
    box-shadow: 0 6px 24px rgba(0,0,0,0.45);
    border: 1px solid rgba(233,69,96,0.35);
    transition: opacity 0.18s ease;
    pointer-events: none;
    white-space: normal;
}
.term .tip::after {
    content: '';
    position: absolute;
    top: 100%;
    left: 50%;
    margin-left: -6px;
    border-width: 6px;
    border-style: solid;
    border-color: #1a1a2e transparent transparent transparent;
}
.term:hover .tip {
    visibility: visible;
    opacity: 1;
}
/* 소득 산출 테이블 */
.income-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.93rem;
    margin-bottom: 1rem;
}
.income-table th {
    background: #1a1a2e;
    color: #a0a0c0;
    padding: 8px 14px;
    text-align: left;
    font-weight: 600;
    border-bottom: 2px solid #e94560;
}
.income-table td {
    padding: 9px 14px;
    border-bottom: 1px solid #eee;
    color: #222;
    vertical-align: middle;
}
.income-table tr:hover td { background: #f8f0f2; }
.income-table td:last-child { font-weight: 700; color: #1a1a2e; text-align: right; }
.nav-card {
    background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 18px;
    padding: 2rem 1.2rem 1.6rem;
    text-align: center;
    border: 1px solid rgba(233,69,96,0.2);
    box-shadow: 0 6px 28px rgba(0,0,0,0.25);
    color: white;
    margin-bottom: 0.6rem;
    min-height: 260px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 0.35rem;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.nav-card-icon { font-size: 2.6rem; }
.nav-card-title { font-size: 1.0rem; font-weight: 700; color: #e0e0f0; margin-top: 0.3rem; }
.nav-card-metric { font-size: 1.85rem; font-weight: 900; color: #e94560; margin: 0.15rem 0; }
.nav-card-metric-label { font-size: 0.74rem; color: #a0a0b0; }
.nav-card-desc { font-size: 0.79rem; color: #8888aa; line-height: 1.5; margin-top: 0.45rem; padding: 0 0.2rem; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════
#  유틸리티 함수
# ════════════════════════════════════════════════

def parse_date_to_float(date_str):
    """날짜 문자열(년.월)을 float으로 변환 (예측 모델 입력용)"""
    try:
        parts = str(date_str).split('.')
        year  = int(parts[0])
        month = int(parts[1])
        return year + (month - 1) / 12.0
    except Exception:
        return 2000.0

def float_to_date_str(f):
    """float을 날짜 문자열(년.월)로 역변환"""
    year  = int(f)
    month = int(round((f - year) * 12) + 1)
    if month > 12:
        year += 1
        month = 1
    return f"{year}.{month:02d}"

def float_to_date_std(f):
    """float을 표준 날짜 문자열(YYYY-MM-DD)로 역변환"""
    year  = int(f)
    month = int(round((f - year) * 12) + 1)
    if month > 12:
        year += 1
        month = 1
    return f"{year}-{month:02d}-01"

def str_date_to_std(date_str):
    """YYYY.MM 문자열을 YYYY-MM-01 표준 날짜 문자열로 변환"""
    try:
        parts = str(date_str).split('.')
        year = int(parts[0])
        month = int(parts[1])
        return f"{year}-{month:02d}-01"
    except Exception:
        return str(date_str)

def apply_dark_theme(fig):
    """라이트 모드 유지를 위해 테마 설정을 변경하지 않고 그대로 반환"""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1a1a2e", family="Noto Sans KR"),
        xaxis=dict(
            gridcolor="rgba(0, 0, 0, 0.05)",
            zerolinecolor="rgba(0, 0, 0, 0.1)",
            linecolor="rgba(0, 0, 0, 0.1)"
        ),
        yaxis=dict(
            gridcolor="rgba(0, 0, 0, 0.05)",
            zerolinecolor="rgba(0, 0, 0, 0.1)",
            linecolor="rgba(0, 0, 0, 0.1)"
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_color="#1a1a2e",
            bordercolor="rgba(233, 69, 96, 0.3)"
        )
    )
    return fig

def smooth_series(values, window=3):
    s = pd.Series(values)
    return s.rolling(window=window, center=True, min_periods=1).mean().tolist()

def scenario_price(current_price, r, years=20):
    return round(current_price * ((1 + r) ** years), 2)



# ════════════════════════════════════════════════
#  데이터 레이어
# ════════════════════════════════════════════════

@st.cache_data
def load_price_data():
    try:
        df = pd.read_csv("price_data.csv", encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv("price_data.csv", encoding="cp949")
    except FileNotFoundError:
        st.error("❌ price_data.csv 파일을 찾을 수 없습니다.")
        st.stop()
    return df

@st.cache_data
def load_wage_data():
    try:
        df = pd.read_csv("wage_data.csv", encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv("wage_data.csv", encoding="cp949")
    except FileNotFoundError:
        st.error("❌ wage_data.csv 파일을 찾을 수 없습니다.")
        st.stop()
    return df

@st.cache_data
def load_pir_data(price_df, wage_df, area=84):
    """월별 데이터 기준 PIR 계산 (가구연소득 = 월임금 × 12 × 1.65)"""
    household_multiplier = 1.65
    merged = pd.merge(price_df, wage_df, on="날짜", how="inner")
    pir_df = pd.DataFrame({"날짜": merged["날짜"]})
    pir_df["가구연소득"] = merged["월임금"] * 12 * household_multiplier
    regions = [c for c in price_df.columns if c not in ["날짜", "날짜_표준"]]
    for region in regions:
        total_price = merged[region] * area          # 만원/㎡ × ㎡ = 만원
        pir_df["PIR_" + region] = total_price / pir_df["가구연소득"]
    return pir_df


# ════════════════════════════════════════════════
#  예측 모델
# ════════════════════════════════════════════════

def forecast_poly(years_float, values, target_year=2044, degree=2, add_cycle=False, only_noise=False):
    """2차 다항 회귀 예측. years_float = float 형태의 연도 리스트"""
    X = np.array(years_float).reshape(-1, 1)
    y = np.array(values, dtype=float)
    poly  = PolynomialFeatures(degree=degree)
    Xp    = poly.fit_transform(X)
    model = LinearRegression().fit(Xp, y)

    # 마지막 시점 다음 달부터 목표 연도 말까지 월 단위 예측
    future_floats = np.arange(years_float[-1] + 1/12, target_year + 1, 1/12)
    fut_vals      = model.predict(poly.transform(future_floats.reshape(-1, 1)))
    target_val    = model.predict(poly.transform([[target_year]]))[0]

    if add_cycle:
        # ── 데이터 기반 사이클 및 위상 자동 추정 ──
        hist_trend = model.predict(Xp).flatten()
        y_flat = y.flatten()
        rel_res = (y_flat - hist_trend) / hist_trend
        
        # 진폭(Amplitude) 추정 (RMS 기반)
        auto_amp_pct = np.clip(np.std(rel_res) * np.sqrt(2), 0.02, 0.30)
        
        # 주기(Period) 추정 (이동평균 영점 교차)
        import pandas as pd
        s_res = pd.Series(rel_res).rolling(12, center=True, min_periods=1).mean().values
        zero_crossings = np.where(np.diff(np.sign(s_res)))[0]
        if len(zero_crossings) >= 2:
            avg_cross_dist = np.mean(np.diff(zero_crossings)) # 반주기(개월)
            auto_period_years = np.clip((avg_cross_dist * 2) / 12.0, 3.0, 15.0)
        else:
            auto_period_years = 8.0
            
        # 현재 위상(Phase) 추정 (현재 상승장인지 하락장인지 매끄럽게 연결)
        current_rel = s_res[-1]
        prev_rel    = s_res[-2] if len(s_res) > 1 else 0
        phi = np.arcsin(np.clip(current_rel / auto_amp_pct, -0.99, 0.99))
        if current_rel < prev_rel: # 하락 구간
            phi = np.pi - phi
            
        current_year = years_float[-1]
        base_amplitude = values[-1] * auto_amp_pct
        
        # 1. 거시적 사이클 파동
        if only_noise:
            cycle_wave = 0
            target_cycle = 0
        else:
            cycle_wave = base_amplitude * np.sin(2 * np.pi * (future_floats - current_year) / auto_period_years + phi)
            target_cycle = base_amplitude * np.sin(2 * np.pi * (target_year - current_year) / auto_period_years + phi)
        
        # 2. 미시적 무작위 변동성 (실제 데이터와 같은 질감 부여)
        np.random.seed(int(target_year) + int(y_flat[0])) # 변수별로 다른 패턴
        hist_pct_change = np.diff(y_flat) / y_flat[:-1]
        volatility = np.std(hist_pct_change)
        
        noise = np.zeros(len(future_floats))
        noise[0] = np.random.normal(0, volatility * 0.5)
        for i in range(1, len(future_floats)):
            noise[i] = 0.9 * noise[i-1] + np.random.normal(0, volatility * 0.4)
            
        noise_wave = fut_vals.flatten() * noise
        
        # 예측값에 사이클과 노이즈 반영
        fut_vals = fut_vals.flatten() + cycle_wave + noise_wave
        target_noise = target_val * np.random.normal(0, volatility)
        target_val = target_val + target_cycle + target_noise

    # 연결용 (마지막 실제값 포함)
    conn_floats = np.concatenate([[years_float[-1]], future_floats])
    conn_vals   = np.concatenate([[values[-1]], fut_vals])
    return target_val, future_floats, fut_vals, conn_floats, conn_vals

def calc_required_income(future_price_manwon, dsr=0.40, ltv=0.60,
                          loan_rate=0.045, loan_years=30):
    """DSR 기반 필요 연소득 산출 (만원 단위)"""
    loan_amount   = future_price_manwon * ltv
    mr = loan_rate / 12
    n  = loan_years * 12
    mp = loan_amount * (mr * (1 + mr)**n) / ((1 + mr)**n - 1)
    annual_payment = mp * 12
    required_income = annual_payment / dsr
    self_equity     = future_price_manwon * (1 - ltv)
    return {
        "총매매가_만원":    round(future_price_manwon),
        "필요자기자본_만원": round(self_equity),
        "연간원리금_만원":  round(annual_payment),
        "필요연소득_만원":  round(required_income),
        "필요월소득_만원":  round(required_income / 12),
    }

def calc_income_by_pir(future_price_manwon, pir_target=12):
    """PIR 기준 필요 가구 연소득 (만원)"""
    return round(future_price_manwon / pir_target)

def forecast_backtest(years_float, values, test_years=3, degree=2):
    """최근 데이터를 테스트셋으로 빼서 모델의 과거 예측 정확도를 검증(백테스트)합니다."""
    # 월 단위 기준 (test_years * 12 개월)
    split_idx = int(test_years * 12)
    
    # 훈련용(과거)과 테스트용(최근 3년) 데이터 분리
    train_x = np.array(years_float[:-split_idx]).reshape(-1, 1)
    train_y = np.array(values[:-split_idx], dtype=float)
    test_x  = np.array(years_float[-split_idx:]).reshape(-1, 1)
    test_y  = np.array(values[-split_idx:], dtype=float)
    
    # 과거 데이터로만 다항 회귀 모델 학습
    poly = PolynomialFeatures(degree=degree)
    model = LinearRegression().fit(poly.fit_transform(train_x), train_y)
    
    # 최근 3년 구간 예측
    pred_y = model.predict(poly.transform(test_x))
    
    # 오차율(MAPE) 계산
    mape = mean_absolute_percentage_error(test_y, pred_y) * 100
    
    return mape, test_x.flatten(), test_y, pred_y


# ════════════════════════════════════════════════
#  사이드바
# ════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🏙️ 나홀로 집값 분석")
    st.markdown("**세대 간 자산 형성 구조 변화**")
    st.markdown("---")
    st.markdown("### ⚙️ 분석 설정")

    # price_data.csv 컬럼에서 동적으로 지역 목록 로드 (날짜 컬럼 제외)
    try:
        _tmp_price = pd.read_csv("price_data.csv", encoding="utf-8-sig", nrows=0)
    except Exception:
        try:
            _tmp_price = pd.read_csv("price_data.csv", encoding="cp949", nrows=0)
        except Exception:
            _tmp_price = pd.DataFrame(columns=["날짜"])
    REGION_LIST = [c for c in _tmp_price.columns if c != "날짜"]
    if not REGION_LIST:
        # fallback 목록
        REGION_LIST = [
            "전국", "서울특별시", "서울 용산구", "서울 성동구", "서울 마포구",
            "서울 서초구", "서울 강남구", "서울 송파구", "서울 강동구", "서울 노원구",
            "부산광역시", "대구광역시", "인천광역시", "인천 연수구", "세종특별자치시",
            "경기도", "경기 수원시", "경기 성남시", "경기 성남 분당구",
            "경기 고양시", "경기 과천시", "경기 광명시", "경기 하남시",
            "경기 화성시", "경기 김포시", "제주특별자치도",
        ]

    selected_region = st.selectbox("분석 지역", REGION_LIST, index=6)
    target_year     = st.slider("예측 목표 연도", 2030, 2050, 2044, step=1)
    area            = st.selectbox(
        "아파트 면적", 
        [20, 33, 49, 59, 74, 84, 99, 114, 132, 150, 180, 220], 
        index=5, 
        format_func=lambda x: f"{x}㎡ (약 {int(x / 3.3058)}평)"
    )
    area_str        = f"{area}㎡(약 {int(area/3.3058)}평)"
    pir_target      = st.slider("목표 PIR (사회적 합의)", 8, 20, 12)

    st.markdown("---")
    st.markdown("### 💡 DSR 설정")
    dsr        = st.slider("DSR 한도 (%)", 30, 50, 40) / 100
    ltv        = st.slider("LTV (%)", 40, 80, 60) / 100
    loan_rate  = st.slider("대출 금리 (%)", 2.0, 8.0, 4.5, step=0.1) / 100
    loan_years = st.selectbox("대출 기간 (년)", [10, 15, 20, 25, 30, 35, 40, 45, 50], index=4)

    st.markdown("---")
    st.markdown("### 💰 나의 자산 설정")
    my_asset = st.number_input("현재 보유 자산 (만원)", min_value=0, value=5000, step=1000)
    my_savings = st.number_input("월 저축 가능액 (만원)", min_value=0, value=150, step=10)

    st.markdown("---")
    st.markdown("### 🌊 부동산 시뮬레이션 (데이터 기반 자동 분석)")
    use_cycle = st.checkbox("과거 추세 기반 사이클/노이즈 자동 반영", value=True, help="과거 실거래가의 등락폭과 주기를 분석하여 미래 예측에 자동으로 적용합니다.")

    st.markdown("---")
    st.markdown("**📊 데이터 출처**")
    st.markdown("- KB부동산 데이터Hub (아파트 평당 매매 평균)")
    st.markdown("- 고용노동부 월간 임금 통계")
    st.success("✅ 월별 실거래 데이터 연동됨")


# ════════════════════════════════════════════════
#  데이터 로드 및 전처리
# ════════════════════════════════════════════════

price_df = load_price_data()
wage_df  = load_wage_data()

# CSV에서 날짜가 float(2013.04)으로 읽힐 수 있으므로 문자열로 강제 변환
price_df["날짜"] = price_df["날짜"].apply(lambda x: f"{x:.2f}" if isinstance(x, float) else str(x))
wage_df["날짜"]  = wage_df["날짜"].apply(lambda x: f"{x:.2f}" if isinstance(x, float) else str(x))

pir_df   = load_pir_data(price_df, wage_df, area=area)
pir_df["날짜"] = pir_df["날짜"].apply(lambda x: f"{x:.2f}" if isinstance(x, float) else str(x))

# PIR 데이터 계산 이후에 공통적으로 날짜_표준 생성 (KeyError 방지)
price_df["날짜_표준"] = price_df["날짜"].apply(str_date_to_std)
wage_df["날짜_표준"]  = wage_df["날짜"].apply(str_date_to_std)
pir_df["날짜_표준"]   = pir_df["날짜"].apply(str_date_to_std)

# 가격: 만원/㎡ → 면적 기준 총 가격(만원)
price_vals       = (price_df[selected_region] * area).tolist()   # 만원
wage_annual_vals = (wage_df["월임금"] * 12).tolist()             # 만원 (연환산)

price_date_list = price_df["날짜"].tolist()
wage_date_list  = wage_df["날짜"].tolist()

price_dates_std_list = price_df["날짜_표준"].tolist()
wage_dates_std_list  = wage_df["날짜_표준"].tolist()

price_floats = price_df["날짜"].apply(parse_date_to_float).tolist()
wage_floats  = wage_df["날짜"].apply(parse_date_to_float).tolist()

# CAGR (연 환산)
num_years_price = (len(price_df) - 1) / 12.0
num_years_wage  = (len(wage_df)  - 1) / 12.0
price_cagr = ((price_vals[-1] / price_vals[0]) ** (1 / num_years_price) - 1) * 100
wage_cagr  = ((wage_annual_vals[-1] / wage_annual_vals[0]) ** (1 / num_years_wage) - 1) * 100

# 예측 (집값은 과거 데이터 기반 사이클 자동 적용)
pred_price_manwon, fut_fp, fut_vp, conn_fp, conn_vp = forecast_poly(
    price_floats, price_vals, target_year, 
    add_cycle=use_cycle
)
pred_wage_manwon,  fut_fw, fut_vw, conn_fw, conn_vw = forecast_poly(
    wage_floats, wage_annual_vals, target_year,
    add_cycle=use_cycle, only_noise=True # 임금은 거시 사이클 파동 없이 미시 노이즈만 반영
)

# 미래 날짜 문자열
fut_fp_str   = [float_to_date_str(y) for y in fut_fp]
fut_fw_str   = [float_to_date_str(y) for y in fut_fw]
conn_fp_str  = [float_to_date_str(y) for y in conn_fp]
conn_fw_str  = [float_to_date_str(y) for y in conn_fw]

fut_fp_std_str  = [float_to_date_std(y) for y in fut_fp]
fut_fw_std_str  = [float_to_date_std(y) for y in fut_fw]
conn_fp_std_str = [float_to_date_std(y) for y in conn_fp]
conn_fw_std_str = [float_to_date_std(y) for y in conn_fw]


# 적정 소득 산출
income_result = calc_required_income(pred_price_manwon, dsr, ltv, loan_rate, loan_years)
income_pir    = calc_income_by_pir(pred_price_manwon, pir_target)

current_pir       = pir_df["PIR_" + selected_region].iloc[-1]
current_wage      = wage_annual_vals[-1]   # 연환산 만원
latest_wage_date  = wage_date_list[-1]
lack_ratio        = round(income_pir / current_wage, 1)
dsr_pct           = int(dsr * 100)
ltv_pct           = int(ltv * 100)
pred_price_eok    = pred_price_manwon / 10000   # 억원
pred_pir_future   = pred_price_manwon / (pred_wage_manwon * 1.65)

COLORS = ["#e94560", "#0f3460", "#533483"]


# ── 차트 지역 선택 세션 초기화
if "selected_chart_regions" not in st.session_state:
    _default = [r for r in ["전국", "서울특별시", "경기도", "부산광역시", "인천광역시", "서울 강남구"]
                if r in REGION_LIST]
    if not _default:
        _default = REGION_LIST[:6]
    st.session_state.selected_chart_regions = list(_default)

# 사이드바 선택 지역은 항상 포함
if selected_region not in st.session_state.selected_chart_regions:
    st.session_state.selected_chart_regions.append(selected_region)


# ════════════════════════════════════════════════
#  헤더
# ════════════════════════════════════════════════

st.markdown('<div class="main-title">🏙️ 나홀로 집값 분석</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-title">근로소득과 부동산 자산의 격차 분석 및 {target_year}년 미래 적정 소득 예측 모델링 | '
    f'월별 실거래 데이터 기반 · {selected_region} · {area_str}</div>',
    unsafe_allow_html=True
)

# ── KPI 카드
col1, col2, col3, col4 = st.columns(4, gap="large")
with col1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{income_result['필요연소득_만원']:,}만원</div>
        <div class="metric-label">{target_year}년 필요 연소득 (DSR {dsr_pct}%)</div>
        <div class="metric-sub">{selected_region} · {area_str} 기준</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{pred_price_eok:.1f}억원</div>
        <div class="metric-label">{target_year}년 예상 아파트가 ({area_str})</div>
        <div class="metric-sub">{selected_region} 기준</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{lack_ratio}배</div>
        <div class="metric-label">현재 대비 소득 부족 배율</div>
        <div class="metric-sub">PIR {pir_target}배 기준 · 근로소득만으로 불가능한 영역</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{current_pir:.1f}배</div>
        <div class="metric-label">현재 PIR ({latest_wage_date} 기준)</div>
        <div class="metric-sub">목표 {pir_target}배 기준</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:2.5rem;'></div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════
#  카테고리 네비게이션
# ════════════════════════════════════════════════

if "active_section" not in st.session_state:
    st.session_state.active_section = None

SECTIONS = [
    {"id": "trend",      "icon": "📈", "title": "시계열 추세",    "metric": f"{price_cagr:.1f}%",         "metric_label": f"주택가격 CAGR ({num_years_price:.0f}년)",  "desc": "지역별 아파트 매매가 추이와 임금 상승률을 월 단위로 비교합니다."},
    {"id": "decoupling", "icon": "🔍", "title": "나홀로 집값 분석",  "metric": f"{current_pir:.1f}배",        "metric_label": f"현재 PIR ({selected_region})",             "desc": "소득-자산 격차를 수치화하고 전국 PIR 추이를 분석합니다."},
    {"id": "forecast",   "icon": "🔮", "title": "미래 예측",      "metric": f"{pred_price_eok:.1f}억",     "metric_label": f"{target_year}년 예상 매매가",              "desc": "월별 다항 회귀 모델로 미래 가격과 임금을 예측합니다."},
    {"id": "income",     "icon": "💰", "title": "적정 소득 산출", "metric": f"{lack_ratio}배",             "metric_label": "소득 부족 배율",                           "desc": "PIR / DSR 기준 미래 필요 소득과 자기자본을 산출합니다."},
    {"id": "scenario",   "icon": "📊", "title": "시나리오 분석",  "metric": "3가지",                       "metric_label": "성장률 시나리오",                          "desc": "비관·표준·낙관 시나리오별 민감도를 분석합니다."},
]


# ═══════════════════════════════════════════════════════════════════════════
#  홈 화면 — 카드 그리드
# ═══════════════════════════════════════════════════════════════════════════
if st.session_state.active_section is None:
    st.markdown("""
    <div style='margin-bottom:1.2rem;'>
        <span style='font-size:1.15rem;font-weight:800;color:#1a1a2e;'>📂 분석 카테고리</span>
        <span style='font-size:0.88rem;color:#999;margin-left:0.8rem;'>카드를 클릭하면 해당 분석으로 이동합니다.</span>
    </div>""", unsafe_allow_html=True)

    _card_cols = st.columns(5, gap="medium")
    for _i, _sec in enumerate(SECTIONS):
        with _card_cols[_i]:
            st.markdown(f"""
            <div class="nav-card">
                <div class="nav-card-icon">{_sec['icon']}</div>
                <div class="nav-card-title">{_sec['title']}</div>
                <div class="nav-card-metric">{_sec['metric']}</div>
                <div class="nav-card-metric-label">{_sec['metric_label']}</div>
                <div class="nav-card-desc">{_sec['desc']}</div>
            </div>""", unsafe_allow_html=True)
            if st.button("분석 시작 →", key=f"nav_{_sec['id']}", width='stretch'):
                st.session_state.active_section = _sec["id"]
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
#  개별 분석 화면
# ═══════════════════════════════════════════════════════════════════════════
else:
    # ── 상단 네비게이션 바
    _nav_c = st.columns([1] + [2] * 5, gap="small")
    with _nav_c[0]:
        if st.button("← 목록", key="back_btn", width='stretch'):
            st.session_state.active_section = None
            st.rerun()
    for _i, _sec in enumerate(SECTIONS):
        with _nav_c[_i + 1]:
            _is_active = st.session_state.active_section == _sec["id"]
            if st.button(
                f"{_sec['icon']} {_sec['title']}",
                key=f"topnav_{_sec['id']}",
                width='stretch',
                type="primary" if _is_active else "secondary"
            ):
                if not _is_active:
                    st.session_state.active_section = _sec["id"]
                    st.rerun()
    st.markdown("---")
    _active = st.session_state.active_section


    # ── 📈 시계열 추세 ─────────────────────────────────────────────────────
    if _active == "trend":
        st.markdown('<div class="section-header">전국 지역별 아파트 매매가 추이 (만원/㎡)</div>', unsafe_allow_html=True)

        # 지역 선택 컨트롤
        with st.expander("📍 차트에 표시할 지역 선택", expanded=True):
            avail_regions = [r for r in REGION_LIST if r not in st.session_state.selected_chart_regions]
            c1, c2, c3 = st.columns([4, 1.2, 1.2])
            with c1:
                if avail_regions:
                    region_to_add = st.selectbox("추가할 지역", avail_regions, label_visibility="collapsed", key="chart_add_sel")
                else:
                    st.markdown("✅ 모든 지역이 선택되어 있습니다"); region_to_add = None
            with c2:
                if st.button("➕ 추가", key="chart_add_btn", width='stretch'):
                    if region_to_add and region_to_add not in st.session_state.selected_chart_regions:
                        st.session_state.selected_chart_regions.append(region_to_add); st.rerun()
            with c3:
                if st.button("🔄 초기화", key="chart_reset_btn", width='stretch'):
                    _def = [r for r in ["전국", "서울특별시", "경기도", "부산광역시", "인천광역시", "서울 강남구"] if r in REGION_LIST]
                    st.session_state.selected_chart_regions = list(_def) if _def else REGION_LIST[:6]; st.rerun()
            st.markdown("**현재 선택된 지역** (➖ 클릭 시 제거):")
            _active_r = list(st.session_state.selected_chart_regions)
            _nc = min(len(_active_r), 7)
            if _nc > 0:
                _rcols = st.columns(_nc)
                for _ri, _rgn in enumerate(_active_r):
                    with _rcols[_ri % _nc]:
                        if st.button(f"➖ {_rgn}", key=f"rm_region_{_rgn}", width='stretch'):
                            if len(st.session_state.selected_chart_regions) > 1:
                                st.session_state.selected_chart_regions.remove(_rgn); st.rerun()

        st.markdown(f"""<div class="info-box">📌 범례에서 지역을 클릭하여 특정 지역선만 켜고 끌 수 있습니다. · 가격 단위: 억원 ({area}㎡ 기준)</div>""", unsafe_allow_html=True)

        display_regions = [r for r in REGION_LIST if r in st.session_state.selected_chart_regions]
        fig_trend = go.Figure()
        
        for rgn in display_regions:
            rgn_price_eok = [(v * area / 10000) for v in price_df[rgn]]
            fig_trend.add_trace(
                go.Scatter(
                    x=price_dates_std_list,
                    y=rgn_price_eok,
                    name=rgn,
                    mode="lines",
                    line=dict(width=2.2),
                    hovertemplate="%{x|%Y년 %m월}<br>" + f"{rgn}: %" + "{y:.2f}억원<extra></extra>"
                )
            )
            
        fig_trend.update_xaxes(
            type="date",
            tickformat="%Y년",
            dtick="M24",
            title_text="연도"
        )
        fig_trend.update_yaxes(title_text=f"매매가 (억원, {area}㎡)")
        fig_trend.update_layout(
            title=dict(text=f"<b>지역별 아파트 매매가 추이 ({area}㎡ 기준)</b>", x=0.5, font=dict(size=16)),
            height=500,
            margin=dict(t=80, b=50, l=60, r=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.28,
                x=0.5,
                xanchor="center",
                font=dict(size=11)
            ),
            hovermode="x unified"
        )
        apply_dark_theme(fig_trend)
        st.plotly_chart(fig_trend, use_container_width=True)

        col_l, col_r = st.columns(2, gap="large")
        with col_l:
            st.markdown('<div class="section-header">월 평균 임금 추이 (만원)</div>', unsafe_allow_html=True)
            fig_wage = px.area(wage_df, x="날짜_표준", y="월임금",
                               color_discrete_sequence=["#0f3460"],
                               labels={"월임금": "월임금 (만원)", "날짜_표준": "날짜"})
            fig_wage.update_xaxes(type="date", tickformat="%Y년", dtick="M24")
            fig_wage.update_layout(height=350, showlegend=False)
            apply_dark_theme(fig_wage)
            fig_wage.update_traces(
                fill='tozeroy',
                fillcolor="rgba(15, 52, 96, 0.15)",
                line=dict(color="#0f3460", width=2.5)
            )
            st.plotly_chart(fig_wage, use_container_width=True)

        with col_r:
            st.markdown('<div class="section-header">전년 동월 대비 상승률 비교 (YoY, %)</div>', unsafe_allow_html=True)
            yoy_price = price_df[selected_region].pct_change(12) * 100
            yoy_wage  = wage_df["월임금"].pct_change(12) * 100
            
            fig_yoy = go.Figure()
            fig_yoy.add_trace(go.Bar(
                x=price_df["날짜_표준"][12:],
                y=yoy_price[12:],
                name=f"{selected_region} 아파트가격 YoY",
                marker_color="#e94560",
                opacity=0.85
            ))
            fig_yoy.add_trace(go.Bar(
                x=wage_df["날짜_표준"][12:],
                y=yoy_wage[12:],
                name="임금 YoY",
                marker_color="#0f3460",
                opacity=0.85
            ))
            fig_yoy.update_xaxes(type="date", tickformat="%Y년", dtick="M24")
            fig_yoy.update_layout(
                height=350,
                barmode="group",
                yaxis_title="%",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center")
            )
            apply_dark_theme(fig_yoy)
            st.plotly_chart(fig_yoy, use_container_width=True)


    # ── 🔍 디커플링 분석 ───────────────────────────────────────────────────
    elif _active == "decoupling":
        st.markdown('<div class="section-header">소득-자산 격차(Gap) 시각화</div>', unsafe_allow_html=True)
        st.markdown("""<div class="formula-box">
        Gap = ∫ ( P_housing − I_wage ) dt<br><br>
        <span style="font-size:0.9rem;color:#ccc;font-weight:400;">부동산 자산 = 복리 성장 + 금융 레버리지 → 기하급수적 팽창<br>근로소득 = 선형 성장</span>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">누적 상승 지수 비교 (최초 시점 = 100 기준)</div>', unsafe_allow_html=True)

        # 가격/임금 공통 시작 날짜 기준 정렬
        base_date = price_df["날짜"].iloc[0]
        pf = price_df[price_df["날짜"] >= base_date]
        wf = wage_df[wage_df["날짜"] >= base_date]
        base_p = pf[selected_region].iloc[0]
        base_w = wf["월임금"].iloc[0]
        idx_price = (pf[selected_region] / base_p * 100).tolist()
        idx_wage  = (wf["월임금"] / base_w * 100).tolist()

        fig_idx = go.Figure()
        fig_idx.add_trace(go.Scatter(
            x=pf["날짜_표준"], y=idx_price,
            name=selected_region + " 주택가격",
            line=dict(color="#e94560", width=3),
            fill="tozeroy", fillcolor="rgba(233, 69, 96, 0.08)"
        ))
        fig_idx.add_trace(go.Scatter(
            x=wf["날짜_표준"], y=idx_wage,
            name="연평균 임금",
            line=dict(color="#0f3460", width=3),
            fill="tozeroy", fillcolor="rgba(15, 52, 96, 0.1)"
        ))
        fig_idx.update_xaxes(type="date", tickformat="%Y년", dtick="M24")
        fig_idx.update_layout(
            height=450,
            yaxis_title=f"지수 ({base_date}=100)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center")
        )
        apply_dark_theme(fig_idx)
        st.plotly_chart(fig_idx, use_container_width=True)

        st.markdown('<div class="section-header">전국 지역별 PIR (Price-to-Income Ratio) 월별 추이</div>', unsafe_allow_html=True)
        st.markdown("""<div class="info-box">📌 범례에서 지역을 클릭하여 특정 지역선만 켜고 끌 수 있습니다. · 시계열 탭에서 선택한 지역 필터가 동일하게 적용됩니다.<br>
        🟠 주황 점선: 적정 PIR (5배) &nbsp;|&nbsp; 🔴 빨강 점선: 위험 PIR (10배)</div>""", unsafe_allow_html=True)

        pir_cols = [c for c in pir_df.columns if c.startswith("PIR_")]
        _pir_disp = [r for r in st.session_state.selected_chart_regions if "PIR_" + r in pir_cols]
        
        fig_pir_all = go.Figure()
        for rgn in _pir_disp:
            fig_pir_all.add_trace(go.Scatter(
                x=pir_df["날짜_표준"],
                y=pir_df["PIR_" + rgn],
                name=rgn,
                mode="lines",
                line=dict(width=2),
                hovertemplate="%{x|%Y년 %m월}<br>" + f"{rgn} PIR: %" + "{y:.1f}배<extra></extra>"
            ))
            
        fig_pir_all.add_hline(y=5,  line_dash="dash", line_color="orange", line_width=1.5)
        fig_pir_all.add_hline(y=10, line_dash="dash", line_color="red", line_width=1.5)
        
        # 주석 위치 조정
        fig_pir_all.add_annotation(x=pir_df["날짜_표준"].iloc[-1], y=5.3, text="적정 PIR (5배)", showarrow=False, font=dict(color="orange", size=10), xanchor="right")
        fig_pir_all.add_annotation(x=pir_df["날짜_표준"].iloc[-1], y=10.3, text="위험 PIR (10배)", showarrow=False, font=dict(color="red", size=10), xanchor="right")
        
        fig_pir_all.update_xaxes(type="date", tickformat="%Y년", dtick="M24")
        fig_pir_all.update_yaxes(title_text="PIR (배)")
        fig_pir_all.update_layout(
            height=500,
            margin=dict(t=60, b=50, l=60, r=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.28,
                x=0.5,
                xanchor="center",
                font=dict(size=11)
            ),
            hovermode="x unified"
        )
        apply_dark_theme(fig_pir_all)
        st.plotly_chart(fig_pir_all, use_container_width=True)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric(f"주택가격 CAGR ({num_years_price:.0f}년)", f"{price_cagr:.2f}%")
        col_b.metric(f"임금 CAGR ({num_years_wage:.0f}년)",     f"{wage_cagr:.2f}%")
        col_c.metric("나홀로 집값 배율", f"{price_cagr/wage_cagr:.1f}x",
                     delta=f"+{price_cagr - wage_cagr:.2f}%p 격차")


    # ── 🔮 미래 예측 ────────────────────────────────────────────────────────
    elif _active == "forecast":
        st.markdown(f'<div class="section-header">{selected_region} — {target_year}년까지 예측 (월별 2차 다항 회귀)</div>', unsafe_allow_html=True)
        st.markdown("""<div class="info-box">📌 <b>차트 읽는 법</b>: 실선 = 실제 데이터 · 점선 = 회귀 예측 · 두 선은 마지막 실제값에서 자연스럽게 연결됩니다.<br>
        자산 가격과 근로 소득의 나홀로 집값(괴리)을 직관적으로 확인하기 위해 <b>이중 Y축 통합 차트</b>로 제공됩니다.<br>
        좌측 Y축: 아파트 예상 매매가 (억원, 면적 기준) &nbsp;|&nbsp; 우측 Y축: 연평균 임금 (만원)</div>""", unsafe_allow_html=True)

        # 백테스트 실행 (최근 3년 기준)
        mape_err, bt_x, bt_true, bt_pred = forecast_backtest(price_floats, price_vals, test_years=3)
        st.markdown(f"""<div style="background-color:rgba(15, 52, 96, 0.1); padding:1rem; border-radius:8px; margin-bottom:1.5rem; border-left:4px solid #0f3460;">
        📊 <b>예측 모델 검증 (Backtesting)</b><br>
        이 모델은 과거 데이터를 기준으로 최근 3년을 예측했을 때, 실제 집값과의 <b>평균 오차율이 {mape_err:.1f}%</b> 수준으로 검증되었습니다.
        </div>""", unsafe_allow_html=True)

        # 1. 과거 실제 데이터
        price_eok_hist = [v / 10000 for v in price_vals] # 억원
        wage_ann_hist  = wage_annual_vals # 만원

        # 2. 연결용 예측 데이터
        # conn_vp는 이미 (price_df[selected_region] * area) 기준으로 학습된 만원 단위 예측값임
        conn_vp_eok = [v / 10000 for v in conn_vp]
        # conn_vw는 만원 단위 예측값임
        conn_vw_manwon = conn_vw

        # 이중 Y축 차트 생성
        fig_dual = make_subplots(specs=[[{"secondary_y": True}]])

        # trace 1: 실제 아파트 가격 (좌측 Y축)
        fig_dual.add_trace(
            go.Scatter(
                x=price_dates_std_list,
                y=price_eok_hist,
                name=f"실제 매매가 ({area}㎡, 억원)",
                line=dict(color="#e94560", width=3),
                mode="lines",
                hovertemplate="%{x|%Y년 %m월}<br>매매가: %{y:.2f}억원<extra></extra>"
            ),
            secondary_y=False
        )

        # trace 2: 예측 아파트 가격 (좌측 Y축)
        fig_dual.add_trace(
            go.Scatter(
                x=conn_fp_std_str,
                y=conn_vp_eok,
                name=f"예측 매매가 ({area}㎡, 억원)",
                line=dict(color="rgba(233, 69, 96, 0.85)", width=2),
                mode="lines",
                hovertemplate="%{x|%Y년 %m월}<br>예측가: %{y:.2f}억원<extra></extra>"
            ),
            secondary_y=False
        )

        # trace 3: 실제 임금 (우측 Y축)
        fig_dual.add_trace(
            go.Scatter(
                x=wage_dates_std_list,
                y=wage_ann_hist,
                name="실제 연평균 임금 (만원)",
                line=dict(color="#0f3460", width=3),
                mode="lines",
                hovertemplate="%{x|%Y년 %m월}<br>임금: %{y:,.0f}만원<extra></extra>"
            ),
            secondary_y=True
        )

        # trace 4: 예측 임금 (우측 Y축)
        fig_dual.add_trace(
            go.Scatter(
                x=conn_fw_std_str,
                y=conn_vw_manwon,
                name="예측 연평균 임금 (만원)",
                line=dict(color="rgba(15, 52, 96, 0.85)", width=2),
                mode="lines",
                hovertemplate="%{x|%Y년 %m월}<br>예측임금: %{y:,.0f}만원<extra></extra>"
            ),
            secondary_y=True
        )

        # 세로 안내선 추가
        current_date_std = price_dates_std_list[-1]
        target_date_std  = f"{target_year}-01-01"

        # 현재 시점 세로선
        fig_dual.add_shape(
            type="line", x0=current_date_std, x1=current_date_std,
            y0=0, y1=1, xref="x", yref="y domain",
            line=dict(color="rgba(150, 150, 150, 0.7)", dash="dash", width=1.5)
        )
        fig_dual.add_annotation(
            x=current_date_std, y=0.95, xref="x", yref="y domain",
            text="현재 시점", showarrow=False,
            font=dict(color="#888", size=11), xanchor="right", yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.8)"
        )

        # 예측 목표 연도 세로선
        fig_dual.add_shape(
            type="line", x0=target_date_std, x1=target_date_std,
            y0=0, y1=1, xref="x", yref="y domain",
            line=dict(color="rgba(150, 150, 150, 0.7)", dash="dot", width=1.5)
        )
        fig_dual.add_annotation(
            x=target_date_std, y=0.95, xref="x", yref="y domain",
            text=f"<b>{target_year}년 예측 목표</b>", showarrow=False,
            font=dict(color="#e94560", size=11), xanchor="left", yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.8)"
        )

        # 레이아웃 설정
        fig_dual.update_xaxes(
            type="date",
            tickformat="%Y년",
            dtick="M36", # 3년 단위로 틱 표시
            title_text="연도",
            hoverformat="%Y년 %m월"
        )
        fig_dual.update_yaxes(title_text=f"아파트 매매가 (억원, {area}㎡)", secondary_y=False)
        fig_dual.update_yaxes(title_text="연평균 근로소득 (만원)", secondary_y=True)

        fig_dual.update_layout(
            title=dict(
                text=f"<b>{selected_region} 아파트 가격 vs 근로소득 나홀로 집값 예측 동향</b>",
                x=0.5,
                font=dict(size=16, color="#16213e")
            ),
            height=500,
            margin=dict(t=80, b=50, l=60, r=60),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                x=0.5,
                xanchor="center",
                font=dict(size=11)
            ),
            hovermode="x unified"
        )
        apply_dark_theme(fig_dual)

        st.plotly_chart(fig_dual, use_container_width=True)

        col_p, col_w, col_pi = st.columns(3)
        col_p.metric(f"{target_year}년 예상 매매가 ({area}㎡)",
                     f"{pred_price_eok:.1f}억원",
                     delta=f"현재 대비 +{pred_price_eok - price_vals[-1]/10000:.1f}억원")
        col_w.metric(f"{target_year}년 예측 임금",
                     f"{pred_wage_manwon:,.0f}만원",
                     delta=f"현재 대비 +{pred_wage_manwon - current_wage:,.0f}만원")
        col_pi.metric(f"{target_year}년 예측 PIR",
                      f"{pred_pir_future:.1f}배",
                      delta=f"현재 {current_pir:.1f}배 → {pred_pir_future:.1f}배",
                      delta_color="inverse")


    # ── 💰 적정 소득 산출 ───────────────────────────────────────────────────
    elif _active == "income":
        st.markdown(f'<div class="section-header">{target_year}년 미래 적정 소득 산출 방법론</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="formula-box">
        Income_target = Price_{target_year} / PIR_target<br><br>
        <span style="color:#ccc; font-size:0.9rem; font-weight:400;">
        · Price_{target_year}: {area}㎡ 기준 예상 매매가 ({pred_price_eok:.2f}억원)<br>
        · 월별 데이터 기반 CAGR ({price_cagr:.1f}%/년) 적용<br>
        · PIR_target: 사회적 합의 기반 주거부담배율 ({pir_target}배)
        </span>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">주요 지역별 입력 변수 요약</div>', unsafe_allow_html=True)
        table_data = []
        for _region in REGION_LIST:
            if _region not in price_df.columns:
                continue
            _sqm_vals   = price_df[_region].tolist()
            _tot_vals   = [v * area for v in _sqm_vals]
            _n = (len(price_df) - 1) / 12.0
            _cagr = ((_tot_vals[-1] / _tot_vals[0]) ** (1 / _n) - 1) * 100 if _tot_vals[0] > 0 else 0
            _pred_sqm, _, _, _, _ = forecast_poly(
                price_floats, _sqm_vals, target_year,
                add_cycle=use_cycle
            )
            _pred_tot_eok = round(_pred_sqm * area / 10000, 1)
            table_data.append({
                "분석 지역": _region,
                f"현재 매매가 ({area}㎡)": f"{_tot_vals[-1]/10000:.2f}억원",
                "실측 CAGR": f"{_cagr:.1f}%",
                f"{target_year}년 예상가": f"약 {_pred_tot_eok}억원",
            })
        st.dataframe(pd.DataFrame(table_data), width='stretch', hide_index=True)

        col1, col2 = st.columns(2, gap="large")
        r = income_result
        with col1:
            st.markdown(f"""
<table class="income-table">
  <thead><tr><th>항목</th><th style="text-align:right">금액</th></tr></thead>
  <tbody>
    <tr>
      <td>{target_year}년 예상 총 매매가<br><small style="color:#888">{selected_region} · {area}㎡ 기준</small></td>
      <td><b>{r['총매매가_만원']:,}만원</b> <small>({pred_price_eok:.1f}억원)</small></td>
    </tr>
    <tr>
      <td>
        필요 자기자본
        <span class="term">
          LTV {ltv_pct}%
          <span class="tip">📌 <b>LTV (Loan To Value)</b><br>담보인정비율. 주택 가격 대비 대출 가능한 최대 금액의 비율입니다.<br><br>예) LTV 60%면 10억 집에 최대 6억 대출 가능 → 나머지 4억은 자기자본으로 조달해야 합니다.</span>
        </span>
      </td>
      <td><b>{r['필요자기자본_만원']:,}만원</b></td>
    </tr>
    <tr>
      <td>
        <span class="term">
          연간 원리금 상환액
          <span class="tip">📌 <b>원리금 (元利金)</b><br>원금 + 이자를 합친 대출 상환 금액입니다.<br><br>현재 설정: 대출금리 {loan_rate*100:.1f}% · 대출기간 {loan_years}년 기준 매년 갚아야 하는 총액입니다.</span>
        </span>
      </td>
      <td><b>{r['연간원리금_만원']:,}만원</b></td>
    </tr>
    <tr style="background:rgba(233, 69, 96, 0.12)">
      <td>
        <span class="term">
          DSR {dsr_pct}% 기준
          <span class="tip">📌 <b>DSR (Debt Service Ratio)</b><br>총부채원리금상환비율. 연소득 중 모든 대출의 원리금 상환액이 차지하는 비율입니다.<br><br>예) DSR 40%면 연소득의 40%까지만 대출 상환에 쓸 수 있습니다. 정부 규제 기준치입니다.</span>
        </span>
        필요 연소득
      </td>
      <td><b style="color:#e94560;font-size:1.05rem">{r['필요연소득_만원']:,}만원</b></td>
    </tr>
    <tr>
      <td>필요 월소득</td>
      <td><b>{r['필요월소득_만원']:,}만원</b></td>
    </tr>
    <tr style="background:#fff5f7">
      <td>
        <span class="term">
          PIR {pir_target}배
          <span class="tip">📌 <b>PIR (Price to Income Ratio)</b><br>주택가격 대비 소득 비율. 연간 가구소득으로 집 한 채를 사려면 몇 년이 걸리는지를 나타냅니다.<br><br>PIR {pir_target}배 = 한 푼도 안 쓰고 {pir_target}년치 연봉을 모아야 집을 살 수 있다는 의미입니다.<br>UN 기준 적정 PIR은 3~5배입니다.</span>
        </span>
        기준 필요 연소득
      </td>
      <td><b style="color:#e94560;font-size:1.05rem">{income_pir:,}만원</b></td>
    </tr>
    <tr>
      <td>현재 연평균 임금 <small>({latest_wage_date})</small></td>
      <td>{current_wage:,.0f}만원</td>
    </tr>
    <tr>
      <td>소득 부족 배율 (PIR 기준)</td>
      <td><b style="color:#c0392b;font-size:1.1rem">{lack_ratio}배</b></td>
    </tr>
  </tbody>
</table>
""", unsafe_allow_html=True)
            st.info(f"📌 PIR {pir_target}배 유지 시 {selected_region} 기준 필요 연소득 **{income_pir:,}만원** — 현재 대비 약 {lack_ratio}배 소득 성장 필요")

        with col2:
            _labels = [f"현재 임금 ({latest_wage_date})", f"예측 임금 ({target_year})",
                       f"PIR {pir_target}배 필요소득", f"DSR {dsr_pct}% 필요소득"]
            _values = [round(current_wage), round(pred_wage_manwon), income_pir, r["필요연소득_만원"]]
            fig_income = go.Figure(go.Bar(
                x=_labels, y=_values,
                marker_color=["#0f3460","#533483","#e94560","#c0392b"],
                text=[f"{v:,}만원" for v in _values], textposition="outside"
            ))
            fig_income.update_layout(height=400, yaxis_title="연소득 (만원)", showlegend=False)
            apply_dark_theme(fig_income)
            st.plotly_chart(fig_income, use_container_width=True)

        st.markdown('<div class="section-header">나의 자산 진단 (목표 달성 가능성)</div>', unsafe_allow_html=True)
        
        required_equity = r['필요자기자본_만원']
        shortfall = required_equity - my_asset
        progress_val = min(my_asset / required_equity, 1.0) if required_equity > 0 else 1.0
        
        st.progress(progress_val)
        st.caption(f"목표 자기자본 대비 달성률: **{progress_val*100:.1f}%** ({my_asset:,}만원 / {required_equity:,}만원)")
        
        if shortfall > 0:
            if my_savings > 0:
                months_needed = shortfall / my_savings
                years_needed = months_needed / 12
                yrs_to_target = target_year - int(price_date_list[-1].split(".")[0])
                
                if years_needed <= yrs_to_target:
                    diag_msg = f"✅ 목표 달성을 위해 **{shortfall:,}만원**이 추가로 필요합니다. 현재 저축액 유지 시 약 **{years_needed:.1f}년** 뒤 도달 가능하여 {target_year}년 내 매수가 가능할 것으로 예상됩니다!"
                    st.success(diag_msg)
                else:
                    diag_msg = f"⚠️ 목표 달성을 위해 **{shortfall:,}만원**이 추가로 필요합니다. 약 **{years_needed:.1f}년**이 소요되어 {target_year}년보다 늦어집니다. 월 저축액을 늘리거나 목표 연도를 늦춰보세요."
                    st.warning(diag_msg)
            else:
                diag_msg = f"⚠️ 목표 달성을 위해 **{shortfall:,}만원**이 추가로 필요합니다. 월 저축액을 입력해주세요."
                st.warning(diag_msg)
        else:
            diag_msg = f"🎉 이미 필요 자기자본({required_equity:,}만원)을 달성했습니다!"
            st.success(diag_msg)

        st.markdown('<div class="section-header">세대별 주택 구입 부담 지수 변화</div>', unsafe_allow_html=True)
        gen_labels = ["베이비부머 (1960년대)", "X세대 (1980년대)", "밀레니얼 (2000년대)",
                      f"Z세대 (2020년대)\n{latest_wage_date}", f"알파세대 ({target_year}년대 예측)"]
        gen_pir    = [3.2, 5.8, 9.4, round(current_pir, 1), round(pred_pir_future, 1)]
        fig_gen = go.Figure(go.Bar(
            x=gen_labels, y=gen_pir,
            marker_color=["#16213e","#0f3460","#533483","#e94560","#c0392b"],
            text=[f"PIR {v}" for v in gen_pir], textposition="outside"
        ))
        fig_gen.add_hline(y=5,  line_dash="dash", line_color="orange")
        fig_gen.add_hline(y=10, line_dash="dash", line_color="red")
        fig_gen.update_layout(height=420, yaxis_title="PIR (배)", showlegend=False)
        apply_dark_theme(fig_gen)
        st.plotly_chart(fig_gen, use_container_width=True)


    # ── 📊 시나리오 분석 ────────────────────────────────────────────────────
    elif _active == "scenario":
        st.markdown('<div class="section-header">거시 경제 요인 반영 시나리오 분석</div>', unsafe_allow_html=True)
        st.markdown("<div style='color:#666; margin-bottom:1.5rem; padding-left:0.5rem;'>집값에 영향을 주는 거시 요인(금리, 통화량)을 조절하여 미래 자산 가격을 시뮬레이션해 보세요.</div>", unsafe_allow_html=True)
        
        # 거시 요인 슬라이더 (사용자 입력)
        macro_col1, macro_col2 = st.columns(2)
        with macro_col1:
            sim_rate = st.slider("📉 향후 평균 기준금리 예상 (%)", 1.0, 7.0, 3.5, step=0.25, 
                                 help="금리가 높아질수록 자산 가격 상승률은 둔화됩니다.")
        with macro_col2:
            sim_m2 = st.slider("📈 시중 통화량(M2) 증가율 예상 (%)", 0.0, 10.0, 5.0, step=0.5,
                               help="시중에 풀린 돈(통화량)이 많아질수록 자산 가격 상승률은 가팔라집니다.")
            
        # 주변 요인(금리, 통화량)을 반영한 동적 상승률(r) 계산식 (가상의 민감도 식)
        base_r = 0.04 # 기본 성장률 4% (과거 추세)
        adj_r = base_r - ((sim_rate - 3.5) * 0.01) + ((sim_m2 - 5.0) * 0.005)

        scenarios = {
            "거시 요인 반영 (동적)": {
                "r": adj_r, 
                "color": "#8e44ad", 
                "desc": f"사용자가 설정한 금리 {sim_rate}%, 통화량 {sim_m2}% 증가율을 복합 반영한 동적 시나리오입니다."
            },
            "과거 추세 유지 (기본)": {
                "r": base_r, 
                "color": "#e94560", 
                "desc": "과거 10년간의 나홀로 집값 추세(기본 4%)가 그대로 유지될 경우의 기준선입니다."
            },
            "자산 가격 침체 (비관)": {
                "r": 0.02, 
                "color": "#0f3460", 
                "desc": "강력한 정책적 개입 또는 거시 경제 침체로 상승률이 연 2%로 둔화될 경우입니다."
            }
        }
        yrs_to_target = target_year - int(price_date_list[-1].split(".")[0])
        current_total_manwon = price_df[selected_region].iloc[-1] * area  # 만원

        s_cols = st.columns(3, gap="large")
        for _idx, (_name, _info) in enumerate(scenarios.items()):
            _sp_manwon = scenario_price(current_total_manwon, _info["r"], yrs_to_target)  # 만원
            _sp_eok    = _sp_manwon / 10000
            _si_pir    = calc_income_by_pir(_sp_manwon, pir_target)
            _si_dsr    = calc_required_income(_sp_manwon, dsr, ltv, loan_rate, loan_years)["필요연소득_만원"]
            with s_cols[_idx]:
                st.markdown(f"""<div class="scenario-card" style="background:{_info['color']};">
                    <div style="font-size:1.2rem;font-weight:700;">{_name}</div>
                    <div style="font-size:1.8rem;font-weight:800;margin:0.8rem 0;">{_sp_eok:.1f}억원</div>
                    <div style="font-size:0.9rem;opacity:0.9;">{target_year}년 예상 매매가 ({area}㎡)</div>
                    <div style="font-size:1.0rem;font-weight:600;margin-top:0.6rem;">PIR 기준: {_si_pir:,}만원</div>
                    <div style="font-size:0.95rem;margin-top:0.3rem;">DSR 기준: {_si_dsr:,}만원</div>
                    <div style="font-size:0.85rem;opacity:0.85;margin-top:0.6rem;line-height:1.4;">{_info['desc']}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">시나리오별 연도별 가격 예측</div>', unsafe_allow_html=True)
        future_years_sc = list(range(int(price_date_list[-1].split(".")[0]), target_year + 1))
        fig_sc = go.Figure()

        # 과거 실제 데이터 (만원/㎡ → 억원으로 스케일 맞춤, x축을 float 연도로 통일)
        price_eok_hist = [v / 10000 for v in price_vals]
        price_smooth   = smooth_series(price_eok_hist, window=6)
        fig_sc.add_trace(go.Scatter(x=price_floats, y=price_smooth,
                                     name="실제 (과거, 평활)", mode="lines",
                                     line=dict(color="#333", width=2, dash="dot")))
        for _name, _info in scenarios.items():
            _prices_sc = [current_total_manwon / 10000 * ((1 + _info["r"]) ** _i)
                          for _i in range(len(future_years_sc))]
            fig_sc.add_trace(go.Scatter(x=future_years_sc, y=_prices_sc, name=_name,
                                         mode="lines", line=dict(color=_info["color"], width=3)))
        # 현재 시점 세로선 (add_shape 사용)
        _current_x = price_floats[-1]
        fig_sc.add_shape(
            type="line", x0=_current_x, x1=_current_x,
            y0=0, y1=1, xref="x", yref="y domain",
            line=dict(color="rgba(150,150,150,0.7)", dash="dash", width=1.5)
        )
        fig_sc.add_annotation(
            x=_current_x, y=0.97, xref="x", yref="y domain",
            text="현재 시점", showarrow=False,
            font=dict(color="#888", size=11), xanchor="right", yanchor="top",
            bgcolor="rgba(255,255,255,0.8)"
        )
        fig_sc.update_layout(height=450,
                              xaxis_title="연도", yaxis_title=f"매매가 ({area}㎡, 억원)",
                              hovermode="x unified", 
                              legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0.5, xanchor="center"))
        apply_dark_theme(fig_sc)
        st.plotly_chart(fig_sc, use_container_width=True)

        st.markdown('<div class="section-header">시나리오별 상세 수치 비교</div>', unsafe_allow_html=True)
        table_rows = []
        for _name, _info in scenarios.items():
            _sp_manwon  = scenario_price(current_total_manwon, _info["r"], yrs_to_target)
            _sp_eok     = _sp_manwon / 10000
            _si_pir     = calc_income_by_pir(_sp_manwon, pir_target)
            _si_dsr     = calc_required_income(_sp_manwon, dsr, ltv, loan_rate, loan_years)["필요연소득_만원"]
            _si_equity  = calc_required_income(_sp_manwon, dsr, ltv, loan_rate, loan_years)["필요자기자본_만원"]
            table_rows.append({
                "시나리오":                        _name,
                f"{target_year}년 예상가":          f"{_sp_eok:.1f}억원",
                f"PIR {pir_target}배 필요소득":     f"{_si_pir:,}만원",
                f"DSR {dsr_pct}% 필요소득":         f"{_si_dsr:,}만원",
                f"필요 자기자본 (LTV {ltv_pct}%)":  f"{_si_equity:,}만원",
                "현재 대비 배율":                   f"{_si_pir / current_wage:.1f}배",
            })
        st.dataframe(pd.DataFrame(table_rows), width='stretch', hide_index=True)

        st.markdown("""<div class="info-box">
        💡 <b>분석 결론</b><br><br>
        · 개인: 생애 주기별 자산 형성 계획의 현실적 가이드라인 제공<br>
        · 정책: 주택 공급 및 소득 지원 정책의 목표 수치 산정 근거<br>
        · 기업: 임직원 주거 복지 및 보상 체계 설계를 위한 데이터<br><br>
        <i>"본 분석 모델은 월별 실거래 데이터에 기반한 현실적인 미래상을 제시함으로써, 세대 간 자산 격차 해소를 위한 논의의 시작점을 제공합니다."</i>
        </div>""", unsafe_allow_html=True)


# ── 푸터
st.markdown("---")
st.markdown("""<div style="text-align:center;color:#888;font-size:0.85rem;margin-top:2rem;">
데이터 출처: KB부동산 데이터Hub · 고용노동부 월간 노동통계<br>
⚠️ 본 분석은 실거래 데이터 기반의 다항회귀 예측 모형이며, 투자 제안 목적이 아닌 학술 연구용 시뮬레이터입니다.<br>
Data Science Project: Decoupling Analysis 2026
</div>""", unsafe_allow_html=True)
