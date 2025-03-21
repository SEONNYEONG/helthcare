import streamlit as st

st.set_page_config(layout="wide")

st.markdown("""
    <style>
    .main .block-container {
        max-width: 100% !important;
        width: 100% !important;
        padding-left: 4rem !important;
        padding-right: 4rem !important;
    }
    </style>

    <script>
    const interval = setInterval(() => {
        const container = window.parent.document.querySelector('.block-container');
        if (container) {
            container.style.maxWidth = '100%';
            container.style.width = '100%';
            container.style.paddingLeft = '4rem';
            container.style.paddingRight = '4rem';
            clearInterval(interval);
        }
    }, 100);
    </script>
""", unsafe_allow_html=True)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ======= 데이터 불러오기 ======= #
df = pd.read_csv('data/heart_disease_uci.csv')

# ======= 불필요한 열 제거 ======= #
df.drop(['id', 'dataset'], axis=1, inplace=True, errors='ignore')

# ======= 결측치 처리 ======= #
# fbs는 이진값이므로 결측치 및 이상치(0, 1 이외) 제거
if 'fbs' in df.columns:
    df = df[df['fbs'].isin([0, 1])]

# 소수의 결측치는 대체, 다수 결측치는 행 제거
df.dropna(subset=['slope', 'thal', 'ca'], inplace=True)

# ======= Label Encoding ======= #
label_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
label_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# ======= 타겟 이진화 (0: 정상, 1: 심장질환 있음) ======= #
df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
df.drop('num', axis=1, inplace=True)

# ======= 이상치 제거 (IQR 기반) ======= #
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('target')
# 바이올린 플롯 부적합한 이진 변수 fbs 제거
if 'fbs' in numeric_cols:
    numeric_cols = numeric_cols.drop('fbs')
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]

# ======= 특성과 타겟 분리 ======= #
X = df.drop('target', axis=1)
y = df['target']

# ======= 스케일링 ======= #
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======= 데이터 분할 ======= #
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ======= 모델 학습 ======= #
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ======= 특성 중요도 ======= #
feature_importances = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
feature_importances = feature_importances.sort_values(by="Importance", ascending=False)

# ======= 성능 평가 ======= #
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['No Disease', 'Disease'])

# ======= 탭 메뉴 UI (버튼 방식) ======= #
with st.sidebar:
    st.markdown("""
        <style>
        .sidebar-title {
            font-size: 22px;
            font-weight: 600;
            color: #4B8BBE;
            margin-bottom: 1rem;
        }
        .custom-button {
            display: block;
            width: 100%;
            padding: 0.5rem 1rem;
            margin-bottom: 0.5rem;
            background-color: #f0f2f6;
            border: 1px solid #ddd;
            border-radius: 8px;
            text-align: center;
            color: black;
            font-weight: 500;
            cursor: pointer;
            transition: background-color 0.2s ease;
        }
        .custom-button:hover {
            background-color: #e0e8f0;
        }
        .custom-button-selected {
            background-color: #4B8BBE !important;
            color: white !important;
            font-weight: bold;
        }
        </style>
        
    """, unsafe_allow_html=True)

    if 'page' not in st.session_state:
        st.session_state.page = 'Home'

    def set_page(p):
        st.session_state.page = p

    home_btn = st.button("🏠 Home", on_click=set_page, args=("Home",), key="home_btn")
    analyze_btn = st.button("🧪 데이터 분석", on_click=set_page, args=("데이터 분석",), key="analyze_btn")
    eda_btn = st.button("📊 데이터 시각화", on_click=set_page, args=("EDA",), key="eda_btn")
    perf_btn = st.button("📈 머신러닝 보고서", on_click=set_page, args=("Model Performance",), key="perf_btn")

menu = st.session_state.page

# 메뉴 목록에 데이터 분석 추가
if menu not in ["Home", "데이터 분석", "EDA", "머신러닝 보고서"]:
    st.session_state.page = "Home"

def home():
    st.markdown("""
        <style>
        .home-title {
            font-size: 32px;
            font-weight: 700;
            color: #333;
        }
        .home-section {
            font-size: 18px;
            padding-top: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='home-title'>💓 Heart Disease Classification Dashboard</div>", unsafe_allow_html=True)

    st.markdown("""
        <div class='home-section'>
            📌 본 대시보드는 심장병 여부를 예측하기 위한 머신러닝 분류 모델 결과를 시각화한 것입니다.
        </div>
        <div class='home-section'>
            🎯 <b>타겟 정의:</b> <br> - 0 = 정상 <br> - 1 = 심장질환 있음
        </div>
        <div class='home-section'>
            🔍 <b>총 사용된 특성 수:</b> {}개
        </div>
    """.format(X.shape[1]), unsafe_allow_html=True)



def analyze():
    st.title("데이터 분석")
    analyze_tabs = st.tabs(["상위 데이터", "데이터 통계", "컬럼별 데이터", "조건별 데이터"])

    with analyze_tabs[0]:
        st.subheader("📌 상위 데이터 미리보기")
        st.dataframe(df.head())

    with analyze_tabs[1]:
        st.subheader("📊 기본 통계 요약")
        st.dataframe(df.describe().T)

    with analyze_tabs[2]:
        st.subheader("📋 컬럼별 고유값 및 데이터 타입")
        all_columns = df.columns.tolist()
        selected_col = st.selectbox("컬럼 선택", all_columns, key="column_select_unique")

        col_data = df[selected_col]
        col_summary = pd.DataFrame({
            '항목': ['데이터 타입', '결측치 수', '고유값 수', '예시 값'],
            '정보': [
                col_data.dtype,
                col_data.isnull().sum(),
                col_data.nunique(),
                ', '.join(map(str, col_data.unique()[:5])) + (' ...' if col_data.nunique() > 5 else '')
            ]
        })
        st.table(col_summary.set_index('항목'))

        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('## 🔎 여러 컬럼 선택해서 보기')
        multi_cols = st.multiselect('복수의 컬럼을 선택하세요', df.columns, key='multi_col_select')
        if multi_cols:
            st.dataframe(df[multi_cols])

    with analyze_tabs[3]:
        st.subheader("🔍 조건에 따른 데이터 필터링")
        selected_col = st.selectbox("컬럼 선택", df.columns, key="filter_column_select")
        unique_vals = df[selected_col].unique()
        selected_val = st.selectbox("값 선택", unique_vals, key="filter_value_select")
        filtered_df = df[df[selected_col] == selected_val]
        st.write(f"선택한 조건: {selected_col} = {selected_val} → {len(filtered_df)}행")
        st.dataframe(filtered_df)


    st.markdown("""
        <style>
        .home-title {
            font-size: 32px;
            font-weight: 700;
            color: #333;
        }
        .home-section {
            font-size: 18px;
            padding-top: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

def eda():
    st.title('데이터 시각화')
    chart_tabs = st.tabs(['수치형 변수 분포', '타겟별 변수 비교', '범주형 변수 분포', '상관관계 분석'])

    with chart_tabs[0]:
        st.subheader("▶ 수치형 변수 분포 (히스토그램 & KDE)")
        colors = sns.color_palette('husl', len(numeric_cols))
        num_cols = 2
        num_rows = int(np.ceil(len(numeric_cols) / num_cols))
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, num_rows * 5), constrained_layout=True)
        axes = axes.flatten()
        for i, col in enumerate(numeric_cols):
            sns.histplot(df[col], kde=True, bins=20, ax=axes[i], color=colors[i])
            axes[i].set_title(f"{col} Distribution")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Count')
        # 빈 플롯 숨기기
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        st.pyplot(fig)

    with chart_tabs[1]:
        st.subheader("▶ 타겟(target) 별 변수 분포 (바이올린 플롯)")
        num_cols = 2
        num_rows = int(np.ceil(len(numeric_cols) / num_cols))
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, num_rows * 5), constrained_layout=True)
        axes = axes.flatten()
        for i, col in enumerate(numeric_cols):
            sns.violinplot(x='target', y=col, data=df, ax=axes[i], palette='pastel')
            axes[i].set_title(f"{col} by Heart Disease Status")
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        st.pyplot(fig)

    with chart_tabs[2]:
        cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
        st.subheader("▶ 범주형 변수 분포 (막대그래프)")
        num_cols = 2
        num_rows = int(np.ceil(len(cat_cols) / num_cols))
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, num_rows * 5), constrained_layout=True)
        axes = axes.flatten()
        for i, col in enumerate(cat_cols):
            sns.countplot(x=col, data=df, ax=axes[i], palette='Set3')
            axes[i].set_title(f"{col} Count")
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        st.pyplot(fig)

    with chart_tabs[3]:
        st.subheader("▶ 변수 간 상관관계 (히트맵)")
        fig, ax = plt.subplots(figsize=(16, 10))
        corr = df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

def model_performance():
    st.title("모델 성능 평가")
    st.write(f"### Accuracy: {accuracy:.2f}")

    # classification_report를 dict로 변환 후 DataFrame 생성
    report_dict = classification_report(
        y_test, y_pred, target_names=['No Disease', 'Disease'], output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose().reset_index()
    report_df.rename(columns={'index': 'Label'}, inplace=True)

    # 표 출력
    st.write("### Classification Report")
    st.dataframe(report_df.style.format({
        "precision": "{:.2f}",
        "recall": "{:.2f}",
        "f1-score": "{:.2f}",
        "support": "{:.0f}"
    }))

    # 중요도 시각화
    st.write("### Feature Importances")
    st.bar_chart(feature_importances.set_index('Feature'))

if menu == 'Home':
    home()
elif menu == '데이터 분석':
    analyze()
elif menu == 'EDA':
    eda()
elif menu == '머신러닝 보고서':
    model_performance()
