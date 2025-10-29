import streamlit as st
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher
import re
from collections import Counter
import sys
import importlib
import inspect

# 1. 캐시 초기화는 유지
#st.cache_data.clear()

# =========================================================================
# 🚀 Streamlit 앱 메인 실행 로직 (초기 설정)
# =========================================================================
st.set_page_config( 
    page_title="작업계획서 작성지원 챗봇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================================
# 📌 고정 상수 정의 및 경로 설정 
# =========================================================================
BASE_DIR = Path(__file__).parent
CSV_DIR = BASE_DIR / "input_csv" 

# 위험성 평가 CSV 파일 목록
CSV_FILENAMES = [
    "위험성평가(서부본부).csv", "위험성평가(경북본부).csv", "위험성평가(서울본부).csv",
    "위험성평가(광주본부).csv", "위험성평가(전남본부).csv", "위험성평가(대전충남본부).csv",
    "위험성평가(오송고속).csv",
]
CSV_FILES = [CSV_DIR / filename for filename in CSV_FILENAMES]

# -------------------------------------------------------------
# 💡 유지보수 세칙 CSV 파일 목록 (탭 2번의 검색 대상 데이터)
# -------------------------------------------------------------
MAINTENANCE_CSV_FILENAMES = [
    "정보통신설비 유지보수 세칙.csv",
    "신호제어설비 유지보수 세칙.csv",
    "전철전력설비 유지보수 세칙(송변전설비).csv",
    "전철전력설비 유지보수 세칙(전력설비).csv",
    "전철전력설비 유지보수 세칙(전차선로설비, 강체구간).csv",
    "전철전력설비 유지보수 세칙(전차선로설비, 지상구간).csv",
]
MAINTENANCE_CSV_FILES = [CSV_DIR / filename for filename in MAINTENANCE_CSV_FILENAMES]
# 표준 스키마: 이 스키마로 데이터를 로드하고 통일할 것입니다.
MAINTENANCE_NEEDED_COLUMNS = ["세칙명", "설비명", "점검항목", "점검주기", "점검종류"] 

# 💡 유지보수 세칙 파일별 컬럼 맵핑 정의 (원본 컬럼명 : 표준 컬럼명)
MAINTENANCE_COLUMN_MAPS: Dict[str, Dict[str, str]] = {
    # '정보통신설비 유지보수 세칙.csv'
    "정보통신설비 유지보수 세칙.csv": {
        "설비명": "설비명",
        "점검 항목": "점검항목", 
        "점검주기": "점검주기"
    },
    "신호제어설비 유지보수 세칙.csv": {
        "시설명": "설비명", 
        "점검항목": "점검항목", 
        "점검주기": "점검주기" 
    },
    "전철전력설비 유지보수 세칙(송변전설비).csv": { 
        "시설명": "설비명", 
        "점검항목": "점검항목", 
        "점검주기": "점검주기", 
        "점검종류": "점검종류" 
    },
    "전철전력설비 유지보수 세칙(전력설비).csv": {
        "시설명": "설비명", 
        "점검항목": "점검항목", 
        "점검주기": "점검주기", 
        "점검종류": "점검종류" 
    },
    "전철전력설비 유지보수 세칙(전차선로설비, 강체구간).csv": {
        "시설명": "설비명", 
        "점검항목": "점검항목", 
        "점검주기": "점검주기", 
        "점검종류": "점검종류" 
    },
    "전철전력설비 유지보수 세칙(전차선로설비, 지상구간).csv": {
        "시설명": "설비명", 
        "점검항목": "점검항목", 
        "점검주기": "점검주기"
    }
}

# -------------------------------------------------------------
# 💡 시설물 점검 CSV 파일 목록 (사용하지 않음)
# -------------------------------------------------------------
FACILITY_CSV_FILENAMES = [
    "시설물_점검항목_승강기.csv", 
    "시설물_점검항목_전차선.csv", 
    "시설물_점검항목_신호기.csv",
    "시설물_점검항목_스크린도어.csv",
    "시설물_점검항목_냉난방장치.csv",
    "시설물_점검항목_급전선.csv",
]
FACILITY_CSV_FILES = [CSV_DIR / filename for filename in FACILITY_CSV_FILENAMES]
FACILITY_NEEDED_COLUMNS = ["시설물명", "점검항목", "점검주기"] 

# 시설물 점검 CSV 파일별 컬럼 맵핑 정의
FACILITY_COLUMN_MAPS: Dict[str, Dict[str, str]] = {
    "시설물_점검항목_급전선.csv": {
        "시설물 명": "시설물명", 
        "점검 내용": "점검항목", 
        "주기": "점검주기", 
    },
}

# =========================================================================
# 📚 유틸리티 함수 (CSV/세칙 파싱 및 검색) 
# =========================================================================

def _get_rule_name_from_filename(filename: str) -> str:
    name = filename.replace('.csv', '').replace('.pdf', '')
    match = re.search(r'(.+설비)\s*유지보수\s*세칙', name)
    if match: return match.group(1).strip() + " 유지보수 세칙"
    return name.strip()

def _clean_value(value: str) -> str:
    if value is None: return ""
    value = re.sub(r'[\s,]+', ' ', value).strip()
    value = re.sub(r'^[\d\s\)\.\-]+\s*|\s*[\d\s\)\.]+$', '', value).strip()
    return value
    
@st.cache_data
def load_all_safety_data(csv_file_paths, required_cols: List[str]):
    """
    여러 CSV 파일을 로드하여 하나의 DataFrame으로 결합하고, 필수 컬럼을 체크합니다.
    (로딩 실패 반복 루프 제거됨)
    """
    all_dfs = []
    load_logs = []

    # [새로운 정의] 점검항목과 점검 세부항목을 결합해야 하는 파일 목록
    FILES_TO_MERGE = ["신호제어설비 유지보수 세칙.csv", "전철전력설비 유지보수 세칙(송변전설비).csv"]

    for p in csv_file_paths:
        if not p.exists():
            load_logs.append((str(p), "❌ 파일 없음"))
            continue

        df = None
        last_err = None
        
        # --- 🚨 수정된 핵심 로직: 단 하나의 로딩 시도 ---
        try:
            # 1. UTF-8-SIG와 쉼표(,)로 우선 시도
            df = pd.read_csv(p, encoding="utf-8-sig", sep=",", low_memory=False)
            
        except Exception:
            # 2. 실패 시, CP949로 재시도 (한글 윈도우 인코딩)
            try:
                df = pd.read_csv(p, encoding="cp949", sep=",", low_memory=False)
            except Exception as e:
                # 3. 최종 실패 기록
                last_err = e
                df = None
        # --- 🚨 수정된 핵심 로직 끝 ---
        
        # ... (파일 로드 실패 처리 생략)
        if df is None:
            # 원인 분석을 위해 에러 로그를 명확하게 남김
            if not load_logs or "❌ 파일 없음" not in load_logs[-1]: 
                 load_logs.append((str(p), f"❌ 최종 읽기 실패: {type(last_err).__name__} (파일 인코딩/형식 오류)"))
            continue

        # 컬럼 정리: 공백/개행/BOM 제거
        df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]

        # 동의어 맵핑 (위험성 평가 CSV에 주로 사용되는 동의어만 임시 적용)
        col_map = {
            "작업 명": "작업명", "작업": "작업명", "위험 요인": "위험요인", "안전 조치 방법": "안전조치방법",
        }
        for src, dst in col_map.items():
            if src in df.columns and dst not in df.columns:
                df[dst] = df[src]
        
        # 💡 유지보수 세칙/시설물 점검 CSV 파일별 맞춤 컬럼 맵핑 적용 (rename)
        filename = p.name
        map_config = {}
        current_required_cols = list(required_cols)

        if current_required_cols == MAINTENANCE_NEEDED_COLUMNS and filename in MAINTENANCE_COLUMN_MAPS:
            map_config = MAINTENANCE_COLUMN_MAPS[filename]
        elif current_required_cols == FACILITY_NEEDED_COLUMNS and filename in FACILITY_COLUMN_MAPS:
            map_config = FACILITY_COLUMN_MAPS[filename]

        new_cols = {}
        for original, target in map_config.items():
            if original in df.columns and original != target:
                new_cols[original] = target
            
        if new_cols:
            df.rename(columns=new_cols, inplace=True)
        
        # 🚨 특수 처리: '점검항목'과 '점검 세부항목' 결합 로직
        if filename in FILES_TO_MERGE:
            COL_ITEM = "점검항목"
            COL_DETAIL = "점검 세부항목" # 🚨 컬럼명 통일
            
            if COL_ITEM in df.columns and COL_DETAIL in df.columns:
                
                df[COL_ITEM] = df[COL_ITEM].astype(str).str.strip()
                df[COL_DETAIL] = df[COL_DETAIL].astype(str).str.strip()
                
                mask = df[COL_DETAIL].str.len() > 0
                
                df.loc[mask, COL_ITEM] = (
                    df.loc[mask, COL_ITEM] + " (" + df.loc[mask, COL_DETAIL] + ")"
                )
                
                df.drop(columns=[COL_DETAIL], inplace=True)
                
        # 파일명에서 세칙명을 추출하여 컬럼 추가 (세칙 CSV 로드 시)
        if "세칙명" in current_required_cols and "세칙명" not in df.columns:
             df["세칙명"] = _get_rule_name_from_filename(p.name)

        # 필수 열 체크
        final_check_cols = [
            c for c in required_cols 
            if c in df.columns or (c == "점검 세부항목" and filename not in FILES_TO_MERGE)
        ]
        
        missing = [c for c in final_check_cols if c not in df.columns]
        
        if missing:
            load_logs.append((str(p), f"❗ 필수 열 누락: {missing} → 스킵 (현재 컬럼: {list(df.columns)})"))
            continue
        
        # 필수 열에 대해 클리닝 적용
        for c in current_required_cols:
            if c not in df.columns: continue 

            df[c] = (
                df[c]
                .astype(str)
                .str.replace(r"[\r\n\t]+", " ", regex=True)
                .str.strip()
            )

        all_dfs.append(df) 
        load_logs.append((str(p), f"✅ 로드 완료: {len(df)} 행"))

    # 로드 요약
    print("=== CSV LOAD LOGS ===")
    for rec in load_logs:
        print(" -", rec[0], ":", rec[1])

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True).dropna(how="all")
    
    # 최종적으로 필요한 열만 반환합니다.
    final_cols = list(set(required_cols) & set(combined.columns))
    return combined[final_cols]

def search_csv_safety_data(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """위험성 평가 CSV에서 작업명을 기반으로 검색합니다."""
    # ... (이전 코드와 동일, 생략)
    if df.empty or not query or "작업명" not in df.columns: 
        return pd.DataFrame()
        
    query_lower = query.lower().strip()
    filtered_df = df[df['작업명'].str.contains(query_lower, case=False, na=False)].copy()
    
    if not filtered_df.empty: results = filtered_df
    else:
        # 🚨 'float' object has no attribute 'lower' 오류 방지 로직 추가
        unique_tasks = df['작업명'].astype(str).unique() 
        match_ratios = {}
        for task in unique_tasks:
            if task.lower() == 'nan': continue # NaN 값은 건너뜀
            
            ratio = SequenceMatcher(None, query_lower, task.lower()).ratio()
            if ratio > 0.55: match_ratios[task] = ratio
        
        if match_ratios:
            best_matches = sorted(match_ratios.items(), key=lambda item: item[1], reverse=True)[:5]
            best_tasks = [item[0] for item in best_matches]
            results = df[df['작업명'].isin(best_tasks)].copy()
        else: return pd.DataFrame()
            
    if not results.empty:
        results = results[['작업명', '위험요인', '안전조치방법']].drop_duplicates().reset_index(drop=True)
    return results

def search_maintenance_data(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """유지보수 세칙 데이터를 기반으로 설비명/시설물명을 검색합니다."""
    # ... (이전 코드와 동일, 생략)
    REQUIRED_COLS = ["세칙명", "설비명", "점검항목", "점검주기", "점검종류"] 
    
    if df.empty or not query or REQUIRED_COLS[1] not in df.columns:
        return pd.DataFrame()
    
    query_lower = query.lower().strip()
    
    filtered_df = df[df[REQUIRED_COLS[1]].str.contains(query_lower, case=False, na=False)].copy()
    
    if not filtered_df.empty:
        results = filtered_df[REQUIRED_COLS].drop_duplicates().reset_index(drop=True)
        return results
    else:
        # 🚨 'float' object has no attribute 'lower' 오류 방지 로직 추가
        unique_facilities = df[REQUIRED_COLS[1]].astype(str).unique()
        match_ratios = {}
        for facility in unique_facilities:
            if facility.lower() == 'nan': continue # NaN 값은 건너뜀
            
            ratio = SequenceMatcher(None, query_lower, facility.lower()).ratio()
            if ratio > 0.55: match_ratios[facility] = ratio
        
        if match_ratios:
            best_matches = sorted(match_ratios.items(), key=lambda item: item[1], reverse=True)[:5]
            best_facilities = [item[0] for item in best_matches]
            results = df[df[REQUIRED_COLS[1]].isin(best_facilities)].copy()
            results = results[REQUIRED_COLS].drop_duplicates().reset_index(drop=True)
            return results
        else:
            return pd.DataFrame()

# 🎯 문장 요약 결과를 생성하는 함수 (변경 없음)
def generate_maintenance_summary_sentences(df: pd.DataFrame) -> List[str]:
    """
    유지보수 세칙 결과를 기반으로 자연어 요약 문장을 생성합니다.
    (설비명), (점검주기), (점검항목: 최대 2개 + '등'), (점검종류)를 사용합니다.
    """
    if df.empty:
        return []

    df['점검종류'] = df['점검종류'].fillna("").astype(str).str.strip()

    grouped = df.groupby(['설비명', '점검주기', '점검종류']).agg(
        점검항목=('점검항목', lambda x: list(x.unique()))
    ).reset_index()

    sentences = []
    
    for _, row in grouped.iterrows():
        facility = row['설비명']
        cycle = row['점검주기']
        check_type = row['점검종류']
        items = row['점검항목']
        
        # 1. 점검항목 요약 로직: 3개 이상이면 2개 + '등' 처리
        if len(items) >= 3:
            item_summary = f"'{items[0]}', '{items[1]}' 등"
        elif len(items) == 2:
            item_summary = f"'{items[0]}' 및 '{items[1]}'"
        elif len(items) == 1:
            item_summary = f"'{items[0]}'"
        else:
            continue
            
        # 2. 문장 구성
        if check_type:
            sentence = (
                f"**{facility}**는 점검주기(**{cycle}**)에 따라 {item_summary}의 점검항목을 **{check_type}** 점검해야 합니다."
            )
        else:
             sentence = (
                f"**{facility}**는 점검주기(**{cycle}**)에 따라 {item_summary}의 점검항목을 점검해야 합니다."
            )
        
        sentences.append(sentence)
        
    return sentences

# =========================================================================
# 🚀 Streamlit 앱 메인 실행 로직 (데이터 로드 및 UI)
# =========================================================================

# 1. 데이터 로드 (캐싱 적용)
data_df = load_all_safety_data(csv_file_paths=CSV_FILES, required_cols=["작업명", "위험요인", "안전조치방법"]) 

maintenance_df = load_all_safety_data(
    csv_file_paths=MAINTENANCE_CSV_FILES, 
    required_cols=MAINTENANCE_NEEDED_COLUMNS
)

# 2. 안전장치
if not isinstance(data_df, pd.DataFrame): data_df = pd.DataFrame()
if not isinstance(maintenance_df, pd.DataFrame): maintenance_df = pd.DataFrame()


# --- 사이드바 및 로딩 상태 확인 ---
st.title("🛡️ 작업계획서 작성지원 챗봇")
st.caption("위험성 평가 데이터, 유지보수 세칙 등을 기반으로 정보를 검색합니다.")

# 데이터 로딩 상태 표시
with st.sidebar:
    st.header("📊 데이터 로딩 상태") 
    
    if not data_df.empty:
        st.success(f"💾 **1. 위험성평가** 항목 수: {len(data_df):,}개")
    else:
        st.error("❌ **1. 위험성평가** 로드 실패")

    if not maintenance_df.empty:
        st.success(f"💾 **2. 유지보수 세칙** 항목 수: {len(maintenance_df):,}개")
    else:
        st.error("❌ **2. 유지보수 세칙** 로드 실패") 
    
    st.markdown("---")
    

# --- 메인 탭 정의 ---
tab1, tab2 = st.tabs([
    "🔍 작업명 검색 (위험성 평가)", 
    "⚙️ 시설물 검색 (점검항목 및 주기)", 
]) 

# =========================================================
# tab1: 작업명 검색 (위험성 평가 CSV 검색) - 변경 없음
# =========================================================
with tab1:
    
    st.subheader("1. 작업명 기반 위험성 평가 검색 (CSV)")
    
    csv_query = st.text_input(
        "검색할 **작업명**을 입력하세요 (예: 승강기 유지보수, 전차선 작업):", 
        key="csv_input"
    )
    
    if st.button("위험성평가 검색", key="csv_button") and csv_query:
        if data_df.empty:
              st.warning("⚠️ 위험성 평가 CSV 데이터가 로드되지 않아 검색할 수 없습니다. (사이드바 확인)")
        else:
            with st.spinner(f"'{csv_query}'에 대한 위험성 평가 검색 중..."):
                csv_results = search_csv_safety_data(data_df, csv_query)
            
            if csv_results.empty:
                st.warning(f"🚨 **'{csv_query}'** (또는 유사한 작업명)에 대한 위험성 평가 결과가 없습니다.")
            else:
                st.success(f"✅ **'{csv_query}'**와 가장 유사한 {len(csv_results):,}개의 위험성 평가 결과입니다.")
                
                st.dataframe(csv_results, use_container_width=True)
                
                st.markdown("### 📝 주요 위험요인 및 안전 조치 요약")
                top_risks = Counter(csv_results['위험요인']).most_common(5)
                top_actions = Counter(csv_results['안전조치방법']).most_common(5)
                
                st.markdown("#### 위험요인 (Top 5)")
                for risk, count in top_risks: st.markdown(f"- **{risk}** (반복 {count}회)")
                
                st.markdown("#### 안전조치방법 (Top 5)")
                for action, count in top_actions: st.markdown(f"- **{action}** (반복 {count}회)")


# =========================================================
# tab2: 시설물 검색 (유지보수 세칙 기반) - '점검항목 및 주기 요약' 삭제
# =========================================================
with tab2:
    st.subheader("1. 설비명 기반 점검 항목 및 주기 검색 (유지보수 세칙)")
    
    maintenance_query = st.text_input(
        "검색할 **설비명/시설물명**을 입력하세요 (예: 변압기, 신호기장치):", 
        key="maintenance_input"
    )
    
    if st.button("시설물/세칙 검색", key="maintenance_button") and maintenance_query:
        if maintenance_df.empty: 
              st.warning("⚠️ 유지보수 세칙 데이터가 로드되지 않아 검색할 수 없습니다. (사이드바 확인)")
        else:
            with st.spinner(f"'{maintenance_query}'에 대한 점검 정보 검색 중..."):
                maintenance_results = search_maintenance_data(maintenance_df, maintenance_query) 
            
            if maintenance_results.empty:
                st.warning(f"🚨 **'{maintenance_query}'**에 대한 점검 정보 결과가 없습니다. (유사도 검색 포함)")
            else:
                st.success(f"✅ **'{maintenance_query}'**와 관련된 {len(maintenance_results):,}개의 점검 정보 결과입니다.")

                # --- 1. 상세 조회 결과 (DataFrame) ---
                st.subheader("상세 조회 결과")
                display_cols = ["세칙명", "설비명", "점검주기", "점검종류", "점검항목"]
                display_cols = [c for c in display_cols if c in maintenance_results.columns]
                
                st.dataframe(maintenance_results[display_cols], use_container_width=True)
                
                # --- 2. 문장 요약 결과 (신규 요청) ---
                summary_sentences = generate_maintenance_summary_sentences(maintenance_results)
                
                st.markdown("---")
                st.subheader("💬 문장 요약 결과")
                
                if summary_sentences:
                    for sentence in summary_sentences:
                        st.markdown(f"- {sentence}")
                else:

                    st.info("문장 요약 결과를 생성할 수 없습니다.")
