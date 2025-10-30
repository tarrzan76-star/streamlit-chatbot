import streamlit as st
import pandas as pd
from pathlib import Path
from typing import List, Dict
from difflib import SequenceMatcher
import re
from collections import Counter
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ──────────────────────────────────────────────────────────
# 기본 UI 및 경로
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="작업계획서 작성지원 + 문서 QA (Gemini)",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
CSV_DIR = BASE_DIR / "input_csv"
CSV_DIR.mkdir(exist_ok=True)

RISK_CSV_GLOBS = ["위험성평가(*).csv"]
MAINT_CSV_GLOBS = ["*유지보수 세칙*.csv"]

MAINTENANCE_NEEDED_COLUMNS = ["세칙명", "설비명", "점검항목", "점검주기", "점검종류"]

MAINTENANCE_COLUMN_MAPS: Dict[str, Dict[str, str]] = {
    "정보통신설비 유지보수 세칙.csv": {"설비명": "설비명", "점검 항목": "점검항목", "점검주기": "점검주기"},
    "신호제어설비 유지보수 세칙.csv": {
        "시설명": "설비명", "점검항목": "점검항목", "점검 세부항목": "점검 세부항목", "점검주기": "점검주기"},
    "전철전력설비 유지보수 세칙(송변전설비).csv": {
        "시설명": "설비명", "점검항목": "점검항목", "점검 세부항목": "점검 세부항목",
        "점검주기": "점검주기", "점검종류": "점검종류"},
    "전철전력설비 유지보수 세칙(전력설비).csv": {
        "시설명": "설비명", "점검항목": "점검항목", "점검주기": "점검주기", "점검종류": "점검종류"},
    "전철전력설비 유지보수 세칙(전차선로설비, 강체구간).csv": {
        "시설명": "설비명", "점검항목": "점검항목", "점검주기": "점검주기", "점검종류": "점검종류"},
    "전철전력설비 유지보수 세칙(전차선로설비, 지상구간).csv": {
        "시설명": "설비명", "점검항목": "점검항목", "점검주기": "점검주기"},
}

# ──────────────────────────────────────────────────────────
# CSV 유틸
# ──────────────────────────────────────────────────────────
def _get_rule_name_from_filename(filename: str) -> str:
    name = filename.replace(".csv", "").replace(".pdf", "")
    m = re.search(r"(.+설비)\s*유지보수\s*세칙", name)
    return (m.group(1).strip() + " 유지보수 세칙") if m else name.strip()

@st.cache_data(show_spinner=False)
def _read_csv_any(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except Exception:
        return pd.read_csv(path, encoding="cp949", low_memory=False)

@st.cache_data(show_spinner=False)
def load_risk_data() -> pd.DataFrame:
    all_dfs = []
    for pattern in RISK_CSV_GLOBS:
        for p in CSV_DIR.glob(pattern):
            try:
                df = _read_csv_any(p)
                df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
                if "작업명" not in df.columns:
                    if "작업 명" in df.columns: df["작업명"] = df["작업 명"]
                    elif "작업" in df.columns: df["작업명"] = df["작업"]
                if "위험요인" not in df.columns and "위험 요인" in df.columns:
                    df["위험요인"] = df["위험 요인"]
                if "안전조치방법" not in df.columns and "안전 조치 방법" in df.columns:
                    df["안전조치방법"] = df["안전 조치 방법"]

                need = [c for c in ["작업명", "위험요인", "안전조치방법"] if c in df.columns]
                if need:
                    df = df[need].copy()
                    for c in need:
                        df[c] = df[c].astype(str).str.replace(r"[\r\n\t]+", " ", regex=True).str.strip()
                    all_dfs.append(df)
            except Exception as e:
                st.warning(f"CSV 로드 실패: {p.name} → {type(e).__name__}")
    return pd.concat(all_dfs, ignore_index=True).dropna(how="all") if all_dfs else pd.DataFrame(columns=["작업명","위험요인","안전조치방법"])

@st.cache_data(show_spinner=False)
def load_maint_data() -> pd.DataFrame:
    all_dfs = []
    for pattern in MAINT_CSV_GLOBS:
        for p in CSV_DIR.glob(pattern):
            try:
                df = _read_csv_any(p)
                df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
                if p.name in MAINTENANCE_COLUMN_MAPS:
                    df = df.rename(columns=MAINTENANCE_COLUMN_MAPS[p.name])
                if "세칙명" not in df.columns:
                    df["세칙명"] = _get_rule_name_from_filename(p.name)
                if "점검항목" in df.columns and "점검 세부항목" in df.columns:
                    df["점검항목"] = df["점검항목"].astype(str).str.strip()
                    df["점검 세부항목"] = df["점검 세부항목"].astype(str).str.strip()
                    mask = df["점검 세부항목"].str.len() > 0
                    df.loc[mask, "점검항목"] = df.loc[mask, "점검항목"] + " (" + df.loc[mask, "점검 세부항목"] + ")"
                    df.drop(columns=["점검 세부항목"], inplace=True)
                for col in MAINTENANCE_NEEDED_COLUMNS:
                    if col not in df.columns: df[col] = ""
                for c in MAINTENANCE_NEEDED_COLUMNS:
                    df[c] = df[c].astype(str).str.replace(r"[\r\n\t]+", " ", regex=True).str.strip()
                all_dfs.append(df[MAINTENANCE_NEEDED_COLUMNS])
            except Exception as e:
                st.warning(f"세칙 CSV 로드 실패: {p.name} → {type(e).__name__}")
    return pd.concat(all_dfs, ignore_index=True).dropna(how="all") if all_dfs else pd.DataFrame(columns=MAINTENANCE_NEEDED_COLUMNS)

def search_risk(df: pd.DataFrame, q: str) -> pd.DataFrame:
    if df.empty or not q or "작업명" not in df.columns:
        return pd.DataFrame(columns=["작업명","위험요인","안전조치방법"])
    ql = q.lower().strip()
    exact = df[df["작업명"].str.contains(ql, case=False, na=False)].copy()
    if not exact.empty:
        return exact[["작업명","위험요인","안전조치방법"]].drop_duplicates().reset_index(drop=True)
    scores = {}
    for t in df["작업명"].astype(str).unique():
        if t.lower() == "nan": continue
        r = SequenceMatcher(None, ql, t.lower()).ratio()
        if r > 0.55: scores[t] = r
    if not scores: return pd.DataFrame(columns=["작업명","위험요인","안전조치방법"])
    best = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    return df[df["작업명"].isin([b[0] for b in best])][["작업명","위험요인","안전조치방법"]].drop_duplicates().reset_index(drop=True)

def search_maint(df: pd.DataFrame, q: str) -> pd.DataFrame:
    cols = MAINTENANCE_NEEDED_COLUMNS
    if df.empty or not q or cols[1] not in df.columns:
        return pd.DataFrame(columns=cols)
    ql = q.lower().strip()
    exact = df[df[cols[1]].str.contains(ql, case=False, na=False)].copy()
    if not exact.empty: return exact[cols].drop_duplicates().reset_index(drop=True)
    scores = {}
    for name in df[cols[1]].astype(str).unique():
        if name.lower() == "nan": continue
        r = SequenceMatcher(None, ql, name.lower()).ratio()
        if r > 0.55: scores[name] = r
    if not scores: return pd.DataFrame(columns=cols)
    best = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    return df[df[cols[1]].isin([b[0] for b in best])][cols].drop_duplicates().reset_index(drop=True)

def summarize_maint(df: pd.DataFrame) -> List[str]:
    if df.empty: return []
    df = df.copy()
    if "점검종류" in df.columns:
        df["점검종류"] = df["점검종류"].fillna("").astype(str).str.strip()
    group_cols = [c for c in ["설비명","점검주기","점검종류"] if c in df.columns]
    grouped = df.groupby(group_cols).agg(점검항목=("점검항목", lambda x: list(pd.Series(x).dropna().astype(str).unique()))).reset_index()
    sents = []
    for _, r in grouped.iterrows():
        items = r["점검항목"]
        if   len(items) >= 3: item_txt = f"'{items[0]}', '{items[1]}' 등"
        elif len(items) == 2: item_txt = f"'{items[0]}' 및 '{items[1]}'"
        elif len(items) == 1: item_txt = f"'{items[0]}'"
        else: continue
        if r.get("점검종류", ""):
            s = f"**{r['설비명']}**는 점검주기(**{r['점검주기']}**)에 따라 {item_txt}의 점검항목을 **{r['점검종류']}** 점검해야 합니다."
        else:
            s = f"**{r['설비명']}**는 점검주기(**{r['점검주기']}**)에 따라 {item_txt}의 점검항목을 점검해야 합니다."
        sents.append(s)
    return sents

# ──────────────────────────────────────────────────────────
# 문서 QA (지연 로딩: 무거운 라이브러리 import는 함수 안에서)
# ──────────────────────────────────────────────────────────
def build_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def get_text(uploaded_docs):
    from loguru import logger
    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
    docs = []
    for doc in uploaded_docs:
        file_name = doc.name
        with open(file_name, "wb") as f:
            f.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if file_name.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_name)
        elif file_name.lower().endswith(".docx"):
            loader = Docx2txtLoader(file_name)
        else:
            continue
        docs.extend(loader.load_and_split())
    return docs

def tiktoken_len(text: str) -> int:
    try:
        import tiktoken
        tok = tiktoken.get_encoding("cl100k_base")
        return len(tok.encode(text))
    except Exception:
        return len(text)

def split_chunks(documents):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=100, length_function=tiktoken_len
    )
    return splitter.split_documents(documents)

def chunks_to_vectordb(chunks):
    from langchain_community.vectorstores import FAISS
    emb = build_embeddings()
    return FAISS.from_documents(chunks, emb)

def get_conversation_chain(vstore, gemini_api_key: str):
    from langchain.chains import ConversationalRetrievalChain
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.memory import ConversationBufferMemory

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=gemini_api_key,
        temperature=0
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vstore.as_retriever(search_type="mmr", search_kwargs={"k": 4}),
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer"),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=False,
    )
    return chain

# ──────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────
st.title("🛡️ 작업계획서 작성지원 챗봇 + 📚 문서 QA (Gemini)")
st.caption("CSV 기반 위험성/세칙 검색과 업로드 문서 QA를 한 곳에서!")

with st.sidebar:
    st.subheader("🔧 공통 설정")
    gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not gemini_api_key:
        st.info("Secrets에 GEMINI_API_KEY를 등록하세요. (⋯ → Edit secrets)")
    st.markdown("---")
    st.subheader("📚 문서 업로드 (QA)")
    uploaded_files = st.file_uploader(
        "PDF / DOCX 파일 업로드",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )
    process_docs = st.button("문서 임베딩 생성")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"assistant","content":"안녕하세요! 문서 QA 또는 CSV 검색 탭에서 시작해 보세요."}]

qa_tab, risk_tab, maint_tab = st.tabs(["📖 문서 QA", "🔍 위험성 평가 (CSV)", "⚙️ 유지보수 세칙 (CSV)"])

with qa_tab:
    st.subheader("문서 기반 Q&A")
    st.info("좌측 사이드바에서 문서를 업로드한 뒤 **문서 임베딩 생성**을 눌러주세요.")
    if process_docs:
        if not gemini_api_key:
            st.warning("Gemini API Key를 Secrets에 추가해 주세요."); st.stop()
        if not uploaded_files:
            st.warning("문서를 업로드해 주세요."); st.stop()
        with st.spinner("문서 처리 및 벡터스토어 생성 중... (최초 실행은 시간이 걸릴 수 있습니다)"):
            docs = get_text(uploaded_files)
            chunks = split_chunks(docs)
            vdb = chunks_to_vectordb(chunks)
            st.session_state.qa_chain = get_conversation_chain(vdb, gemini_api_key)
        st.success("문서 임베딩 및 체인 준비 완료!")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_q = st.chat_input("문서에 대해 질문을 입력하세요")
    if user_q:
        if st.session_state.qa_chain is None:
            st.warning("먼저 문서를 업로드하고 임베딩을 생성해 주세요.")
        else:
            st.session_state.messages.append({"role":"user","content":user_q})
            with st.chat_message("user"):
                st.markdown(user_q)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state.qa_chain({"question": user_q})
                    answer = result.get("answer","")
                    srcs = result.get("source_documents",[])
                    st.markdown(answer)
                    if srcs:
                        with st.expander("참고 문서 보기"):
                            for i, d in enumerate(srcs[:5], start=1):
                                src = d.metadata.get("source","unknown")
                                page = d.metadata.get("page", None)
                                meta = f"{src}" + (f" (p.{page+1})" if isinstance(page, int) else "")
                                st.markdown(f"**{i}.** {meta}", help=d.page_content[:1000])
            st.session_state.messages.append({"role":"assistant","content":answer})

with risk_tab:
    st.subheader("작업명 기반 위험성 평가 검색")
    risk_df = load_risk_data()
    if risk_df.empty:
        st.error("input_csv 폴더에서 위험성평가 CSV를 찾지 못했습니다. 파일을 추가하세요.")
    else:
        st.success(f"로드된 위험성평가 항목: {len(risk_df):,}개")
    q = st.text_input("검색할 작업명 (예: 승강기 유지보수, 전차선 작업 등)")
    if st.button("위험성평가 검색") and q:
        with st.spinner(f"'{q}' 검색 중..."):
            res = search_risk(risk_df, q)
        if res.empty:
            st.warning(f"'{q}' 관련 결과가 없습니다 (유사도 검색 포함).")
        else:
            st.success(f"'{q}'와 관련된 {len(res):,}개 결과")
            st.dataframe(res, use_container_width=True)
            st.markdown("### 📝 주요 위험요인 / 안전조치 Top 5")
            top_risks = Counter(res["위험요인"]).most_common(5)
            top_actions = Counter(res["안전조치방법"]).most_common(5)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 위험요인")
                for t, c in top_risks: st.markdown(f"- **{t}** (반복 {c}회)")
            with c2:
                st.markdown("#### 안전조치방법")
                for t, c in top_actions: st.markdown(f"- **{t}** (반복 {c}회)")

with maint_tab:
    st.subheader("설비명 기반 점검 항목 및 주기 (유지보수 세칙)")
    maint_df = load_maint_data()
    if maint_df.empty:
        st.error("input_csv 폴더에서 유지보수 세칙 CSV를 찾지 못했습니다. 파일을 추가하세요.")
    else:
        st.success(f"로드된 세칙 항목: {len(maint_df):,}개")
    q2 = st.text_input("설비명/시설물명 (예: 변압기, 신호기장치 등)", key="maint_q")
    if st.button("세칙 검색", key="maint_btn") and q2:
        with st.spinner(f"'{q2}' 검색 중..."):
            res = search_maint(maint_df, q2)
        if res.empty:
            st.warning(f"'{q2}' 관련 결과가 없습니다 (유사도 검색 포함).")
        else:
            st.success(f"'{q2}' 관련 {len(res):,}개 결과")
            show_cols = [c for c in ["세칙명","설비명","점검주기","점검종류","점검항목"] if c in res.columns]
            st.dataframe(res[show_cols], use_container_width=True)
            st.markdown("---")
            st.subheader("💬 문장 요약 결과")
            for s in summarize_maint(res):
                st.markdown(f"- {s}")
