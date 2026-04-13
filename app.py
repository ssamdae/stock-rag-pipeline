import streamlit as st
import pandas as pd
import json
import io
import re
from openai import OpenAI
from pinecone import Pinecone
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import PyPDF2

# --- [1] 초기 설정 ---
st.set_page_config(page_title="Stock AI Partner", layout="wide")

# API 클라이언트 초기화
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index = pc.Index("stock-rag-db")

# 구글 드라이브 서비스 빌드 함수
def get_drive_service():
    creds = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
    return build('drive', 'v3', credentials=creds)

# --- [2] 핵심 로직 함수 ---

# PDF에서 장전 뉴스 텍스트 추출
def extract_premarket_text(pdf_content):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
    full_text = "".join([page.extract_text() for page in pdf_reader.pages])
    # 정규표현식: [장전 뉴스] 부터 [장후 결과] 전까지 추출
    pattern = r"\[장전 뉴스\](.*?)(?=\[장후 결과\]|---|[\n\r]{2,}\[|\Z)"
    match = re.search(pattern, full_text, re.DOTALL)
    return match.group(1).strip() if match else None

# 상승 종목 Top 10 분석 (JSON 응답)
def get_top_10_predictions(pre_market_text):
    prompt = f"""
    당신은 주식 퀀트 분석가입니다. 아래 [장전 뉴스]를 분석하여 오늘 상승 확률이 가장 높은 종목 10개를 선정하세요.
    반드시 확률(%)이 높은 순서대로 내림차순 정렬하여 JSON 형식으로만 답변하세요.
    
    {{
      "top_picks": [
        {{"stock": "종목명", "probability": 95, "reason": "상승 핵심 이유 1줄"}}
      ]
    }}

    [장전 뉴스]
    {pre_market_text}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.2
    )
    return json.loads(response.choices[0].message.content).get("top_picks", [])

# --- [3] UI 구성 ---
st.title("📈 RAG 시스템 & 시뮬레이터")

tab1, tab2 = st.tabs(["🚀 실시간 장전 분석", "🧪 모델 성능 시뮬레이션"])

# --- Tab 1: 실시간 분석 ---
with tab1:
    st.subheader("오늘의 구글 드라이브 리포트 분석")
    if st.button("드라이브에서 최신 PDF 가져와 분석 시작", type="primary"):
        service = get_drive_service()
        folder_id = st.secrets["GOOGLE_DRIVE_FOLDER_ID"]
        
        # 최신 파일 1개 검색
        query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
        results = service.files().list(q=query, orderBy="createdTime desc", pageSize=1).execute()
        files = results.get('files', [])
        
        if files:
            file_id = files[0]['id']
            st.info(f"분석 파일: {files[0]['name']}")
            
            # 다운로드 및 추출
            request = service.files().get_media(fileId=file_id)
            pdf_content = request.execute()
            pre_text = extract_premarket_text(pdf_content)
            
            if pre_text:
                predictions = get_top_10_predictions(pre_text)
                df = pd.DataFrame(predictions)
                
                st.success("분석 완료! 상승 확률 Top 10 종목입니다.")
                st.dataframe(
                    df,
                    column_config={
                        "probability": st.column_config.ProgressColumn("상승 확률", format="%d%%", min_value=0, max_value=100),
                        "reason": st.column_config.TextColumn("분석 근거", width="large")
                    },
                    hide_index=True, use_container_width=True
                )
            else:
                st.error("장전 뉴스 섹션을 찾을 수 없습니다.")
        else:
            st.warning("폴더에 PDF 파일이 없습니다.")

# --- Tab 2: 시뮬레이션 (백테스트) ---
with tab2:
    st.subheader("과거 데이터 기반 모델 적중률 테스트")
    st.write("모델이 과거 장전 뉴스만 보고 예측한 결과와 실제 장후 결과를 대조합니다.")
    
    if st.button("백테스트 시뮬레이션 실행"):
        # 시뮬레이션용 샘플 데이터 (실제론 Pinecone이나 DB에서 호출)
        hist_samples = [
            {"date": "2026-03-20", "pre": "반도체 수주 확대 뉴스...", "post": "삼성전자 5% 상승, SK하이닉스 7% 급등"},
            {"date": "2026-03-21", "pre": "바이오 임상 성공 소식...", "post": "셀트리온 상한가, 바이오 섹터 강세"}
        ]
        
        sim_results = []
        for data in hist_samples:
            preds = get_top_10_predictions(data["pre"])
            # 간단한 적중 확인 로직 (실제 결과 텍스트에 종목이 있고 '상승' 단어가 있는지)
            hit_count = 0
            for p in preds:
                if p["stock"] in data["post"] and "상승" in data["post"] or "급등" in data["post"]:
                    hit_count += 1
            
            sim_results.append({
                "날짜": data["date"],
                "예측 Top 1": preds[0]["stock"] if preds else "N/A",
                "확률": preds[0]["probability"] if preds else 0,
                "실제 결과": data["post"],
                "적중여부": "✅" if hit_count > 0 else "❌"
            })
        
        st.table(pd.DataFrame(sim_results))
