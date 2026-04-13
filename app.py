import streamlit as st
import pandas as pd
import json
import io
import re
import fitz  # PyMuPDF
from openai import OpenAI
from pinecone import Pinecone
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# --- [1] 페이지 및 클라이언트 초기화 ---
st.set_page_config(page_title="Stock RAG Partner", layout="wide", page_icon="📈")

# API 클라이언트 초기화 (Secrets 보안 로드)
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("stock-rag-db")
except Exception as e:
    st.error(f"API 키 설정 오류: {e}")
    st.stop()

# 구글 드라이브 서비스 빌드
@st.cache_resource
def get_drive_service():
    creds = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
    return build('drive', 'v3', credentials=creds)

# --- [2] 핵심 로직 함수 ---

# Signal Report 전용 텍스트 추출 (PyMuPDF)
def extract_signal_report_text(pdf_content):
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    is_target_section = False
    extracted_lines = []
    
    for page in doc:
        blocks = page.get_text("dict").get("blocks", [])
        for b in blocks:
            if "lines" not in b: continue
            for l in b["lines"]:
                line_text = "".join([s["text"] for s in l["spans"]]).strip()
                if not line_text: continue
                
                # 섹션 시작 및 종료 감지
                clean_line = line_text.replace(" ", "")
                if "<경제일반>" in clean_line:
                    is_target_section = True
                if "<기타>" in clean_line and is_target_section:
                    return "\n".join(extracted_lines)
                
                if is_target_section:
                    extracted_lines.append(line_text)
    return "\n".join(extracted_lines) if extracted_lines else None

# 상승 확률 기반 Top 10 종목 예측
def get_stock_predictions(context_text):
    prompt = f"""
    당신은 최고의 수익률을 내는 주식 퀀트 분석가입니다. 
    제공된 [장전 뉴스 데이터]를 분석하여 오늘 상승 확률이 가장 높은 종목을 10개 선정하세요.
    반드시 확률(%)이 높은 순서대로 내림차순 정렬하여 JSON 형식으로만 답변하세요.
    이미 어제 급등했던 종목은 상승 확률을 보수적으로 잡아 줘.
    그리고 계속 언급된 이슈 말고, 최근 오늘 발생된 이슈와 연관된 종목에 더 큰 가중치를 부여해 줘.

    {{
      "top_picks": [
        {{"stock": "종목명", "probability": 95, "reason": "상승 핵심 이유 1줄 요약"}}
      ]
    }}

    [장전 뉴스 데이터]
    {context_text}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.2
    )
    return json.loads(response.choices[0].message.content).get("top_picks", [])

# --- [3] UI 구성 (Tabs) ---
st.title("📈RAG & Signal Analysis")

tab1, tab2, tab3 = st.tabs(["🚀 실시간 리포트 분석", "🤖 RAG 투자 파트너", "🧪 백테스트 시뮬레이션"])

# --- Tab 1: 실시간 리포트 분석 (구글 드라이브 연동) ---
with tab1:
    st.subheader("오늘의 Signal Report 정밀 분석")
    st.write("구글 드라이브에서 최신 PDF를 가져와 상승 확률이 높은 종목을 추출합니다.")
    
    if st.button("분석 시작", type="primary"):
        with st.spinner("드라이브 연결 및 데이터 추출 중..."):
            try:
                service = get_drive_service()
                folder_id = st.secrets["GOOGLE_DRIVE_FOLDER_ID"]
                
                # 최신 파일 1개 검색
                query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
                results = service.files().list(q=query, orderBy="createdTime desc", pageSize=1).execute()
                files = results.get('files', [])
                
                if not files:
                    st.warning("폴더에 분석할 PDF 파일이 없습니다.")
                else:
                    file_id = files[0]['id']
                    st.info(f"분석 중인 파일: **{files[0]['name']}**")
                    
                    # 다운로드 및 전용 로직 파싱
                    request = service.files().get_media(fileId=file_id)
                    pdf_bytes = request.execute()
                    pre_market_text = extract_signal_report_text(pdf_bytes)
                    
                    if pre_market_text:
                        # Top 10 예측 실행
                        predictions = get_stock_predictions(pre_market_text)
                        df = pd.DataFrame(predictions)
                        
                        st.success("상승 확률 Top 10 종목을 찾았습니다.")
                        st.dataframe(
                            df,
                            column_config={
                                "probability": st.column_config.ProgressColumn(
                                    "상승 확률", format="%d%%", min_value=0, max_value=100
                                ),
                                "reason": st.column_config.TextColumn("분석 근거", width="large")
                            },
                            hide_index=True, use_container_width=True
                        )
                        with st.expander("추출된 원문 텍스트 확인"):
                            st.text(pre_market_text)
                    else:
                        st.error("리포트에서 타겟 섹션을 찾지 못했습니다. PDF 형식을 확인해 주세요.")
            except Exception as e:
                st.error(f"분석 중 오류 발생: {e}")

# --- Tab 2: RAG 투자 파트너 (대화형 챗봇) ---
with tab2:
    st.subheader("과거 3년 데이터 기반 Q&A")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "과거 시황이나 특정 종목에 대해 궁금한 점을 물어보세요!"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("질문을 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # 1. 벡터 검색
            embed = client.embeddings.create(input=prompt, model="text-embedding-3-large").data[0].embedding
            results = index.query(vector=embed, top_k=5, include_metadata=True)
            context = "\n\n".join([m.metadata['text'] for m in results.matches if 'text' in m.metadata])
            
            # 2. 답변 생성
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"당신은 투자 파트너입니다. 아래 참고 데이터를 기반으로 답변하세요.\n\n[데이터]\n{context}"},
                    {"role": "user", "content": prompt}
                ]
            )
            ans = response.choices

# --- Tab 3: 백테스트 시뮬레이션 ---
with tab3:
    st.subheader("과거 데이터 기반 모델 성능 검증 (Backtest)")
    st.write("과거의 '장전 뉴스'만 AI에게 보여주고, 장이 끝난 후의 '실제 결과'와 비교하여 예측 승률을 채점합니다.")

    if st.button("백테스트 시뮬레이션 시작", type="primary"):
        # 테스트용 가상의 과거 데이터 3일 치 (실제로는 DB나 CSV에서 불러옵니다)
        historical_data = [
            {"date": "2026-04-10", "pre_news": "<경제일반> A반도체, 엔비디아 향 HBM 공급 테스트 통과 임박... <기타>", "post_result": "A반도체 상한가 마감, 반도체 장비주 동반 상승"},
            {"date": "2026-04-11", "pre_news": "<경제일반> B바이오, FDA 신약 승인 보류 소식에 제약 섹터 전반적 우려... C제약은 반사이익 기대 <기타>", "post_result": "B바이오 하한가, C제약은 5% 상승 마감"},
            {"date": "2026-04-12", "pre_news": "<경제일반> D자동차, 북미 판매 호조로 역대 최대 분기 실적 달성 공시... <기타>", "post_result": "D자동차 8% 급등, 자동차 부품주 강세 주도"}
        ]

        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []
        hit_count = 0

        for i, data in enumerate(historical_data):
            status_text.text(f"[{data['date']}] 과거 데이터 분석 및 예측 중... ({i+1}/{len(historical_data)})")
            
            # 1. 모델 예측 (과거의 장전 뉴스만 제공)
            predictions = get_stock_predictions(data["pre_news"])
            predicted_stocks = [p["stock"] for p in predictions]
            
            # 2. 적중 판별 (예측 종목이 실제 '장후 결과' 텍스트에 들어있고, 긍정적 단어와 함께 쓰였는지)
            is_hit = False
            for stock in predicted_stocks:
                if stock in data["post_result"] and any(word in data["post_result"] for word in ["상승", "급등", "상한가", "강세"]):
                    is_hit = True
                    break # 하나라도 맞추면 그 날의 예측은 성공으로 간주

            if is_hit:
                hit_count += 1

            # 3. 결과 기록
            results.append({
                "날짜 (Date)": data["date"],
                "AI 추천 종목": ", ".join(predicted_stocks) if predicted_stocks else "추천 없음",
                "실제 장후 결과": data["post_result"],
                "적중 여부": "✅ 성공" if is_hit else "❌ 실패"
            })
            
            progress_bar.progress((i + 1) / len(historical_data))

        status_text.text("✨ 백테스트 시뮬레이션 완료!")
        
        # --- 결과 출력 화면 ---
        st.markdown("---")
        st.subheader("📊 백테스트 리포트")
        win_rate = (hit_count / len(historical_data)) * 100
        
        # 핵심 지표 표시
        col1, col2, col3 = st.columns(3)
        col1.metric("테스트 기간", f"{len(historical_data)}일")
        col2.metric("예측 적중 일수", f"{hit_count}일")
        col3.metric("AI 모델 승률 (Hit Rate)", f"{win_rate:.1f}%")
        
        # 상세 내역 표
        st.dataframe(pd.DataFrame(results), use_container_width=True)
