import streamlit as st
import pandas as pd
import json
import io
import re
import fitz  # PyMuPDF
import datetime

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

    [선정 최우선 기준]
    1. (중소형주 집중) 삼성전자, SK하이닉스, 현대차, 셀트리온 등 시가총액 상위 대형주는 추천 대상에서 무조건 제외하세요.
    2. 뉴스 테마에 즉각적이고 민감하게 반응할 수 있는 '코스닥 및 코스피 중소형주(테마주, 개별주)' 위주로 10개를 발굴하세요.
    3. (이슈의 신선도) 시장에서 계속 언급된 낡은(Stale) 이슈는 배제하고, 오늘 새롭게 부각되었거나 당일 발생한 강력한 모멘텀을 가진 종목에 가장 큰 가중치를 부여하세요.
    4. (상승 여력) 이미 어제 상한가를 기록했거나 단기 급등한 종목은 피로도가 높다고 판단하여 상승 확률을 보수적으로(낮게) 평가하세요.
    5. (차트 위치 및 가격 메리트) 이미 차트상 높은 위치(신고가, 과열 구간)에 있는 종목보다는, 오랜 조정을 거쳐 바닥권에 있거나 낙폭과대 후 첫 반등이 기대되는 위치의 종목에 가장 높은 가중치를 부여하세요. (뉴스 텍스트 내 '낙폭과대', '바닥', '저점', '반등', '소외주' 등의 힌트를 적극적으로 포착하세요.)

    위 기준을 엄격히 적용하여, 반드시 상승 확률(%)이 높은 순서대로 내림차순 정렬하여 JSON 형식으로만 답변하세요.

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

# ------- 사이드 바 -----------
st.subheader("🗄️ Pinecone DB 데이터 적재 현황 스캐너")
st.write("지정한 기간 내에 DB에 데이터가 빠짐없이 들어있는지 점검합니다.")

col1, col2 = st.columns(2)
with col1:
    check_start = st.date_input("점검 시작일", datetime.date(2026, 3, 1), key="chk_start")
with col2:
    check_end = st.date_input("점검 종료일", datetime.date(2026, 3, 31), key="chk_end")

if st.button("DB 상태 스캔하기"):
    scan_results = []
    total_days = (check_end - check_start).days + 1
    date_list = [check_start + datetime.timedelta(days=x) for x in range(total_days)]
    
    scan_bar = st.progress(0)
    
    for i, current_date in enumerate(date_list):
        date_str_yymmdd = current_date.strftime("%y%m%d")
        date_str_full = current_date.strftime("%Y-%m-%d")
        
        # 대표님의 명명 규칙인 ID 생성
        target_id = f"daily_{date_str_yymmdd}_chunk_0"
        
        try:
            # 해당 ID로 Pinecone DB 찔러보기
            fetch_response = index.fetch(ids=[target_id])
            
            if target_id in fetch_response['vectors']:
                status = "🟢 데이터 있음"
                # (옵션) 텍스트가 몇 글자인지 길이도 확인
                text_len = len(fetch_response['vectors'][target_id]['metadata']['text'])
            else:
                status = "🔴 없음 (휴장일 또는 누락)"
                text_len = 0
                
            scan_results.append({
                "날짜": date_str_full,
                "Pinecone ID": target_id,
                "상태": status,
                "텍스트 길이": f"{text_len} 자" if text_len > 0 else "-"
            })
            
        except Exception as e:
            st.error(f"스캔 중 오류: {e}")
            
        scan_bar.progress((i + 1) / total_days)
        
    st.success("스캔 완료!")
    st.dataframe(pd.DataFrame(scan_results), use_container_width=True)


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

# --- Tab 3: 실제 데이터 기반 백테스트 시뮬레이션 ---
with tab3:
    st.subheader("🧪 벡터 DB 실제 데이터 기반 모델 성능 검증 (Backtest)")
    st.write("Pinecone DB에 저장된 과거 3년 치 데이터를 불러와 실제 모델의 승률을 채점합니다.")

    # 날짜 선택 UI
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("시작일", datetime.date(2026, 3, 1))
    with col2:
        end_date = st.date_input("종료일", datetime.date(2026, 3, 25))

    if st.button("실제 데이터로 시뮬레이션 시작", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        hit_count = 0
        total_days = (end_date - start_date).days + 1
        
        # 날짜 리스트 생성 및 Pinecone ID 조합
        date_list = [start_date + datetime.timedelta(days=x) for x in range(total_days)]
        
        for i, current_date in enumerate(date_list):
            date_str_yymmdd = current_date.strftime("%y%m%d") # 예: 260325
            date_str_full = current_date.strftime("%Y-%m-%d")
            
            # Pinecone ID 조합 (미리 구축하신 규칙 적용)
            # 청크가 여러 개라면 chunk_0, chunk_1 등을 순회해야 하지만, 여기선 기본 chunk_0 기준
            target_id = f"daily_{date_str_yymmdd}_chunk_0"
            
            status_text.text(f"[{date_str_full}] Pinecone 데이터 조회 중... ({i+1}/{total_days})")
            
            try:
                # 1. Pinecone에서 해당 날짜 ID의 데이터 Fetch (직접 조회)
                fetch_response = index.fetch(ids=[target_id])
                
                # 데이터가 없는 날(주말, 공휴일 등) 패스
                if target_id not in fetch_response['vectors']:
                    continue
                    
                raw_text = fetch_response['vectors'][target_id]['metadata']['text']
                
                # 2. [장전 뉴스]와 [장후 결과] 분리 (구축하신 '---' 구분자 활용)
                if "---" in raw_text:
                    parts = raw_text.split("---")
                    pre_news = parts[0].strip()
                    post_result = parts[1].strip()
                else:
                    pre_news = raw_text # 구분자가 없을 경우 전체를 장전뉴스로 가정
                    post_result = ""

                # 3. 모델 예측 (장전 뉴스만 AI에게 제공)
                status_text.text(f"[{date_str_full}] AI 예측 중... ({i+1}/{total_days})")
                predictions = get_stock_predictions(pre_news) # Tab 1에서 만든 함수 재사용
                predicted_stocks = [p["stock"] for p in predictions]
                
                # 4. 적중 판별 (AI 예측 종목이 실제 장후 결과에 등장하고, 긍정적 단어와 매칭되는지)
                is_hit = False
                for stock in predicted_stocks:
                    if stock in post_result and any(word in post_result for word in ["상승", "급등", "상한가", "강세", "돌파", "수혜"]):
                        is_hit = True
                        break

                if is_hit:
                    hit_count += 1

                # 5. 결과 기록
                results.append({
                    "날짜": date_str_full,
                    "AI 추천 종목": ", ".join(predicted_stocks) if predicted_stocks else "추천 없음",
                    "장후 실제 결과": post_result[:100] + "..." if len(post_result) > 100 else post_result, # 너무 길면 자름
                    "적중": "✅ 성공" if is_hit else "❌ 실패"
                })
                
            except Exception as e:
                st.warning(f"{date_str_full} 데이터 처리 중 오류: {e}")
                
            progress_bar.progress((i + 1) / total_days)

        # --- 최종 결과 출력 ---
        status_text.text("✨ 실제 데이터 백테스트 완료!")
        st.markdown("---")
        
        valid_days = len(results)
        if valid_days > 0:
            win_rate = (hit_count / valid_days) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("테스트 진행 일수", f"{valid_days}일 (휴장일 제외)")
            col2.metric("예측 적중 일수", f"{hit_count}일")
            col3.metric("실제 데이터 승률", f"{win_rate:.1f}%")
            
            st.dataframe(pd.DataFrame(results), use_container_width=True)
        else:
            st.error("선택한 기간에 해당하는 Pinecone 데이터가 없습니다.")
