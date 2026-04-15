import os
import json
import re
import time
from collections import defaultdict
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import fitz  # PyMuPDF
from openai import OpenAI
from pinecone import Pinecone

# 1. 환경 변수 및 설정
GCP_CREDENTIALS_JSON = os.environ.get('GCP_CREDENTIALS')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '').strip()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '').strip()

# ★ 구글 드라이브 폴더 ID (수정 필수)
SOURCE_FOLDER_ID = '1XnTl0GnMRKcZZm6CoZIq5ncfm8xkRZ-V' 
TARGET_FOLDER_ID = '1Nor--eDStNRIZRV9P7LOLrwTEsEpkW0c'

# 2. 클라이언트 초기화
credentials_dict = json.loads(GCP_CREDENTIALS_JSON)
credentials = service_account.Credentials.from_service_account_info(credentials_dict)
drive_service = build('drive', 'v3', credentials=credentials)

openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("stock-rag-db")

def extract_text_from_pdf(file_id, report_type, max_retries=3):
    """PDF에서 텍스트를 추출 (통신 에러 시 최대 3회 재시도)"""
    for attempt in range(max_retries):
        try:
            request = drive_service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            
            doc = fitz.open(stream=fh.getvalue(), filetype="pdf")
            
            if report_type == 'pre':
                is_target_section = False
                extracted_lines = []
                for page in doc:
                    blocks = page.get_text("dict").get("blocks", [])
                    for b in blocks:
                        if "lines" not in b: continue
                        for l in b["lines"]:
                            line_text = "".join([s["text"] for s in l["spans"]]).strip()
                            if not line_text: continue
                            
                            clean_line = line_text.replace(" ", "")
                            if "<경제일반>" in clean_line:
                                is_target_section = True
                            if "<기타>" in clean_line and is_target_section:
                                return "\n".join(extracted_lines).strip()
                            if is_target_section:
                                extracted_lines.append(line_text)
                                
                return "\n".join(extracted_lines).strip() if extracted_lines else ""
                
            elif report_type == 'post':
                raw_text = ""
                for page in doc:
                    raw_text += page.get_text() + "\n"
                marker_suffix = "- to the DEEP ]"
                if marker_suffix in raw_text:
                    raw_text = raw_text.split(marker_suffix)[-1]
                return raw_text.strip()

        except Exception as e:
            print(f"   [경고] 파일 다운로드/파싱 실패 (시도 {attempt+1}/{max_retries}) - 에러: {e}")
            if attempt < max_retries - 1:
                time.sleep(3)  # 3초 대기 후 다시 시도
            else:
                print("   [치명적 오류] 3회 재시도 후에도 실패했습니다. 이 파일은 건너뜁니다.")
                raise e

def move_to_backup(file_id):
    """처리 완료된 파일을 백업 폴더로 이동"""
    drive_service.files().update(
        fileId=file_id,
        addParents=TARGET_FOLDER_ID,
        removeParents=SOURCE_FOLDER_ID,
        fields='id, parents'
    ).execute()

def get_pdfs_from_drive():
    """구글 드라이브 폴더 내의 모든 PDF 파일을 페이지네이션을 통해 전부 가져옵니다 (100개 제한 돌파)"""
    query = f"'{SOURCE_FOLDER_ID}' in parents and mimeType='application/pdf' and trashed=false"
    all_files = []
    page_token = None

    while True:
        response = drive_service.files().list(
            q=query, 
            pageSize=1000, 
            fields="nextPageToken, files(id, name)",
            pageToken=page_token
        ).execute()
        
        all_files.extend(response.get('files', []))
        page_token = response.get('nextPageToken', None)
        
        if page_token is None:
            break
            
    print(f"드라이브에서 총 {len(all_files)}개의 PDF 파일을 발견했습니다.")
    return all_files

def process_paired_pdfs():
    """파일 페어링 -> JSON 구조화 -> 인과관계 Chunking -> 임베딩 및 DB 업로드"""
    files = get_pdfs_from_drive()
    if not files:
        print("새로 업로드된 시황 PDF 파일이 없습니다.")
        return

    # 날짜별로 파일 그룹화 (하이픈 무시, 6자리 추출)
    paired_files = defaultdict(dict)
    for f in files:
        name = f['name']
        match = re.search(r'(\d{2})-?(\d{2})-?(\d{2})', name) 
        
        if match:
            date_str = match.group(1) + match.group(2) + match.group(3)
            if 'Signal Report' in name:
                paired_files[date_str]['pre'] = f
            elif 'Signal Evening' in name:
                paired_files[date_str]['post'] = f

    for date_str, pair in paired_files.items():
        if 'pre' in pair and 'post' in pair:
            print(f"\n[{date_str}] 인과관계 매칭 및 JSON 구조화 시작...")
            
            # ★ 핵심 방어막: 전체 프로세스를 try-except로 감싸서 특정 날짜의 에러가 전체를 멈추지 않게 함
            try:
                # 1) 각각의 텍스트 추출 (재시도 로직이 포함된 함수 호출)
                pre_text = extract_text_from_pdf(pair['pre']['id'], 'pre')
                post_text = extract_text_from_pdf(pair['post']['id'], 'post')
                
                # 추출 결과가 없으면 건너뜀
                if not pre_text or not post_text:
                    print(f"[{date_str}] ⚠️ 텍스트 추출 결과가 비어있어 건너뜁니다.")
                    continue

                # 2) GPT를 활용한 JSON 구조화 및 인과관계 기반 Chunking
                prompt = f"""
                당신은 주식 시장 데이터 엔지니어입니다.
                제공된 '장전 뉴스'와 '장후 결과' 데이터를 바탕으로, 아침의 이슈(원인)와 오후의 결과(상승/하락)를 논리적인 인과관계(테마/섹터) 단위로 묶어 JSON 형식으로 반환하세요.
                
                [요구사항]
                1. 반드시 아래 JSON 구조만 출력하세요.
                {{
                    "date": "20{date_str[:2]}-{date_str[2:4]}-{date_str[4:]}",
                    "chunks": [
                        {{
                            "pre_market": "해당 테마의 장전 핵심 이슈",
                            "post_market": "해당 테마의 오후 상승/하락 결과 및 이유"
                        }}
                    ]
                }}
                2. 각 chunk의 텍스트 길이는 합쳐서 1500자 내외가 되도록 적절히 분할하세요.
                
                [장전 뉴스]
                {pre_text[:3500]}
                
                [장후 결과]
                {post_text[:3500]}
                """
                
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                
                # 3) JSON 파싱 및 데이터 병합
                result_json = json.loads(response.choices[0].message.content)
                chunks = result_json.get("chunks", [])
                vectors_to_upsert = []
                
                print(f"[{date_str}] 총 {len(chunks)}개의 인과관계 Chunk로 분할되었습니다. 임베딩 진행 중...")
                
                for idx, chunk in enumerate(chunks):
                    pre_m = chunk.get("pre_market", "")
                    post_m = chunk.get("post_market", "")
                    
                    merged_text = f"[장전 뉴스]\n{pre_m}\n\n---\n\n[장후 결과]\n{post_m}"
                    
                    # 4) 임베딩 생성
                    embed_response = openai_client.embeddings.create(
                        input=merged_text,
                        model="text-embedding-3-large"
                    )
                    embedding_vector = embed_response.data[0].embedding
                    
                    # 5) 고유 ID 생성 (날짜 + 청크 인덱스)
                    chunk_id = f"daily_{date_str}_chunk_{idx}"
                    metadata = {
                        "source": f"{date_str}_chunk_{idx}",
                        "text": merged_text,
                        "date": result_json.get("date")
                    }
                    vectors_to_upsert.append((chunk_id, embedding_vector, metadata))
                
                # 6) Pinecone에 일괄 업로드
                if vectors_to_upsert:
                    index.upsert(vectors=vectors_to_upsert)
                    print(f"[{date_str}] ✅ Pinecone DB 업로드 완료 ({len(vectors_to_upsert)} chunks).")
                
                # 7) 백업 이동
                move_to_backup(pair['pre']['id'])
                move_to_backup(pair['post']['id'])
                print(f"[{date_str}] 🔄 업데이트 및 백업 이동 완벽하게 종료.")
                
            except Exception as e:
                print(f"[{date_str}] ❌ 전체 처리 중 에러 발생, 이 날짜는 건너뜁니다: {e}")
                
            finally:
                # 8) ★ 과부하 방지 (Rate Limit 대응): 성공하든 실패하든 다음 루프 전 무조건 2초 대기
                time.sleep(2)
                
        else:
            print(f"[{date_str}] 짝이 맞지 않아 대기 중입니다.")

if __name__ == "__main__":
    process_paired_pdfs()
