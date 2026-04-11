import os
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
from pypdf import PdfReader
from openai import OpenAI
from pinecone import Pinecone

# 1. 환경 변수 로드 (GitHub Secrets에서 주입됨)
GCP_CREDENTIALS_JSON = os.environ.get('GCP_CREDENTIALS')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# 구글 드라이브 폴더 ID (여기를 수정하세요!)
SOURCE_FOLDER_ID = '1XnTl0GnMRKcZZm6CoZIq5ncfm8xkRZ-V' 
TARGET_FOLDER_ID = '1Nor--eDStNRIZRV9P7LOLrwTEsEpkW0c'

# 2. 클라이언트 초기화
credentials_dict = json.loads(GCP_CREDENTIALS_JSON)
credentials = service_account.Credentials.from_service_account_info(credentials_dict)
drive_service = build('drive', 'v3', credentials=credentials)

openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("stock-rag-db") # Pinecone 인덱스 이름 확인

def get_pdfs_from_drive():
    """daily_pdf 폴더에서 PDF 파일 목록 가져오기"""
    query = f"'{SOURCE_FOLDER_ID}' in parents and mimeType='application/pdf' and trashed=false"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    return results.get('files', [])

def process_pdf(file_id, file_name):
    """PDF 다운로드, 텍스트 추출, GPT 전처리, 벡터 DB 업로드, 파일 이동"""
    print(f"[{file_name}] 처리 시작...")
    
    # 1) 파일 다운로드 & 텍스트 추출
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    
    fh.seek(0)
    reader = PdfReader(fh)
    raw_text = "".join([page.extract_text() for page in reader.pages])
    
    # 2) GPT-4o-mini를 활용한 텍스트 전처리 (장전 뉴스 / 장후 결과 분리)
    prompt = f"""
    다음은 일일 주식 시황 PDF 텍스트입니다. 
    이 데이터를 분석하여 '장전 뉴스(이슈/테마)'와 '장후 결과(상승 종목 및 이유)'로 명확히 분리해 주세요.
    반드시 '장전 뉴스'와 '장후 결과' 사이에 '---' 구분자를 넣어주세요.
    
    [원본 텍스트]
    {raw_text[:3000]} # 토큰 제한 방지를 위해 적절히 슬라이싱
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    processed_text = response.choices[0].message.content
    
    # 3) text-embedding-3-large 임베딩 생성 (차원: 3072)
    embed_response = openai_client.embeddings.create(
        input=processed_text,
        model="text-embedding-3-large"
    )
    embedding_vector = embed_response.data[0].embedding
    
    # 4) Pinecone DB 업로드 (파일명 기반 고유 ID 부여로 중복 방지)
    metadata = {
        "source": file_name,
        "text": processed_text
    }
    index.upsert(vectors=[(file_id, embedding_vector, metadata)])
    print(f"[{file_name}] DB 업로드 완료.")
    
    # 5) 처리 완료된 파일 processed_backup 폴더로 이동
    drive_service.files().update(
        fileId=file_id,
        addParents=TARGET_FOLDER_ID,
        removeParents=SOURCE_FOLDER_ID,
        fields='id, parents'
    ).execute()
    print(f"[{file_name}] 백업 폴더로 이동 완료.\n")

if __name__ == "__main__":
    files = get_pdfs_from_drive()
    if not files:
        print("새로 업로드된 시황 PDF 파일이 없습니다.")
    else:
        for f in files:
            process_pdf(f['id'], f['name'])
        print("모든 데일리 업데이트 작업이 완료되었습니다!")
