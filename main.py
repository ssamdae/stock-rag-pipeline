import os
import json
import re
import time  # API 호출 제한 방지를 위한 모듈 추가
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

def extract_text_from_pdf(file_id, report_type):
    """PDF에서 텍스트를 추출하고 특정 마커 이후의 내용만 반환 (PyMuPDF 사용)"""
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

def move_to_backup(file_id):
    """처리 완료된 파일을 백업 폴더로 이동"""
    drive_service.files().update(
        fileId=file_id,
        addParents=TARGET_FOLDER_ID,
        removeParents=SOURCE_FOLDER_ID,
        fields='id, parents'
    ).execute()

def get_pdfs_from_drive():
    """구글 드라이브 폴더 내의 모든 PDF 파일을 페이지네이션을 통해 전부 가져옵니다 (100개 제한 해제)"""
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
            
    print(f"드라이브에서 총 {len(
