"""
OpenAI text-embedding-3-large 모델을 사용한 기업 임베딩 생성
논문의 전처리 방법론 적용 (소문자 변환, 특수문자 제거 등)
"""

import json
import re
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import pickle
from datetime import datetime
import os

# OpenAI API 설정
client = OpenAI(api_key="my code")  # API 키를 입력하세요

def preprocess_text(text):
    """
    논문의 전처리 방법론 적용
    - 소문자 변환
    - URL 제거
    - non-ASCII 문자 제거
    - 여러 공백을 하나로 통합
    """
    # URL 제거
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    # HTML 엔티티 처리 (&#8217; → ', &#8220; → " 등)
    text = text.replace('&#8217;', "'")
    text = text.replace('&#8220;', '"')
    text = text.replace('&#8221;', '"')
    text = text.replace('&#38;', '&')
    text = text.replace('&#174;', '®')
    text = text.replace('&#8482;', '™')

    # non-ASCII 문자 제거 (상표 기호 등은 유지)
    text = ''.join(char for char in text if ord(char) < 128 or char in ['®', '™'])

    # 소문자 변환
    text = text.lower()

    # 특수문자를 공백으로 변환 (알파벳, 숫자, 기본 구두점만 유지)
    text = re.sub(r'[^a-z0-9\s.,;:!?\'"()-]', ' ', text)

    # 여러 공백을 하나로 통합
    text = re.sub(r'\s+', ' ', text)

    # 앞뒤 공백 제거
    text = text.strip()

    return text

def generate_embedding(text, model="text-embedding-3-large", dimensions=None):
    """
    OpenAI API를 사용하여 텍스트 임베딩 생성

    Parameters:
    - text: 임베딩할 텍스트
    - model: 사용할 모델 (기본값: text-embedding-3-large)
    - dimensions: 임베딩 차원 (None이면 모델 기본값 3072 사용)
    """
    try:
        params = {
            "input": text,
            "model": model
        }

        # dimensions 파라미터 추가 (지정된 경우)
        if dimensions:
            params["dimensions"] = dimensions

        response = client.embeddings.create(**params)
        return response.data[0].embedding
    except Exception as e:
        print(f"임베딩 생성 중 오류 발생: {e}")
        return None

def process_documents(json_file_path, dimensions=None, save_interval=10):
    """
    JSON 파일의 모든 문서를 처리하여 임베딩 생성
    ticker별로 임베딩 평균 계산

    Parameters:
    - json_file_path: 처리할 JSON 파일 경로
    - dimensions: 임베딩 차원 (None, 1536, 768 등)
    - save_interval: 중간 저장 간격
    """
    # JSON 파일 로드
    print(f"JSON 파일 로드 중: {json_file_path}")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    # ticker별로 문서 그룹화
    ticker_documents = {}
    for doc in documents:
        ticker = doc['ticker']
        if ticker not in ticker_documents:
            ticker_documents[ticker] = []
        ticker_documents[ticker].append(doc)

    print(f"총 {len(ticker_documents)}개 기업 발견")

    # 각 ticker별로 임베딩 생성 및 평균 계산
    ticker_embeddings = {}
    ticker_metadata = {}

    for ticker, docs in tqdm(ticker_documents.items(), desc="기업별 임베딩 생성"):
        chunk_embeddings = []

        for doc in docs:
            # 텍스트 전처리
            preprocessed_text = preprocess_text(doc['text'])

            # 임베딩 생성
            embedding = generate_embedding(preprocessed_text, dimensions=dimensions)

            if embedding:
                chunk_embeddings.append(embedding)

        # ticker별 평균 임베딩 계산
        if chunk_embeddings:
            avg_embedding = np.mean(chunk_embeddings, axis=0).tolist()
            ticker_embeddings[ticker] = avg_embedding

            # 메타데이터 저장
            ticker_metadata[ticker] = {
                'num_chunks': len(docs),
                'total_chunks': docs[0]['total_chunks'] if docs else 0,
                'section_types': list(set(doc['section_type'] for doc in docs)),
                'filing_url': docs[0]['filing_url'] if docs else '',
                'extraction_date': docs[0]['extraction_date'] if docs else ''
            }

            print(f"{ticker}: {len(chunk_embeddings)}개 청크 처리 완료")

    return ticker_embeddings, ticker_metadata

def save_embeddings(embeddings, metadata, dimensions=None):
    """
    임베딩과 메타데이터 저장
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dim_suffix = f"_{dimensions}d" if dimensions else "_3072d"

    # 임베딩 저장 (pickle)
    embedding_file = f"embeddings{dim_suffix}_{timestamp}.pkl"
    with open(embedding_file, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'metadata': metadata,
            'model': 'text-embedding-3-large',
            'dimensions': dimensions or 3072
        }, f)

    print(f"임베딩 저장 완료: {embedding_file}")

    # 메타데이터 저장 (JSON)
    metadata_file = f"embedding_metadata{dim_suffix}_{timestamp}.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump({
            'tickers': list(embeddings.keys()),
            'num_companies': len(embeddings),
            'model': 'text-embedding-3-large',
            'dimensions': dimensions or 3072,
            'preprocessing': 'lowercase, remove_special_chars, remove_urls',
            'ticker_metadata': metadata
        }, f, indent=2)

    print(f"메타데이터 저장 완료: {metadata_file}")

    return embedding_file, metadata_file

def main():
    """
    메인 실행 함수
    """
    # 처리할 JSON 파일 경로
    json_file = "extracted_documents/documents_20250914_024825.json"

    # 차원별 임베딩 생성 (3072, 1536, 768)
    dimensions_to_test = [None, 1536, 768]  # None은 기본 3072 차원

    for dim in dimensions_to_test:
        print(f"\n{'='*50}")
        print(f"임베딩 생성 시작 - 차원: {dim or 3072}")
        print(f"{'='*50}")

        # 임베딩 생성
        embeddings, metadata = process_documents(json_file, dimensions=dim)

        # 결과 저장
        save_embeddings(embeddings, metadata, dimensions=dim)

        # 통계 출력
        print(f"\n처리 완료 통계:")
        print(f"- 총 기업 수: {len(embeddings)}")
        print(f"- 임베딩 차원: {dim or 3072}")
        if embeddings:
            first_ticker = list(embeddings.keys())[0]
            print(f"- 샘플 임베딩 길이 ({first_ticker}): {len(embeddings[first_ticker])}")

if __name__ == "__main__":
    # API 키 확인
    if "YOUR_API_KEY_HERE" in str(client.api_key):
        print("WARNING: OpenAI API 키를 설정해주세요!")
        print("client = OpenAI(api_key='your-actual-api-key')")
    else:
        main()