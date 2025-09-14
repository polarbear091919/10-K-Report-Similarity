# -*- coding: utf-8 -*-
import codecs

# UTF-8로 파일 생성
content = """# -*- coding: utf-8 -*-
import pandas as pd
import time
from sec_api import QueryApi, ExtractorApi
from typing import List, Dict, Optional
import logging
from datetime import datetime
import os
import re
from dataclasses import dataclass, asdict
import json
import pickle

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Document:
    \"\"\"청크 데이터를 저장하기 위한 Document 클래스\"\"\"
    ticker: str
    section_type: str  # 'business' 또는 'risk_factors'
    chunk_id: int
    total_chunks: int
    text: str
    chunk_size: int
    overlap_size: int
    start_position: int
    end_position: int
    filing_url: str
    extraction_date: str
    metadata: Dict = None

    def to_dict(self):
        \"\"\"Document를 dictionary로 변환\"\"\"
        return asdict(self)

    def to_json(self):
        \"\"\"Document를 JSON 문자열로 변환\"\"\"
        return json.dumps(self.to_dict(), ensure_ascii=False)


class TextChunker:
    \"\"\"텍스트를 청크 단위로 분할하는 클래스\"\"\"

    def __init__(self, chunk_size: int = 10000, overlap: int = 500):
        \"\"\"
        TextChunker 초기화

        Args:
            chunk_size: 각 청크의 최대 크기 (문자 수)
            overlap: 청크 간 중복될 크기 (문자 수)
        \"\"\"
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text_with_overlap(self, text: str) -> List[Dict]:
        \"\"\"
        텍스트를 중복을 가진 청크로 분할

        Args:
            text: 분할할 텍스트

        Returns:
            청크 정보의 리스트
        \"\"\"
        if not text:
            return []

        chunks = []
        text_length = len(text)

        # 문장의 끝부분을 찾기 위한 정규식
        sentence_endings = re.compile(r'[.!?]\\s+')

        start = 0
        chunk_id = 0

        while start < text_length:
            # 청크 종료 위치 계산
            end = start + self.chunk_size

            # 마지막 청크면 남은 부분 전부 포함
            if end >= text_length:
                end = text_length
            else:
                # 문장에서 가장 가까운 문장 끝 지점을 찾음
                chunk_text = text[start:end]
                matches = list(sentence_endings.finditer(chunk_text))

                if matches:
                    # 마지막 문장이 끝나는 지점에서 청크 종료
                    last_sentence_end = matches[-1].end()
                    end = start + last_sentence_end

            # 청크 생성
            chunk_info = {
                'chunk_id': chunk_id,
                'text': text[start:end],
                'start_position': start,
                'end_position': end
            }
            chunks.append(chunk_info)

            # 다음 청크 시작 위치 설정 (중복 포함)
            start = end - self.overlap if end < text_length else end
            chunk_id += 1

        # 총 청크 개수 추가
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk['total_chunks'] = total_chunks

        return chunks


class SEC10KExtractorChunked:
    \"\"\"SEC API를 사용하여 10-K 보고서에서 Business와 Risk Factors 섹션을 추출하고 청크로 분할하는 클래스\"\"\"

    def __init__(self, api_key: str, chunk_size: int = 10000, overlap: int = 500):
        \"\"\"
        SEC API 및 TextChunker 초기화

        Args:
            api_key: SEC API 인증 키
            chunk_size: 청크 크기
            overlap: 중복 크기
        \"\"\"
        self.api_key = api_key
        self.query_api = QueryApi(api_key=api_key)
        self.extractor_api = ExtractorApi(api_key=api_key)
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)

    def get_latest_10k_url(self, ticker: str) -> Optional[str]:
        \"\"\"
        특정 ticker의 가장 최신 10-K 보고서 URL 조회

        Args:
            ticker: 기업의 ticker 심볼

        Returns:
            10-K 보고서 URL 또는 None
        \"\"\"
        try:
            # 최신 10-K 보고서 검색 쿼리
            query = {
                "query": f"ticker:{ticker} AND formType:\\"10-K\\"",
                "from": "0",
                "size": "1",
                "sort": [{"filedAt": {"order": "desc"}}]
            }

            logger.info(f"Searching for latest 10-K filing for {ticker}...")
            filings = self.query_api.get_filings(query)

            if filings and filings['filings']:
                filing_url = filings['filings'][0]['linkToFilingDetails']
                logger.info(f"Found 10-K filing for {ticker}: {filing_url}")
                return filing_url
            else:
                logger.warning(f"No 10-K filing found for {ticker}")
                return None

        except Exception as e:
            logger.error(f"Error fetching 10-K URL for {ticker}: {str(e)}")
            return None

    def extract_and_chunk_section(self, filing_url: str, section_id: str, section_name: str, ticker: str) -> List[Document]:
        \"\"\"
        10-K 보고서에서 특정 섹션을 추출하고 청크로 분할

        Args:
            filing_url: 10-K 보고서 URL
            section_id: 섹션 ID ('1' for Business, '1A' for Risk Factors)
            section_name: 섹션 이름
            ticker: 기업 ticker

        Returns:
            Document 객체 리스트
        \"\"\"
        try:
            logger.info(f"Extracting {section_name} section...")
            section_text = self.extractor_api.get_section(filing_url, section_id, 'text')

            if not section_text:
                logger.warning(f"{section_name} section is empty")
                return []

            # 텍스트 정리 (과도한 공백 제거)
            section_text = ' '.join(section_text.split())
            logger.info(f"Successfully extracted {section_name} section ({len(section_text)} characters)")

            # 텍스트를 청크로 분할
            chunks = self.chunker.split_text_with_overlap(section_text)
            logger.info(f"Split {section_name} into {len(chunks)} chunks")

            # Document 객체 생성
            documents = []
            extraction_date = datetime.now().isoformat()

            for chunk_info in chunks:
                doc = Document(
                    ticker=ticker,
                    section_type=section_name.lower().replace(' ', '_'),
                    chunk_id=chunk_info['chunk_id'],
                    total_chunks=chunk_info['total_chunks'],
                    text=chunk_info['text'],
                    chunk_size=self.chunker.chunk_size,
                    overlap_size=self.chunker.overlap,
                    start_position=chunk_info['start_position'],
                    end_position=chunk_info['end_position'],
                    filing_url=filing_url,
                    extraction_date=extraction_date,
                    metadata={
                        'section_id': section_id,
                        'original_text_length': len(section_text)
                    }
                )
                documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Error extracting {section_name} section: {str(e)}")
            return []

    def extract_10k_data_chunked(self, ticker: str) -> List[Document]:
        \"\"\"
        특정 ticker의 10-K 보고서에서 Business와 Risk Factors 섹션을 추출하고 청크로 분할

        Args:
            ticker: 기업의 ticker 심볼

        Returns:
            Document 객체 리스트
        \"\"\"
        all_documents = []

        try:
            # 최신 10-K 보고서 URL 조회
            filing_url = self.get_latest_10k_url(ticker)

            if not filing_url:
                logger.warning(f"Could not find 10-K filing for {ticker}")
                return all_documents

            # Business 섹션 추출 및 청크 분할
            business_docs = self.extract_and_chunk_section(
                filing_url, '1', 'Business', ticker
            )
            all_documents.extend(business_docs)

            # API 호출 간 짧은 대기 (rate limiting 방지)
            time.sleep(0.5)

            # Risk Factors 섹션 추출 및 청크 분할
            risk_factors_docs = self.extract_and_chunk_section(
                filing_url, '1A', 'Risk Factors', ticker
            )
            all_documents.extend(risk_factors_docs)

            logger.info(f"Successfully extracted and chunked data for {ticker}: {len(all_documents)} total documents")

        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")

        return all_documents

    def extract_multiple_10k_data_chunked(self, tickers: List[str]) -> tuple[pd.DataFrame, List[Document]]:
        \"\"\"
        여러 기업의 10-K 보고서 데이터를 추출하고 청크로 분할하여 DataFrame과 Document 리스트로 반환

        Args:
            tickers: 기업 ticker 심볼 리스트

        Returns:
            (DataFrame, Document 리스트) 튜플
        \"\"\"
        all_documents = []
        df_records = []
        total = len(tickers)

        logger.info(f"Starting extraction for {total} companies...")

        for idx, ticker in enumerate(tickers, 1):
            logger.info(f"Processing {idx}/{total}: {ticker}")

            # 데이터 추출 및 청크 분할
            documents = self.extract_10k_data_chunked(ticker)
            all_documents.extend(documents)

            # DataFrame용 요약 생성
            for doc in documents:
                df_records.append({
                    'ticker': doc.ticker,
                    'section_type': doc.section_type,
                    'chunk_id': doc.chunk_id,
                    'total_chunks': doc.total_chunks,
                    'text_preview': doc.text[:200] + '...' if len(doc.text) > 200 else doc.text,
                    'text_length': len(doc.text),
                    'chunk_size': doc.chunk_size,
                    'overlap_size': doc.overlap_size,
                    'filing_url': doc.filing_url,
                    'extraction_date': doc.extraction_date
                })

            # API rate limiting을 위한 대기
            if idx < total:
                time.sleep(1)

        # DataFrame 생성
        df = pd.DataFrame(df_records)

        logger.info(f"Extraction complete. Processed {total} companies, created {len(all_documents)} documents")

        return df, all_documents


def save_documents(documents: List[Document], output_dir: str = 'extracted_documents'):
    \"\"\"
    Document 객체들을 다양한 형식으로 저장

    Args:
        documents: 저장할 Document 객체 리스트
        output_dir: 출력 디렉토리
    \"\"\"
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 날짜와 시간 정보
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 1. Pickle 형식으로 저장 (Document 객체 그대로 저장)
    pickle_file = os.path.join(output_dir, f'documents_{timestamp}.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump(documents, f)
    logger.info(f"Saved {len(documents)} documents to {pickle_file}")

    # 2. JSON 형식으로 저장 (텍스트 형식으로 저장)
    json_file = os.path.join(output_dir, f'documents_{timestamp}.json')
    json_data = [doc.to_dict() for doc in documents]
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved documents to {json_file}")

    # 3. 각 청크를 개별 텍스트 파일로 저장 (옵션)
    text_dir = os.path.join(output_dir, f'text_chunks_{timestamp}')
    os.makedirs(text_dir, exist_ok=True)

    for doc in documents:
        filename = f"{doc.ticker}_{doc.section_type}_chunk_{doc.chunk_id:03d}.txt"
        filepath = os.path.join(text_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Ticker: {doc.ticker}\\n")
            f.write(f"Section: {doc.section_type}\\n")
            f.write(f"Chunk: {doc.chunk_id + 1}/{doc.total_chunks}\\n")
            f.write(f"Position: {doc.start_position}-{doc.end_position}\\n")
            f.write(f"{'='*50}\\n\\n")
            f.write(doc.text)

    logger.info(f"Saved individual text chunks to {text_dir}")


def load_documents(pickle_file: str) -> List[Document]:
    \"\"\"
    저장된 Document 객체를 불러오기

    Args:
        pickle_file: pickle 파일 경로

    Returns:
        Document 객체 리스트
    \"\"\"
    with open(pickle_file, 'rb') as f:
        documents = pickle.load(f)
    logger.info(f"Loaded {len(documents)} documents from {pickle_file}")
    return documents


def load_tickers_from_csv(file_path: str, ticker_column: str = 'ticker') -> List[str]:
    \"\"\"
    CSV 파일에서 ticker 리스트를 읽어오는 함수

    Args:
        file_path: CSV 파일 경로
        ticker_column: ticker 정보가 있는 컬럼명 (기본값: 'ticker')

    Returns:
        ticker 리스트
    \"\"\"
    try:
        logger.info(f"Loading tickers from {file_path}...")

        # CSV 파일 읽기
        df = pd.read_csv(file_path)

        # ticker 컬럼 확인
        if ticker_column not in df.columns:
            # 대소문자 구분 없이 ticker 컬럼 찾기
            possible_columns = [col for col in df.columns if col.lower() == ticker_column.lower()]
            if possible_columns:
                ticker_column = possible_columns[0]
            else:
                logger.error(f"Column '{ticker_column}' not found in CSV file")
                logger.info(f"Available columns: {df.columns.tolist()}")
                return []

        # ticker 리스트 추출 (NaN 값 제거)
        tickers = df[ticker_column].dropna().astype(str).str.strip().tolist()

        # 빈 문자열 제거
        tickers = [t for t in tickers if t]

        logger.info(f"Loaded {len(tickers)} tickers from CSV file")
        return tickers

    except FileNotFoundError:
        logger.error(f"CSV file not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}")
        return []


def create_sample_ticker_csv(file_path: str = 'tickers.csv'):
    \"\"\"
    샘플 ticker CSV 파일 생성

    Args:
        file_path: 생성할 CSV 파일 경로
    \"\"\"
    sample_data = {
        'ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
        'company_name': [
            'Apple Inc.',
            'Microsoft Corporation',
            'Alphabet Inc.',
            'Amazon.com Inc.',
            'Tesla Inc.'
        ],
        'sector': [
            'Technology',
            'Technology',
            'Technology',
            'Consumer Cyclical',
            'Consumer Cyclical'
        ]
    }

    df = pd.DataFrame(sample_data)
    df.to_csv(file_path, index=False)
    logger.info(f"Sample ticker CSV file created: {file_path}")


def main():
    \"\"\"메인 실행 함수\"\"\"

    # SEC API 키 설정
    API_KEY = '861fc216ba170d5341679e9e7127f654864415c6245f952fc4c36301e8804e43'

    # 청크 설정
    CHUNK_SIZE = 10000  # 각 청크의 최대 10,000문자
    OVERLAP_SIZE = 500  # 청크 간 500문자 중복

    # ticker CSV 파일 경로
    ticker_csv_path = 'tickers.csv'

    # CSV 파일이 없으면 샘플 파일 생성
    if not os.path.exists(ticker_csv_path):
        logger.info("Ticker CSV file not found. Creating sample file...")
        create_sample_ticker_csv(ticker_csv_path)

    # CSV 파일에서 ticker 리스트 읽기
    tickers = load_tickers_from_csv(ticker_csv_path)

    if not tickers:
        logger.error("No tickers loaded. Please check your CSV file.")
        return None

    # 처리할 ticker 수 제한 (옵션)
    max_tickers = 3  # 테스트를 위해 처음 3개만 처리
    if len(tickers) > max_tickers:
        logger.info(f"Limiting to first {max_tickers} tickers for this run")
        tickers = tickers[:max_tickers]

    logger.info(f"Processing tickers: {tickers}")
    logger.info(f"Chunk settings - Size: {CHUNK_SIZE}, Overlap: {OVERLAP_SIZE}")

    # Extractor 인스턴스 생성
    extractor = SEC10KExtractorChunked(
        api_key=API_KEY,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP_SIZE
    )

    # 데이터 추출 및 청크 분할
    df, documents = extractor.extract_multiple_10k_data_chunked(tickers)

    # 결과 확인
    print("\\n=== Extraction Results ===")
    print(f"Total companies processed: {len(tickers)}")
    print(f"Total documents created: {len(documents)}")
    print(f"Total chunks in DataFrame: {len(df)}")

    # 기업별 청크 개수 출력
    ticker_summary = df.groupby(['ticker', 'section_type']).agg({
        'chunk_id': 'count',
        'text_length': 'sum'
    }).rename(columns={'chunk_id': 'num_chunks', 'text_length': 'total_text_length'})
    print("\\n=== Chunking Summary ===")
    print(ticker_summary)

    # CSV 파일로 저장 (청크 요약 데이터프레임)
    output_file = f"10k_chunks_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Summary saved to {output_file}")

    # Document 객체들 저장 (pickle, json, text)
    save_documents(documents)

    # Excel 파일로도 저장 (옵션)
    try:
        output_excel = f"10k_chunks_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        df.to_excel(output_excel, index=False, engine='openpyxl')
        logger.info(f"Summary saved to {output_excel}")
    except ImportError:
        logger.warning("openpyxl not installed. Skipping Excel export.")

    return df, documents


if __name__ == "__main__":
    # 메인 함수 실행
    df, docs = main()

    # 샘플 문서 출력 예시
    if docs:
        print(f"\\n=== Sample Document ===")
        sample_doc = docs[0]
        print(f"Ticker: {sample_doc.ticker}")
        print(f"Section: {sample_doc.section_type}")
        print(f"Chunk {sample_doc.chunk_id + 1}/{sample_doc.total_chunks}")
        print(f"Text preview: {sample_doc.text[:300]}...")
"""

# UTF-8 BOM으로 저장
with codecs.open('C:/Users/imyon/PycharmProjects/PythonProject2/data_collection_chunked.py', 'w', encoding='utf-8-sig') as f:
    f.write(content)

print("파일이 UTF-8 BOM 인코딩으로 생성되었습니다.")
print("PyCharm에서 Settings -> Editor -> File Encodings에서 UTF-8로 설정하세요.")