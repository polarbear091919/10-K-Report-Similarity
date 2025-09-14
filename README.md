# 기업 유사도 분석 시스템 (Company Similarity Analysis System)

SEC 10-K 보고서를 활용한 기업 간 유사도 측정 시스템

## 📋 프로젝트 개요

본 프로젝트는 미국 상장 기업들의 SEC 10-K 보고서에서 비즈니스 설명(Item 1) 및 리스크 요인(Item 1A) 섹션을 추출하여, OpenAI의 임베딩 모델을 통해 기업 간 유사도를 계산하는 시스템입니다.

### 주요 기능
- SEC API를 통한 10-K 보고서 자동 수집
- 텍스트 청킹(chunking)을 통한 대용량 문서 처리
- OpenAI text-embedding-3 모델을 활용한 임베딩 생성
- 코사인 유사도 기반 기업 간 유사도 계산
- k-NN(k-최근접 이웃) 알고리즘을 통한 유사 기업 탐색
- 시각화 및 결과 분석

## 🚀 시작하기

### 필수 요구사항
- Python 3.8 이상
- OpenAI API 키
- SEC API 키

### 설치

1. 필요한 패키지 설치:
```bash
pip install pandas numpy openai sec-api tqdm matplotlib seaborn scikit-learn
```

2. API 키 설정:
   - `create_embeddings.py`의 16번째 줄에 OpenAI API 키 입력
   - `data_collection_chunked.py`에 SEC API 키 입력

## 📁 프로젝트 구조

```
PythonProject2/
│
├── data_collection_chunked.py    # SEC 10-K 보고서 수집 및 청킹
├── create_embeddings.py          # OpenAI API를 통한 임베딩 생성
├── calculate_similarity.py       # 기업 간 유사도 계산 및 분석
├── create_utf8_chunked.py        # UTF-8 인코딩 변환 유틸리티
│
├── embeddings_*.pkl              # 생성된 임베딩 데이터 (3072d, 1536d, 768d)
├── embedding_metadata_*.json     # 임베딩 메타데이터
├── knn_results.json              # k-NN 분석 결과
├── similarity_matrix.csv         # 유사도 매트릭스
│
└── README.md                     # 프로젝트 문서
```

## 🔄 실행 순서

### 1단계: 데이터 수집
```bash
python data_collection_chunked.py
```
- S&P 500 기업들의 10-K 보고서 수집
- Item 1 (Business) 및 Item 1A (Risk Factors) 섹션 추출
- 텍스트를 8000자 단위로 청킹 (500자 오버랩)
- 결과: `company_documents_chunked_YYYYMMDD_HHMMSS.pkl`

### 2단계: 임베딩 생성
```bash
python create_embeddings.py
```
- 수집된 텍스트 데이터 전처리 (소문자 변환, 특수문자 제거 등)
- OpenAI text-embedding-3 모델을 통한 임베딩 벡터 생성
- 3가지 차원 옵션: 3072d, 1536d, 768d
- 결과: `embeddings_*d_YYYYMMDD_HHMMSS.pkl`

### 3단계: 유사도 분석
```bash
python calculate_similarity.py
```
- 코사인 유사도 매트릭스 생성
- k-NN 분석 수행
- 시각화 (히트맵, 막대 그래프)
- 결과:
  - `similarity_matrix.csv`
  - `knn_results.json`
  - `similarity_heatmap.png`
  - `aapl_neighbors.png`

## 📊 주요 모듈 설명

### data_collection_chunked.py
- **Document 클래스**: 청크 데이터 구조 정의
- **TextChunker 클래스**: 텍스트 분할 처리
- **SECDataCollector 클래스**: SEC API 연동 및 데이터 수집
- 청크 크기: 8000자 (오버랩 500자)

### create_embeddings.py
- **preprocess_text()**: 텍스트 전처리 (URL 제거, 소문자 변환 등)
- **create_embeddings()**: OpenAI API를 통한 임베딩 생성
- 배치 처리로 API 호출 최적화
- 3가지 차원 옵션 지원

### calculate_similarity.py
- **calculate_cosine_similarity()**: 두 벡터 간 코사인 유사도 계산
- **create_similarity_matrix()**: 전체 기업 간 유사도 매트릭스 생성
- **find_k_nearest_neighbors()**: k개의 가장 유사한 기업 탐색
- **visualize_*()**: 결과 시각화 함수들

## 📈 출력 결과

### 유사도 매트릭스 (similarity_matrix.csv)
- 모든 기업 쌍에 대한 코사인 유사도 값 (0~1)
- 대각선은 1 (자기 자신과의 유사도)

### k-NN 결과 (knn_results.json)
```json
{
  "AAPL": [
    {"ticker": "MSFT", "similarity": 0.8523},
    {"ticker": "GOOGL", "similarity": 0.8234},
    ...
  ],
  ...
}
```

### 시각화
- **similarity_heatmap.png**: 유사도 매트릭스 히트맵
- **aapl_neighbors.png**: AAPL의 가장 유사한 기업 10개 막대 그래프

## ⚙️ 주요 파라미터

### 청킹 설정
- `chunk_size`: 8000 (각 청크의 최대 문자 수)
- `overlap`: 500 (청크 간 중복 문자 수)

### 임베딩 설정
- `dimensions`: 3072, 1536, 768 중 선택
- `model`: "text-embedding-3-large" 또는 "text-embedding-3-small"

### 유사도 분석 설정
- `k`: k-NN에서 찾을 이웃 수 (기본값: 5)
- `test_tickers`: 예시 분석용 기업 목록

## 📝 주의사항

1. **API 키 보안**: API 키를 코드에 직접 입력하지 말고 환경 변수나 별도 설정 파일 사용 권장
2. **API 사용량**: OpenAI API는 유료이므로 대량 데이터 처리 시 비용 주의
3. **데이터 크기**: S&P 500 전체 기업 처리 시 상당한 시간과 저장 공간 필요
4. **메모리 사용**: 대량의 임베딩 데이터 로드 시 메모리 사용량 주의

## 🔬 분석 방법론

1. **데이터 수집**: SEC EDGAR 데이터베이스에서 최신 10-K 보고서 추출
2. **전처리**: 텍스트 정규화, 특수문자 제거, 소문자 변환
3. **임베딩**: OpenAI의 최신 임베딩 모델 활용
4. **유사도 계산**: 코사인 유사도를 통한 벡터 간 거리 측정
5. **결과 분석**: k-NN 및 클러스터링을 통한 유사 기업 그룹 식별

## 📚 참고 자료

- [SEC EDGAR API Documentation](https://sec-api.io/docs)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Scikit-learn Cosine Similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)

## 주의점
Similarity 절대값 수치 자체만 보면, 데이터가 좁은 구간에 몰려 있음을 알 수 있다. 따라서, 0~100 스케일로 측정하려면 전처리가 더 필요하다. 
가장 합리적 방법은 k-NN으로 기업별로 가장 유사도 높은 기업 N개를 선정하는 방법이다. (Company Simlarity using Large Language Models (1).pdf 참고) 

---

*마지막 업데이트: 2025년 9월 14일*