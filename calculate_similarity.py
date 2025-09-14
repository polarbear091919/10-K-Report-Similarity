"""
임베딩 기반 기업 간 유사도 계산
코사인 유사도, k-NN, 유사도 매트릭스 생성
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json

def load_embeddings(pickle_file_path: str) -> Tuple[Dict, Dict]:
    """
    저장된 임베딩 파일 로드

    Returns:
        embeddings: ticker -> embedding vector 딕셔너리
        metadata: 메타데이터 정보
    """
    with open(pickle_file_path, 'rb') as f:
        data = pickle.load(f)

    embeddings = data['embeddings']
    metadata = data.get('metadata', {})

    print(f"로드 완료: {len(embeddings)}개 기업 임베딩")
    print(f"임베딩 차원: {data.get('dimensions', 'Unknown')}")

    return embeddings, metadata

def calculate_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    두 임베딩 벡터 간의 코사인 유사도 계산

    Parameters:
        embedding1: 첫 번째 임베딩 벡터
        embedding2: 두 번째 임베딩 벡터

    Returns:
        코사인 유사도 (0~1)
    """
    # 벡터를 2D 배열로 변환 (sklearn 요구사항)
    embedding1 = np.array(embedding1).reshape(1, -1)
    embedding2 = np.array(embedding2).reshape(1, -1)

    # 정규화
    embedding1 = normalize(embedding1)
    embedding2 = normalize(embedding2)

    # 코사인 유사도 계산
    similarity = cosine_similarity(embedding1, embedding2)[0, 0]

    return similarity

def create_similarity_matrix(embeddings: Dict) -> pd.DataFrame:
    """
    모든 기업 간 유사도 매트릭스 생성

    Parameters:
        embeddings: ticker -> embedding vector 딕셔너리

    Returns:
        유사도 매트릭스 (DataFrame)
    """
    tickers = list(embeddings.keys())
    n = len(tickers)

    # 임베딩을 numpy 배열로 변환
    embedding_matrix = np.array([embeddings[ticker] for ticker in tickers])

    # 정규화
    embedding_matrix = normalize(embedding_matrix)

    # 코사인 유사도 매트릭스 계산
    similarity_matrix = cosine_similarity(embedding_matrix)

    # DataFrame으로 변환
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=tickers,
        columns=tickers
    )

    return similarity_df

def find_k_nearest_neighbors(
    target_ticker: str,
    embeddings: Dict,
    k: int = 5,
    exclude_self: bool = True
) -> List[Tuple[str, float]]:
    """
    특정 기업의 k개 가장 유사한 기업 찾기

    Parameters:
        target_ticker: 대상 기업 ticker
        embeddings: ticker -> embedding vector 딕셔너리
        k: 찾을 이웃 수
        exclude_self: 자기 자신 제외 여부

    Returns:
        [(ticker, similarity_score), ...] 형태의 리스트
    """
    if target_ticker not in embeddings:
        raise ValueError(f"Ticker {target_ticker} not found in embeddings")

    target_embedding = np.array(embeddings[target_ticker]).reshape(1, -1)
    target_embedding = normalize(target_embedding)

    # 모든 기업과의 유사도 계산
    similarities = []
    for ticker, embedding in embeddings.items():
        if exclude_self and ticker == target_ticker:
            continue

        embedding = np.array(embedding).reshape(1, -1)
        embedding = normalize(embedding)

        similarity = cosine_similarity(target_embedding, embedding)[0, 0]
        similarities.append((ticker, similarity))

    # 유사도 기준 정렬
    similarities.sort(key=lambda x: x[1], reverse=True)

    # 상위 k개 반환
    return similarities[:k]

def analyze_similarity_distribution(similarity_matrix: pd.DataFrame) -> Dict:
    """
    유사도 분포 분석

    Parameters:
        similarity_matrix: 유사도 매트릭스

    Returns:
        통계 정보 딕셔너리
    """
    # 대각선 제외 (자기 자신과의 유사도)
    mask = np.ones_like(similarity_matrix, dtype=bool)
    np.fill_diagonal(mask, False)

    similarities = similarity_matrix.values[mask]

    stats = {
        'mean': np.mean(similarities),
        'std': np.std(similarities),
        'min': np.min(similarities),
        'max': np.max(similarities),
        'q25': np.percentile(similarities, 25),
        'q50': np.percentile(similarities, 50),
        'q75': np.percentile(similarities, 75)
    }

    return stats

def visualize_similarity_matrix(
    similarity_matrix: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 10),
    save_path: str = None
):
    """
    유사도 매트릭스 히트맵 시각화

    Parameters:
        similarity_matrix: 유사도 매트릭스
        figsize: 그림 크기
        save_path: 저장 경로 (None이면 저장 안 함)
    """
    plt.figure(figsize=figsize)

    # 히트맵 그리기
    sns.heatmap(
        similarity_matrix,
        cmap='RdYlBu_r',
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Cosine Similarity'},
        xticklabels=True,
        yticklabels=True
    )

    plt.title('Company Similarity Matrix (Cosine Similarity)', fontsize=16)
    plt.xlabel('Company Ticker', fontsize=12)
    plt.ylabel('Company Ticker', fontsize=12)

    # 작은 글씨로 표시
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"히트맵 저장 완료: {save_path}")

    plt.show()

def visualize_top_k_neighbors(
    target_ticker: str,
    neighbors: List[Tuple[str, float]],
    save_path: str = None
):
    """
    특정 기업의 가장 유사한 k개 기업 시각화

    Parameters:
        target_ticker: 대상 기업
        neighbors: k-NN 결과
        save_path: 저장 경로
    """
    tickers = [ticker for ticker, _ in neighbors]
    similarities = [sim for _, sim in neighbors]

    plt.figure(figsize=(10, 6))

    # 막대 그래프
    bars = plt.bar(range(len(tickers)), similarities, color='steelblue')

    # 값 표시
    for i, (bar, sim) in enumerate(zip(bars, similarities)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{sim:.3f}', ha='center', va='bottom', fontsize=9)

    plt.xlabel('Company Ticker', fontsize=12)
    plt.ylabel('Cosine Similarity', fontsize=12)
    plt.title(f'Top {len(neighbors)} Most Similar Companies to {target_ticker}', fontsize=14)
    plt.xticks(range(len(tickers)), tickers, rotation=45, ha='right')
    plt.ylim(0, 1.1)

    # 그리드 추가
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"그래프 저장 완료: {save_path}")

    plt.show()

def main():
    """
    메인 실행 함수
    """
    # 임베딩 파일 경로 (실제 생성된 파일명으로 변경 필요)
    embedding_file = "embeddings_3072d_20250914_032157.pkl"  # 프로젝트에 업로드한 파일명

    try:
        # 1. 임베딩 로드
        print("="*50)
        print("1. 임베딩 로드")
        print("="*50)
        embeddings, metadata = load_embeddings(embedding_file)

        # 2. 유사도 매트릭스 생성
        print("\n" + "="*50)
        print("2. 유사도 매트릭스 생성")
        print("="*50)
        similarity_matrix = create_similarity_matrix(embeddings)
        print(f"유사도 매트릭스 크기: {similarity_matrix.shape}")

        # 3. 유사도 분포 분석
        print("\n" + "="*50)
        print("3. 유사도 분포 분석")
        print("="*50)
        stats = analyze_similarity_distribution(similarity_matrix)
        for key, value in stats.items():
            print(f"{key}: {value:.4f}")

        # 4. 각 기업별 k-NN 찾기
        print("\n" + "="*50)
        print("4. k-NN 분석 (예시: AAPL)")
        print("="*50)

        test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        k = 5

        for ticker in test_tickers:
            if ticker in embeddings:
                print(f"\n{ticker}의 가장 유사한 {k}개 기업:")
                neighbors = find_k_nearest_neighbors(ticker, embeddings, k=k)
                for i, (neighbor_ticker, similarity) in enumerate(neighbors, 1):
                    print(f"  {i}. {neighbor_ticker}: {similarity:.4f}")

        # 5. 시각화
        print("\n" + "="*50)
        print("5. 시각화")
        print("="*50)

        # 전체 매트릭스 히트맵 (기업이 많으면 일부만 표시)
        if len(embeddings) > 20:
            # 상위 20개 기업만 표시
            top_tickers = list(embeddings.keys())[:20]
            sub_matrix = similarity_matrix.loc[top_tickers, top_tickers]
            visualize_similarity_matrix(sub_matrix, save_path="similarity_heatmap_top20.png")
        else:
            visualize_similarity_matrix(similarity_matrix, save_path="similarity_heatmap.png")

        # 특정 기업의 k-NN 시각화
        if 'AAPL' in embeddings:
            neighbors = find_k_nearest_neighbors('AAPL', embeddings, k=10)
            visualize_top_k_neighbors('AAPL', neighbors, save_path="aapl_neighbors.png")

        # 6. 결과 저장
        print("\n" + "="*50)
        print("6. 결과 저장")
        print("="*50)

        # 유사도 매트릭스 CSV로 저장
        similarity_matrix.to_csv("similarity_matrix.csv")
        print("유사도 매트릭스 저장 완료: similarity_matrix.csv")

        # k-NN 결과 JSON으로 저장
        knn_results = {}
        for ticker in embeddings.keys():
            neighbors = find_k_nearest_neighbors(ticker, embeddings, k=10)
            knn_results[ticker] = [
                {'ticker': n_ticker, 'similarity': float(sim)}
                for n_ticker, sim in neighbors
            ]

        with open("knn_results.json", 'w', encoding='utf-8') as f:
            json.dump(knn_results, f, indent=2)
        print("k-NN 결과 저장 완료: knn_results.json")

    except FileNotFoundError:
        print(f"Error: 임베딩 파일 '{embedding_file}'을 찾을 수 없습니다.")
        print("먼저 create_embeddings.py를 실행하여 임베딩을 생성하세요.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()