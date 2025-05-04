import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 경로 설정
input_path = "./resource/example.csv"
output_path = "./result/example_clustered.csv"

# 1. 평균값 계산용 (전체 feature 평균)
features_list = ['avi', 'b02','b03','b04','b05','b08','b8A','b11','b12','b07',
                 'savi','ndwi','ndvi','ndmi','nbr','gndvi','bsi','evi','gci']

# 첫 번째로 전체 평균을 계산하기 위해 파일을 전체 읽음 (한 번만)
print("전체 평균 계산 중...")
df_full = pd.read_csv(input_path, usecols=features_list)
df_full = df_full.fillna(df_full.mean())
mean_values = df_full.mean()

# 정규화 및 KMeans 학습용
scaler = StandardScaler()
scaled_full = scaler.fit_transform(df_full)

kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(scaled_full)

print("클러스터링 모델 학습 완료")

# 청크 단위 처리
chunksize = 100_000
reader = pd.read_csv(input_path, chunksize=chunksize)
first_chunk = True

print("청크 처리 중...")

for chunk in reader:
    # NaN 처리
    chunk[features_list] = chunk[features_list].fillna(mean_values)

    # 정규화
    scaled = scaler.transform(chunk[features_list])

    # 클러스터 예측
    chunk['cluster'] = kmeans.predict(scaled)

    # 결과 저장
    chunk.to_csv(output_path, mode='w' if first_chunk else 'a', index=False, header=first_chunk)
    first_chunk = False

print("완료")

# # 7. 클러스터별 시각화 (예: NDVI vs EVI로)
# plt.figure(figsize=(8,6))
# plt.scatter(df['ndvi'], df['evi'], c=df['cluster'], cmap='tab10', alpha=0.7)
# plt.xlabel('NDVI')
# plt.ylabel('EVI')
# plt.title('NDVI vs EVI by Cluster')
# plt.colorbar(label='Cluster')
# plt.show()