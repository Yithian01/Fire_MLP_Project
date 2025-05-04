import torch
import torch.nn as nn
import pandas as pd
import joblib

# ✅ 모델 구조만 정의 (학습은 안 함)
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32, output_dim=10):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim2, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# 1. 새 데이터 로드
new_df = pd.read_csv('./../resource/example.csv')

# 기존 학습에 사용된 컬럼 순서 (usung_csv.csv에서 사용된 순서)
train_columns = ['vari', 'avi', 'b02', 'b03', 'b04', 'b01', 'b05', 'b08', 'b8A', 'b09', 'b11', 'b12', 'b07', 'savi', 'ndwi', 'ndvi', 'ndmi', 'nbr', 'gndvi', 'bsi', 'evi', 'gci']

# 2. X_new 컬럼 순서를 학습 시 순서대로 맞추기
X_new = new_df[train_columns]

# 3. 스케일러 로드 및 전처리
scaler = joblib.load('./../Spain_Fule_MLP/scaler.pkl')
X_scaled = scaler.transform(X_new)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# 4. 모델 로드 및 예측
model = MLP(input_dim=X_tensor.shape[1])
model.load_state_dict(torch.load('./../Spain_Fule_MLP/mlp_fuel_model.pth'))
model.eval()

with torch.no_grad():
    outputs = model(X_tensor)
    _, predicted = torch.max(outputs, 1)

# 5. 결과 저장
new_df['predicted_cluster'] = predicted.numpy()
new_df.to_csv('./../result/example_clustered.csv', index=False)

print("✅ 예측이 완료되었고, 결과가 'example_clustered.csv'에 저장되었습니다.")
