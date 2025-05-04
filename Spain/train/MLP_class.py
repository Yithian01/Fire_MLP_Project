import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import joblib  # 스케일러 저장용

# 1. 데이터 불러오기
df = pd.read_csv('./../result/example_clustered.csv')

# 2. feature, label 나누기
X = df.drop(columns=['cluster', 'fid', 'x', 'y'])  # 피처
y = df['cluster']  # 라벨 (0~9, 총 10개)

# 3. 전처리
scaler = StandardScaler()
X = scaler.fit_transform(X)

# numpy → torch
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)
y_test = torch.tensor(y_test.values, dtype=torch.long)

# 4. MLP 모델 정의 (논문 구조 기반)
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

model = MLP(input_dim=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 학습
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 6. 평가
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    print(classification_report(y_test, predicted))

# 7. 모델과 스케일러 저장
save_dir = './Spain_Fule_MLP'
os.makedirs(save_dir, exist_ok=True)  # 폴더 없으면 생성
torch.save(model.state_dict(), os.path.join(save_dir, 'mlp_fuel_model.pth'))
joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
print("모델과 스케일러 저장 완료!")