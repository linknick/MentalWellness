import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
class Model:
    def __init__(self, model_type="XGBoost",**kwargs):
        self.model_type = model_type

        if model_type == "NN":
            self.model = NN(kwargs['input_size'])
            self.criterion = nn.MSELoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)
            
        elif model_type == "XGBoost":
            self.model = xgb.XGBRegressor(
                            objective="reg:squarederror",
                            n_estimators=200,
                            learning_rate=0.1,
                            max_depth=4,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            random_state=42
                            )
            
        elif model_type == "LinearRegression":
            self.model = LinearRegression()
        else:
            raise ValueError("Unsupported model_type")
    def fit(self, train_loader):
        if self.model_type == "NN":
            for epoch in range(100):
                self.model.train()
                running_loss = 0.0
                for i, data in enumerate(train_loader, 0):
                    inputs, labels = data
                    self.optimizer.zero_grad()

                    outputs = self.model(inputs)
                    
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()
                    if i % 10 == 9: # Print every 10 batches
                        print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 10:.3f}")
                        running_loss = 0.0

        else:
            self.model.fit(*train_loader)
    
    
    def predict(self,test_loader):
        if self.model_type == "NN":
            preds, labels = [], []

        
            for x, y in test_loader:
                out = self.model(x)
                preds.extend(out.detach().numpy())
                labels.extend(y.detach().numpy())

            Y_test, Y_pred = labels, preds
            
        else:
            Y_test, Y_pred = test_loader[1].squeeze(), self.model.predict(test_loader[0]).squeeze()
        
        return Y_test, Y_pred
    
    
class NN(nn.Module):
    def __init__(self, input_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x