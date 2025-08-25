import pandas as pd
import yaml
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

import analysis
from dataloader import Dataset
from model import Model


if __name__ == "__main__":
    #Load data
    with open("config.yaml", "r") as f:
        CONFIG = yaml.safe_load(f)
    DATA = pd.read_csv(CONFIG['dir'])
    #Analysis
    
    analysis.show_detail(DATA)
    analysis.show_heatmap(DATA)
    analysis.show_bar_charts(DATA)
    analysis.show_distribution(DATA['Sleep_Hours'],DATA['Mood_Score'])
    
    
    #preprocessing
    dataset = Dataset(CONFIG=CONFIG)
    train,test,input_size = dataset.split(DATA = DATA, CONFIG = CONFIG,test_size = 0.2, random_state = 0)
    
    model = Model(model_type = CONFIG['model_type'],input_size = input_size)
    model.fit(train)
    
    Y_test, Y_pred = model.predict(test)
    results = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})
    
    
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    r2 = r2_score(Y_test, Y_pred)

    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
