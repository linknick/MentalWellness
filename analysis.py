import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
def show_detail(data):
    print(f"Data shape: ',{data.shape}")
    print('-'*20)
    print(data.head())
    print('-'*20)
    print(data.info())
    print('-'*20)
    print(data.describe())
    print('-'*20)
    print(data.describe(include='O'))
    print('-'*20)
    print("Missing values:\n")
    print(data.isnull().sum())
    
def show_heatmap(data):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.savefig(plt.gca().get_title()+'.png')
    
def show_bar_charts(data):
    for col in data.iloc[:,2:].select_dtypes(include='object'):
        ax = data.groupby(['Mood_Score', col]).size().unstack(fill_value=0)
        ax.plot(kind='bar', stacked=True)
        plt.title(col + " Count")
        plt.savefig(plt.gca().get_title()+'.png')
def show_distribution(data1,data2):
    plt.figure(figsize=(10, 8))
    plt.scatter(data1, data2,alpha = 0.5)
    m, b = np.polyfit(data1, data2, 1)  # y = m*x + b

    x_line = np.linspace(min(data1), max(data2), 200)
    coef1 = np.polyfit(data1,data2, 1)
    poly1 = np.poly1d(coef1)
    plt.xlabel(data1.name)
    plt.ylabel(data2.name)
    plt.plot(x_line, poly1(x_line), color='red', linewidth=2, label="Degree 1 (Linear)",alpha = 0.5)
    plt.savefig(data1.name+data2.name+'.png')
    