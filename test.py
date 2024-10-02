
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datetime

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
from model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_confusion_matrix(y_pred, y_true, name=None):
    fig, ax = plt.subplots(1,1, figsize=(12,10))
    cm = confusion_matrix(y_true, y_pred)
    cm_normal = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
    ticks = list(index_to_label.keys())
    labels = list(index_to_label.values())
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)
    plt.xlabel('Predicted', fontsize=18)
    plt.ylabel('Ground Truth', fontsize=18)
    ax.tick_params(labelsize=16)
    im = ax.imshow(cm_normal, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    im.set_clim(0.0, 1.0)
    for i in range(np.shape(cm_normal)[0]):
        for j in range(np.shape(cm_normal)[1]):
            if int(cm_normal[i][j] * 100 + 0.5) >= 0:
                ax.text(j, i, str(cm[i][j]) + '\n' + str(round(cm_normal[i][j]*100,1))+'%',
                         ha="center", va="center", fontsize=18,
                         color="white" if cm_normal[i][j] > 0.5 else "black")  # 如果要更改颜色风格，需要同时更改此行
    plt.title(name, fontsize=36)
    fig.savefig(fig_out + f'confusion_matrix_{name}_{date}.svg', dpi=300)
    fig.savefig(fig_out + f'confusion_matrix_{name}_{date}.png', dpi=300)
    plt.show()

def test(model, data, dataset_name):
    model.eval()
    _ = model(data.x, data.edge_index, data.edge_attr)
    out = F.log_softmax(_, dim=1)
    
    y_pred = out.argmax(dim=1)[data.val_mask].cpu()
    y_true = data.y[data.val_mask].cpu()

    plot_confusion_matrix(y_pred, y_true, name=dataset_name)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    
    metrics_df = pd.DataFrame({
            'Dataset': [dataset_name],
            'Accuracy': [accuracy],
            'Precision': [precision],
            'Recall': [recall],
            'F1 Score': [f1]
        })

    metrics_df.to_csv(csv_out + dataset_name + '_' + date + '.csv', index=False)

    def main(dataset_name):

        path_dataset = 'data/final_data/dataset_torch/'
        path_model = 'data/final_data/model/'
        fig_out = 'data/fig/'  # 混淆矩阵输出目录
        csv_out = 'data/final_data/other/'  # 模型各指标分数的输出csv目录

        date = str(datetime.date.today())
        
        dataset = torch.load(path_dataset + dataset_name + '.pth').to(device)
        # 如果没有训练新的模型，使用训练好的模型
        try:
            model = torch.load(path_model + dataset_name + '_' + date + '.pth').to(device)
        except:
            model = torch.load(path_model + dataset_name + '.pth').to(device)

        test(model, dataset, dataset_name)

if __name__ == "__main__":

    dataset_name = 'shanghai_2018'  # 'shanghai_2019', 'suzhou_2018'
    main(dataset_name)