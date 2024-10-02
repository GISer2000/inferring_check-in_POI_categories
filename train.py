import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import datetime

import torch
from model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, data, optimizer, criterion, save_model, epochs=200):
    train_loss = []
    val_loss = []
    val_accs = []
    
    early_stopping_counter = 0
    max_acc = 0.0

    model.train()
    for epoch in range(epochs+1):
        data = data.to(device)
        optimizer.zero_grad()
        _ = model(data.x, data.edge_index, data.edge_attr)
        out = F.log_softmax(_, dim=1)

        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        train_loss.append(loss.item())

        loss.backward()
        optimizer.step()

        preds = out.argmax(dim=1)[data.train_mask].cpu()
        acc = accuracy_score(data.y[data.train_mask].cpu(), preds)

        f1 = f1_score(data.y[data.train_mask].cpu(), preds, average='macro')

        model.eval()
        with torch.no_grad():
            val_loss_ = criterion(out[data.val_mask], data.y[data.val_mask])
            val_loss.append(val_loss_.item())
            
            out = F.log_softmax(model(data.x, data.edge_index, data.edge_attr), dim=1)
            
            val_acc = accuracy_score(data.y[data.val_mask].cpu(), out.argmax(dim=1)[data.val_mask].cpu())
            val_accs.append(val_acc)

            # 早停策略
            if val_acc > max_acc:
                max_acc = val_acc
                early_stopping_counter = 0
                torch.save(model, save_model)
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= 20 and epoch >=100: break

        if epoch % 10 == 0:
            print(f'Epoch [{epoch:03d}/{epochs}], Train_Loss: {loss.item():0.3f}, Val_Loss: {val_loss[-1]:0.3f}, Val_acc: {max(val_accs):.3f}')
            val_accs = []

def main(dataset_name):

    path_dataset = 'data/final_data/dataset_torch/'
    path_model_out = 'data/final_data/model/'
    layers = {
        'shanghai_2018': 2,
        'shanghai_2019': 2,
        'suzhou_2018': 3
    }
    date = str(datetime.date.today())
    
    dataset = torch.load(path_dataset + dataset_name + '.pth')
    gcn = GCN_RES(dataset.num_node_features, dataset.num_node_features, dataset.num_classes, layers[dataset_name]).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(gcn.parameters(), lr=1e-3, weight_decay=5e-4)
    train(gcn, dataset, optimizer, criterion, path_model_out + dataset_name + '_' + date + '.pth', epochs=250)



if __name__ == "__main__":

    dataset_name = 'shanghai_2018'  # 'shanghai_2019', 'suzhou_2018'
    main(dataset_name)