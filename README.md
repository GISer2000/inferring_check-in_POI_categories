# A Graph Convolutional Network Approach for Inferring Categories of Weibo Check-in Locations

## Structure
- `Constructing-graph-dataset.ipynb`: constructing a check-in graph based on user check-in sequences and output dataset in torch format.
- `Comparison-GCN-and-GCN_RES.ipynb`: comparing the prediction accuracy of standard GCNs and GCNs with residual connections.
- `Get-best-GCN_Res-model.ipynb`: setting an early stopping strategy to obtain the best performing model.
- `Fine-tuning-bert.ipynb`: fine-tuning the bert model using labelled data.
- `Kerbert`: analyse the linguistic differences that exist on different categories of check-in locations.
- `Baseline.ipynb`: baseline models used for comparison experiments.
- `Visualisation.ipynb`: visualisation of experimental results.
- `data/final_data/dataset_torch`: the dataset used for model training has been done to divide the training set and validation set. If the dataset has ‘_bert’, it means the default bert model encodes text, otherwise it is the fine-tuned bert model.
- `data/final_data/model`: trained GCN_RES models with the best performance.
- `data/final_data/baseline`: trained baseline models with the best performance.
- `data/final_data/other`: stores the verification results of all models.The four indicators include accuracy, macro-precision, macro-recall and macro-f1.

## Weibo check-in data
Getting Weibo check-in data requires the use of the Weibo API: https://open.weibo.com.

The URL to create a task to retrieve historical data is https://c.api.weibo.com/2/search/statuses/historical/create.json.

- Search parameter setting
  - Shanghai, 2018: "province": "31", "city": "31", "starttime": "1514736000000", "endtime": "1546271999000"
  - Shanghai, 2019: "province": "31", "city": "31", "starttime": "1514736000000", "endtime": "1546271999000"
  - Suzhou, 2018: "province": "32", "city": "5", "starttime": "1546272000000", "endtime": "1577807999000"

 **Note**: the maximum time interval between the start and end of each task cannot exceed one month.

## Installation

pip install -r requirements.txt

## Training

Use `train.py` to train the model. You can start training by entering the dataset name in the main function. The name of the dataset: **'shanghai_2018'**, **'shanghai_2019'** and **'suzhou_2018'**.

```python
python train.py 
```

## Testing

`test.py` is used to evaluate the trained model. Load the trained model and dataset. Output the confusion matrix and a csv file containing the four metrics (accuracy, macro-precision, macro-recall and macro-f1). Enter the dataset name in the main function to automatically match the trained model. The name of the dataset: **'shanghai_2018'**, **'shanghai_2019'** and **'suzhou_2018'**.

```python
python test.py 
```
