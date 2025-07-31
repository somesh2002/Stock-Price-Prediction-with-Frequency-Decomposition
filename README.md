# Stock price prediction using Deep learning and Frequency Decomposition
> The main aim of the project was to run the experiments for Stock Price Prediction using various deep learning models, and extending the sequential models such as LSTM, RNN to extend and incorporate other frequency deomposition methods. The entire pipleline was created from scratch. All the experiments performed in below is a part of the original paper cited in the reference.

## Abstract

There is a high volatility and Non Linearity within the Financial Time Series Data, and it made difficult to stoch price prediction. But the recent developments in Deep Learning Methods have shown significant imrovment in the analyzing these time series data. Further frequency decomposition technique such as Empirical Mode Decomposition (EMD) and Complete Ensemble Empirical Mode Decom
position (CEEMD) algorithms to decompose time series to differnt frequency spectra, and models can be trained to anayze differnt frequency spectra. Based on this theoritical fraework, there are two differnt models,CEEMD-CNN-LSTM and EMD-CNN-LSTM, which can be put to use for extracting deep features from the data. he concept of the suggested algorithm is that  when combining these models, some collaboration is established between them that could enhance the analytical power of the model. Further, the suggested algorithm with CEEMD provides better performance compared to EMD. 

## Datasets
The original dataset has been downloaded using yFinance for the following stock data ("S&P500","Dow Jones","DAX","Nikkei225"). Only Closing values of the stocks has been considered for the process. The timeline for the stock data is between the period from  January 2010 to October 2019. A csv file for the dataset has been also given in the dataset. Exact symbols can be accesses through the YFinance Website.

## Methodology

All the details about the methodology has been given in the report inside the docs folder. 

### Results

## Folder Structure
Overview:
- requirements.txt: Requirements file is available to create the environment.
- src/ : The folder contains all the source code that has been used in the experiment.
- datasets/ : Contains the dataset for running the model(Can be collected using yFinance too)
- models/ : Contains the best model trained on the dataset for multiple experiments
- docs/ : Contains all the figures and final report for the experiment.

## Language and Libraries Used
- Programming Language: Python
- Libraries: Pytorch, Numpy, Pandas, Scikit-Learn, YFinance


## Setup Environment
``` bash
conda create -n stock_freq python=3.10
conda activate stock_freq
pip install -r requirements.txt
```

## References
<!-- @article{Rezaei2020StockPP,
  title={Stock price prediction using deep learning and frequency decomposition},
  author={Hadi Rezaei and Hamidreza Faaljou and Gholamreza Mansourfar},
  journal={Expert Syst. Appl.},
  year={2020},
  volume={169},
  pages={114332},
  url={https://api.semanticscholar.org/CorpusID:229502511}
} -->

Rezaei, Hadi, Hamidreza Faaljou and Gholamreza Mansourfar. “Stock price prediction using deep learning and frequency decomposition.” Expert Syst. Appl. 169 (2020): 114332.

## Developer and License

This project is a part of the experiments as done in the original paper cited. Proper process needs to be followed before using the paper.

Developer Name: Somesh Agrawal

Original Paper:

Rezaei, Hadi, Hamidreza Faaljou and Gholamreza Mansourfar. “Stock price prediction using deep learning and frequency decomposition.” Expert Syst. Appl. 169 (2020): 114332.


