# Stock price prediction using Deep learning and Frequency Decomposition
> The main aim of the project was to run the experiments for Stock Price Prediction using various deep learning models, and extending the sequential models such as LSTM, RNN to extend and incorporate other frequency deomposition methods. The entire pipleline was created from scratch. All the experiments performed in below is a part of the original paper cited in the reference.

## Abstract

## Datasets

## Methodology

## Training and Preprocessing

## Experimentations

## Results

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


