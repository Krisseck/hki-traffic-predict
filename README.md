# hki-traffic-predict

Keras neural network to predict traffic in Helsinki

## Installation

Note: this project can only be run with Python 3.

> pip install -r requirements.txt

## Usage

Check the `active_model` variable for which model will be used

Then run:

> python train.py

That trains the model and saves it as the name of the activei\_model variable, like `dense_1.h5`

## Results

**dense_1**

` - 51s - loss: 0.0317 - val_loss: 0.0309`

**dense_2**

` - 56s - loss: 0.0327 - val_loss: 0.0320`

**lstm_1**

` - 237s - loss: 0.0371 - val_loss: 0.0362`

## Other

Source of CSV: https://hri.fi/data/dataset/liikennemaarat-helsingissa
