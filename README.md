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

**dense_3**

` - 51s - loss: 0.0316 - val_loss: 0.0303`

**conv1d_1**

` - 24s - loss: 0.0328 - val_loss: 0.0321`

**lstm_1**

` - 237s - loss: 0.0371 - val_loss: 0.0362`

**lstm_2**
` - 10s - loss: 0.0331 - val_loss: 0.0324`

## Other

Source of CSV: https://hri.fi/data/dataset/liikennemaarat-helsingissa
