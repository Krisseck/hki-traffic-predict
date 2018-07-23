# hki-traffic-predict

Keras neural network to predict traffic in Helsinki

## Installation

Note: this project can only be run with Python 3.

`pip install -r requirements.txt`

## Usage

There are several scripts included:

- **train_short_term.py** - Given the statistics for past 3 hours, make traffic predictions for the next 4 hours
- **train_shorter_term.py** - Given just the current (now) traffic data, make traffic predictions for the next 4 hours

Check the `active_model` variable for which model will be used

Then run:

`python train_short_term.py`

That trains the model and saves it as the name of the script + active\_model variable, like `short_term_dense_1.h5`

## Results

### short_term

conv1d_1

` - 0s - loss: 0.0271 - val_loss: 0.0263`

conv1d_2

` - 0s - loss: 0.0191 - val_loss: 0.0174`

conv1d_3

` - 0s - loss: 0.0151 - val_loss: 0.0149`

dense_1

` - 2s - loss: 0.0330 - val_loss: 0.0296`

lstm_1

` - 1s - loss: 0.0319 - val_loss: 0.0257`

lstm_2

` - 7s - loss: 0.0251 - val_loss: 0.0211`

lstm_3

` - 4s - loss: 0.0278 - val_loss: 0.0240`

### shorter_term

conv1d_1

`1s 78us/step - loss: 0.0261 - val_loss: 0.0231`

conv1d_2

`1s 82us/step - loss: 0.0236 - val_loss: 0.0205`

dense_1 

`1s 59us/step - loss: 0.0335 - val_loss: 0.0289`

dense_2

`1s 59us/step - loss: 0.0294 - val_loss: 0.0248`

dense_3

`1s 63us/step - loss: 0.0343 - val_loss: 0.0302`

dense_4

`1s 70us/step - loss: 0.0211 - val_loss: 0.0163`

lstm_1

`1s 78us/step - loss: 0.0239 - val_loss: 0.0207`

lstm_2

`5s 358us/step - loss: 0.0314 - val_loss: 0.0279`

## Other

Source of CSV: https://hri.fi/data/dataset/liikennemaarat-helsingissa
