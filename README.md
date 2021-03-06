# A lightweight deep learning module
Implementation of some deep learning models based on Theano, GPU capable.

#### Requirements:
Python, Theano, NumPy


## Example:
```
from modelcompiler import ModelCompiler
from models import LSTM

lstm = ModelCompiler(LSTM, config, optimizer="RMSprop")
lstm.train(x_train, y_train, x_val, y_val, save_model=True)
```
Please see the `example.py` for more details

`class ModelCompiler`with method:

- `train()` training on training data and evaluation on validation data
- `load()`  restore the model from checkpoint
- `predict()` predict on test dataset
- `porba()` get prediction score


### Included models: 
- DNN (multi-layer perceptron)
- Vanilla RNN (Recurrent Neural Networks)
- LSTM (Long short term memory)

### Layers:
- Full connected layer
- Dropout layer
- Softmax layer
- RNN layer
- LSTM layer

### optimizer:
- SGD
- Adagrad
- RMSprop


**Author:** *Cyrus*, *Twbadkid* @ntu
