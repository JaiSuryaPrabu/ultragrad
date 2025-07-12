# Ultragrad

Ultragrad is a tensor based automatic gradient engine. Inspired from Andrej's [Micrograd](https://github.com/karpathy/micrograd). It is a small and custom engine that runs the deep learning model using `numpy`. It is mainly developed for understanding the core concept in the deep learning. 

## Components
Ultragrad contains the following components:
1. Tensors
2. Neural Networks
3. Optimizers
4. Losses

### Tensors
The `Tensor` are the core engine for the deep learning. It takes multi-dimensional array from `numpy` and converts it to tensor which will be useful for deep learning calculations. 
> By default the values are stored in `np.float64` data type

The `Tensor` class uses the **computation graph** which stores the tensors, it's children and operation used to create the children. 

It also contains the gradients for the tensors, a backward function specific for the each tensor.

Here is the code snippet for `Tensor`
```py
from ultragrad import Tensor

data1 = Tensor(data=[1,2,3])
data2 = Tensor(data=[2.0,3.0,4.0])

data3 = data1 + data2
data4 = data1 * data3

data1_sum = data.sum()
data2_mean = data.mean()

data5 = data1 @ data2
```

### Neural Networks

Neural Networks builds the architecture of the deep learning model. Here, we will use a base class `Module` which stores essential information like parameters and helps to load and save model parameters. 

In the version 0.1.0, `ultragrad` contains :
1. `Linear()` - Dense layer
2. `Sequential()` - same as pytorch class for collecting all the layers and makes help for forward pass
3. Activation Layers:
    a. `ReLU`
    b. `Sigmoid`
    c. `TanH`

The neural networks file `nn` contains additional two functions `save` to save the model in the `.safetensors` extenstion and to load the model we can use `load`

Here is the basic usage of `nn` code:
```py
from ultragrad.nn import Sequential, Linear, ReLU, save, load

model = Sequential(
    Linear(1,12),
    ReLU(),
    Linear(12,1)
)

# to save the model
save_path = "path/to/SAVE_PATH.safetensors" # ext is important
save(model,save_path)

# load the model
new_model = Sequential(
    Linear(1,12),
    ReLU(),
    Linear(12,1)
)

load(new_model,load_path)
```
### Optimizers
Optimizer part makes the deep learning model to improve the loss and learning.

In the version 0.1.0, `ultragrad` contains only the `SGD` stochastic gradient descent algorithm to train the deep learning model. 

Here is the code example for optimizer:
```py
from ultragrad.optim import SGD

optimizer = SGD(model.parameters(),0.01)
```

### Losses
Losses in this `ultragrad` helps the deep learning model to find the loss of the model's prediction and it contains two losses
1. MSELoss - Mean square error loss for regression based learning task
2. CrossEntropyLoss - Cross Entropy Loss is for categorization based learning task

Code for instantiating the loss function
```py
from ultragrad.losses import MSELoss

loss_fn = MSELoss()
```

## Training Loop
Here is the training loop as same as pytorch :
```py
for epoch in range(epoches):
    # 1. Forward pass
    y_pred = model(X_train)

    # 2. Calculate loss
    loss = loss_fn(y_pred,y_train)

    # 3. Zero Grad
    model.zero_grad()

    # 4. Backward pass
    loss.backward()

    # 5. Update the params
    optimizer.step()
```