#%%
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from typing import NamedTuple
import torch
from torchvision.datasets import MNIST
from torchvision import transforms as trans
import time

transform_x = trans.Compose(
    [trans.Lambda(lambda x: torch.flatten(x, start_dim=1)),
    trans.Lambda(lambda x: jnp.asarray(x)/255.0)]
)
transform_y = trans.Compose(
    [trans.Lambda(lambda x: jnp.asarray(x)), 
    trans.Lambda(lambda x: jax.nn.one_hot(x, num_classes=10))]
)
train= MNIST(root="./data", download=True, train = True)
test = MNIST(root="./data", download=True, train = False)
x_train = transform_x(train.data)
y_train = transform_y(train.targets)
x_test = transform_x(test.data)
y_test = test.targets

#%%
class Params(NamedTuple):
    w1: jnp.ndarray
    w2 : jnp.ndarray

INP = 28*28
OUP = 10

def init(rng, scale = 1e-2):
    hidden_size = 10
    k1, k2 = jax.random.split(rng)
    w1 = jax.random.normal(k1, (INP, hidden_size)) * scale
    w2 = jax.random.normal(k2, (hidden_size, OUP)) * scale
    return Params(w1, w2)

def linear(layer_weights : jnp.ndarray, inp : jnp.ndarray, activation = jax.nn.relu):
    return activation(jnp.dot(inp, layer_weights))

def forward(params, x):
    xs = linear(params.w1, x)
    xs = linear(params.w2, xs)
    return xs

def mse(params, x, y):
    y_pred = forward(params, x)
    return jnp.mean(jnp.power(y - y_pred,2)) 

def cross_entropy_loss(params, x, y):
    return optax.softmax_cross_entropy(logits = forward(params, x), labels=y).mean()

def update(*, params, loss_fn, x, y, lr):
    grad : Params = jax.grad(loss_fn)(params, x, y)
    return Params(
        w1 = params.w1 - grad.w1*lr,
        w2 = params.w2 - grad.w2*lr
    )
    
@jax.jit
def train_step(params, x, y, lr = 0.005):
    return update(
        params = params, 
        loss_fn = cross_entropy_loss, 
        x = x,
        y = y,
        lr = lr
    )

def accuracy_init(x_test, y_test):
    def accuracy(params):
        y_hat = forward(params, x_test)
        count = 0
        for pred, actual in zip(jnp.argmax(y_hat, axis=-1), y_test):
            if int(pred)==int(actual):
                count+=1
        return count/len(y_test)
    return accuracy

accuracy = accuracy_init(x_test, y_test)

def train(*, params, x, y, lr = 0.005, epochs = 20):
    for e in range(epochs):
        params = train_step(params, x, y, lr = lr)
        if ((e+1)%(epochs//10) == 0):
            print(f'epoch : {e+1} ---> accuracy = {accuracy(params)}')
    return params

#%% 
rng = jax.random.PRNGKey(0)
params = init(rng)
start = time.time()
params = train(params = params, x = x_train, y=y_train, lr=0.1, epochs=500) # ~15s -> 0.83% acc
print(f'training took : {time.time() - start}s')
print(accuracy(params))

#%%
