---
layout: post
title: "From a Single Neuron to a Multi-Layer Perceptron (MLP) in Python"
date: 2026-11-23
categories: [Machine Learning, Neural Networks]
---

---

Neural networks often look intimidating behind heavy frameworks, yet at their core, they are built from remarkably simple mathematical building blocks. Following [Andrej Karpathy's micrograd lecture](http://www.youtube.com/watch?v=VMj-3S1tku0), we can construct an entire neural network library in just a few dozen lines of Python by stacking three components: a **Neuron**, a **Layer**, and a **Multi-Layer Perceptron (MLP)**.

Assuming we already have a scalar autograd `Value` class that tracks gradients and supports basic operations (`+`, `*`, `tanh`), here is how each neural network layer comes together.

---

## 1. The Building Block: A Single Neuron (Perceptron)

Mathematically, an artificial neuron takes a list of inputs `x`, multiplies each input by a learnable weight `w`, adds a bias `b` (which controls how trigger-happy the neuron is), and passes the sum through a non-linear activation function like `tanh`.

`output = tanh(sum_{i} w_i x_i + b)`

In Python, we initialize random weights between `-1` and `1` for each input and implement `__call__` so we can pass inputs directly as `n(x)`:

``` 
import random

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # Compute w * x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b] 
```

* zip(self.w, x) pairs each weight with its corresponding input.

* parameters() collects all learnable Value objects (weights and bias) so we can update  them during gradient descent.

---

## 2. Scaling Up: A Layer of Neurons

A Layer is simply a collection of independent neurons that all receive the exact same input vector x. They do not connect to one another, each neuron computes its own output in parallel.

```
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
```

* `nin` is the number of inputs entering each neuron.

* `nout` is the number of neurons in the layer.

* Returning outs[0] when len(outs) == 1 is a handy convenience so a single-neuron output layer returns a scalar Value rather than a 1-element list.

---

## 3. Connecting Layers: The Multi-Layer Perceptron (MLP)

A Multi-Layer Perceptron (MLP) chains multiple Layer objects sequentially, the output of one layer becomes the input to the next.

```
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```
To build a network that takes 3 inputs, passes them through two hidden layers of 4 neurons, and produces 1 output.

```
x = [2.0, 3.0, -1.0]
model = MLP(3, [4, 4, 1])
prediction = model(x)
```
---

## 4. Training the MLP with Gradient Descent

With the architecture defined, training comes down to a four-step loop:

1. **Forward pass:** Compute predictions for the dataset.

2. **Loss calculation:** Measure how far predictions are from the target values using Mean Squared Error.

3. **Backward pass:** Reset old gradients (p.grad = 0.0) and call loss.backward() to compute fresh gradients via backpropagation.

4. **Parameter update:** Nudge every weight and bias slightly in the opposite direction of its gradient (p.data += -0.05 * p.grad) to minimize the loss.

```
# Sample dataset: 4 examples with 3 features each
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # Desired targets

for step in range(20):
    # 1. Forward pass
    ypred = [model(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # 2. Zero gradients before backprop
    for p in model.parameters():
        p.grad = 0.0

    # 3. Backward pass
    loss.backward()

    # 4. Update weights
    for p in model.parameters():
        p.data += -0.05 * p.grad

    print(f"Step {step}: loss = {loss.data:.4f}")
```