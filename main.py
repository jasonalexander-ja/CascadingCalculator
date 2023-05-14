from __future__ import annotations
from typing import Callable, Union, List
import math
import random


class Activation:

    def __init__(self, function: Callable[[int], int], derivative: Callable[[int], int]) -> None:
        self.function = function
        self.derivative = derivative

    function: Callable[[float], float]
    derivative: Callable[[float], float]


SIGMOID: Activation = Activation(lambda v: 1.0 / (1.0 + math.e ** -v), lambda v: v * (1.0 - v))


class Matrix:

    def __init__(self, rows: int, cols: int,  src: Union[List[List[float]], None] = None, randomise: bool = False) -> None:

        self.rows = rows
        self.cols = cols
        self.data = src if src is not None else [[0 for _ in range(cols)] for _ in range(rows)]

        if randomise:
            self.data = [[random.random() * 2.0 - 1.0 for _ in c] for c in self.data]

    rows: int
    cols: int
    data: List[List[float]]

    def multiply(self, other: Matrix) -> Matrix:

        if self.cols != other.rows:
            raise ArithmeticError(f"Tried to multiply a matrix of {self.cols} columns by a matrix of {other.rows} rows. ")
        
        data = [[
            sum([tc * other.data[tci][ci] for tci, tc in enumerate(self.data[ri])]) 
                for ci in range(other.cols)
        ] for ri in range(self.rows)]

        return Matrix(self.rows, other.cols, data)

    def add(self, other: Matrix) -> Matrix:

        if self.cols != other.cols or self.rows != other.rows:
            raise ArithmeticError(f"Attempted to add a matrix of {self.cols}/{self.rows} cols/rows, to a matrix of {other.cols}/{other.rows} cols/rows. ")
        
        data = [[c + (other.data[ri][ci]) for ci, c in enumerate(r)] for ri, r in enumerate(self.data)]

        return Matrix(self.rows, self.cols, data)

    def dot_multiply(self, other: Matrix) -> Matrix:

        if self.cols != other.cols or self.rows != other.rows:
            raise ArithmeticError(f"Attempted to dot multiply a matrix of {self.cols}/{self.rows} cols/rows, by a matrix of {other.cols}/{other.rows} cols/rows. ")
        
        data = [[c * (other.data[ri][ci]) for ci, c in enumerate(r)] for ri, r in enumerate(self.data)]

        return Matrix(self.rows, self.cols, data)

    def subtract(self, other: Matrix) -> Matrix:

        if self.cols != other.cols or self.rows != other.rows:
            raise ArithmeticError(f"Attempted to subtract a matrix of {self.cols}/{self.rows} cols/rows, from a matrix of {other.cols}/{other.rows} cols/rows. ")
        
        data = [[c - (other.data[ri][ci]) for ci, c in enumerate(r)] for ri, r in enumerate(self.data)]

        return Matrix(self.rows, self.cols, data)

    def map(self, func: Callable[[float], float]) -> Matrix:
        data = [[func(c) for c in r] for r in self.data]

        return Matrix(self.rows, self.cols, data)
    
    def transpose(self) -> Matrix:

        data = [[self.data[ci][ri] for ci in range(self.rows)] for ri in range(self.cols)]
        return Matrix(self.cols, self.rows, data)
    
    def clone(self) -> Matrix:

        data = [[c for c in r] for r in self.data]
        return Matrix(self.rows, self.cols, data)


class Network:

    def __init__(self, layers: List[int], activation: Activation, learning_rate: float) -> None:

        self.layers = layers
        self.weights = [Matrix(layers[li + 1], l, None, True) for li, l in enumerate(layers[:-1])]
        self.biases = [Matrix(layers[li + 1], 1, None, True) for li, l in enumerate(layers[:-1])]
        self.data = []
        self.activation = activation
        self.learning_rate = learning_rate

    layers: List[int]
    weights: List[Matrix]
    biases: List[Matrix]
    data: List[Matrix]
    activation: Activation
    learning_rate: float

    def feed_forwards(self, inputs: List[float]) -> List[float]:

        if len(inputs) != self.layers[0]:
            raise Exception(f"Tried to feed forward inputs of size {inputs.length}, into a network with a first layer size of {self.layers[0]} . ")

        current = Matrix(1, len(inputs), [inputs]).transpose()
        self.data = [current.clone()]

        for i in range(len(self.layers) - 1):
            
            current = self.weights[i].multiply(current.clone())\
                .add(self.biases[i])\
                .map(self.activation.function)
            
            self.data.append(current)

        return current.data[0]
    
    def back_propagate(self, outputs: List[float], targets: List[float]):

        if len(targets) != self.layers[len(self.layers) - 1]:
            raise Exception(f"Back propagation given {targets.length} targets, when the final layer has a size of {self.layers[len(self.layers) - 1]}. ")
        
        parsed = Matrix(1, len(outputs), [outputs])
        errors = Matrix(1, len(targets), [targets]).subtract(parsed)
        gradients = parsed.map(self.activation.derivative)

        for i in range(len(self.layers) - 2, -1, -1):

            gradients = gradients.dot_multiply(errors).map(lambda v: v * self.learning_rate)

            self.weights[i] = self.weights[i].add(gradients.multiply(self.data[i].transpose()))
            self.biases[i] = self.biases[i].add(gradients)

            errors = self.weights[i].transpose().multiply(errors)
            gradients = self.data[i].map(self.activation.derivative)

    def train(self, inputs: List[List[float]], targets: List[List[float]], epochs: int, logProgress: bool = False):
        for iepoch in range(epochs):
            if logProgress and (epochs < 100 or i % (epochs / 100) == 0):
                print(f"Epoch {iepoch + 1} of {epochs}")
            
            for i, input in enumerate(inputs):
                outputs = self.feed_forwards(input)
                self.back_propagate(outputs, targets[i])


def main():

    inputs = [
		[0.0, 0.0],
		[0.0, 1.0],
		[1.0, 0.0],
		[1.0, 1.0],
	]

    targets = [
		[0.0],
		[1.0],
		[1.0],
		[0.0],
	]

    network = Network([2, 3, 1], SIGMOID, 0.55)

    network.train(inputs, targets, 10000)

    print(f"0 and 0: {network.feed_forwards([0.0, 0.0])}")
    print(f"0 and 1: {network.feed_forwards([0.0, 1.0])}")
    print(f"1 and 0: {network.feed_forwards([1.0, 0.0])}")
    print(f"1 and 1: {network.feed_forwards([1.0, 1.0])}")


if __name__ == "__main__":
    main()
