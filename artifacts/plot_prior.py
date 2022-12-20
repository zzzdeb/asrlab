#!/usr/bin/env python3

import argparse
import sys
import matplotlib.pyplot as plt
from math import ceil, e
import numpy as np

_HEIGHT = 60
_WIDTH = 200

def _draw_group(g, epoch):
    lines = g.strip().split('\n')
    layers = []
    for l in range(4):
        layers.append([float(i) for i in lines[l].split(' ')[:-1]])

    print([len(l) for l in layers])
    YALL = np.arange(0, _HEIGHT, _HEIGHT / len(layers))
    color = 'go'
    for i, layer in enumerate(layers):
        scale = 1
        if (i == len(layers)-1):
            scale = 10
        X = np.arange(0, _WIDTH, _WIDTH / len(layer))
        print(len(X))
        Y = np.zeros((len(X)))
        Y.fill(YALL[i])
        plt.plot([X[0], X[-1]], [Y[0], Y[-1]], markersize=0.1, alpha=0.2)
        for j, v in enumerate(layer):
            dv = np.sign(v) * np.log(np.abs(v))
            plt.plot([X[j], X[j]], [Y[j], Y[j] + scale*v], markersize=0.3)

    plt.title(f'Epoch {epoch}')
    plt.axis([-1, _WIDTH+1, -20, _HEIGHT - 10])
    plt.show()

def _main():
    parser = argparse.ArgumentParser(prog = 'prior-plot', description = 'plots linear segmentation')
    parser.add_argument('file')
    args = parser.parse_args()
    with open(args.file, 'r') as f:
        prior = f.readlines()[0]
    prior = [float(x) for x in prior.strip().split(' ')]
    plt.title('Prior')
    for i, p in enumerate(prior):
        plt.plot([i, i], [0, p])
    plt.show()

if __name__ == '__main__':
    _main()
