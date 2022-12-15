#!/usr/bin/env python3

import argparse
import sys
import matplotlib.pyplot as plt
from math import ceil, e
import numpy as np
import re

_HEIGHT = 60

_LAST_REF = []

def _draw_group(g, time):
    global _LAST_REF
    lines = g.strip().split('\n')
    layers = []
    for l in range(5):
        layers.append([float(i) for i in lines[l].split()[:-1]])

    if len(_LAST_REF) == len(layers[4]) and all([x == y for x, y in zip(layers[4], _LAST_REF)]):
        return
    _LAST_REF = layers[4][:]

    _WIDTH = max([len(l) for l in layers])
    YALL = np.arange(0, _HEIGHT, _HEIGHT / len(layers))
    color = 'go'
    scales = []
    for i, layer in enumerate(layers):
        scale = 10/max([abs(i) for i in layer])
        scales.append(scale)
        X = np.arange(0, _WIDTH, _WIDTH / len(layer))
        print(len(X))
        Y = np.zeros((len(X)))
        Y.fill(YALL[i])
        plt.plot([X[0], X[-1]], [Y[0], Y[-1]], markersize=0.1, alpha=0.2)
        for j, v in enumerate(layer):
            dv = np.sign(v) * np.log(np.abs(v))
            plt.plot([X[j], X[j]], [Y[j], Y[j] + scale*v], markersize=0.3)

    plt.title(f'Time {time}, Scales {scales}')
    margin = [_WIDTH/20, _HEIGHT/20]
    plt.axis([-margin[0], _WIDTH+margin[0], -margin[1] - 10, _HEIGHT - margin[1]])
    plt.show()

def _main():
    parser = argparse.ArgumentParser(prog = 'amscore_plot', description = 'plots linear segmentation')
    parser.add_argument('file')
    args = parser.parse_args()
    with open(args.file, 'r') as f:
        file = f.read()
    groups = file.split('==')[:-1]
    for i, group in enumerate(groups[:]):
        _draw_group(group, i)

if __name__ == '__main__':
    _main()
