#!/usr/bin/env python3

import argparse
import sys
import matplotlib.pyplot as plt
from math import ceil, e
import numpy as np

def _draw_norm(sects, axs):
    midx = int(sects[_MIDX])
    mref = int(sects[_MREF])
    mweight = e**-float(sects[_MW])
    means = [float(x) for x in sects[_MR][1:-1].split(',')[:-1]]
    vref = int(sects[_VREF])
    vidx = int(sects[_VIDX])
    vs = [float(x) for x in sects[_VR][1:-1].split(',')[:-1]]
    color = 'go'
    if mref == 0:
        color = 'ro'
    x = means[0];
    y = means[1];
    axs.plot(x, y, color, alpha=0.3, markersize=20*mweight+0.1)
    axs.text(x, y, f'{midx}')
    normed1 = 0
    if vs[0] > 0:
        normed1 = 1/vs[0]**0.5
    normed2 = 0
    if vs[1] > 0:
        normed2 = 1/vs[1]**0.5
    a = [-normed1 + x, normed1 + x]
    b = [-normed2 + y, normed2 + y]
    vcolor = 'g'
    if mref == 0:
        vcolor = 'r'
    axs.plot(a, [y, y], vcolor, markersize=0.2, alpha=0.5)
    axs.plot([x, x], b, vcolor, markersize=0.2, alpha=0.5)

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
    parser = argparse.ArgumentParser(prog = 'amscore_plot', description = 'plots linear segmentation')
    parser.add_argument('file')
    args = parser.parse_args()
    with open(args.file, 'r') as f:
        file = f.read()
    groups = file.split('==')[1:]
    for i, group in enumerate(groups[:]):
        _draw_group(group, i)

if __name__ == '__main__':
    _main()
