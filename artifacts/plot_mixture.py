#!/usr/bin/env python3

import argparse
import sys
import matplotlib.pyplot as plt
from math import ceil, e
import numpy as np

_T = 0;
_MIDX = 1
_MREF = 2
_MW = 3
_MR = 4
_VREF = 5
_VIDX = 6
_VR = 7

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

def _draw_group(g, axs, fts):
    lines = g.split('\n')
    now = lines[0]
    alignment = [int(x) for x in lines[-2].split(' ')[:-1]]
    axs.plot(fts[alignment,0], fts[alignment,1], 'bo', markersize=0.5)
    axs.set_title(now)
    for line in lines[1:-2]:
        sects = line.split(' ')
        if sects[0] == 'n':
            _draw_norm(sects, axs)


def _main():
    parser = argparse.ArgumentParser(prog = 'amscore_plot', description = 'plots linear segmentation')
    parser.add_argument('file')
    parser.add_argument('feature_file')
    args = parser.parse_args()
    with open(args.file, 'r') as f:
        file = f.read()
    with open(args.feature_file, 'r') as f:
        features = f.readlines()
    fts = np.empty((len(features), 25))
    for i, fl in enumerate(features):
        fts[i] = [float(x) for x in fl.split(' ')[:-1] if x != '']
    groups = file.split('==')[1:]
    width = ceil(len(groups)**0.5)
    height = ceil(len(groups) / width)
    fig, axs = plt.subplots(height, width)
    for i, group in enumerate(groups):
        _draw_group(group, axs[int(i/width), i%width], fts)
    plt.show()

if __name__ == '__main__':
    _main()
