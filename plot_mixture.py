#!/usr/bin/env python3

import argparse
import sys
import matplotlib.pyplot as plt
from math import ceil
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
    mweight = float(sects[_MW])
    means = [float(x) for x in sects[_MR][1:-1].split(',')[:-1]]
    vref = int(sects[_VREF])
    vidx = int(sects[_VIDX])
    vs = [float(x) for x in sects[_VR][1:-1].split(',')[:-1]]
    color = 'go'
    if mref == 0:
        color = 'ro'
    axs.plot(means[0], means[1], color, alpha=0.5)
    axs.text(means[0], means[1], f'{midx} {mweight}')

def _draw_group(g, axs, fts):
    lines = g.split('\n')
    now = lines[0]
    alignment = [int(x) for x in lines[-2].split(' ')[:-1]]
    axs.plot(fts[alignment,0], fts[alignment,1], 'bo', markersize=0.3)
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
