#!/usr/bin/env python3

import argparse
import sys
import matplotlib.pyplot as plt
from math import ceil
import numpy as np

_COLUMS = ['Split', 'Align', 'Estimate', 'Score']

def _main():
    parser = argparse.ArgumentParser(prog = 'amscore_plot', description = 'plots linear segmentation')
    parser.add_argument('score_data')
    args = parser.parse_args()
    with open(args.score_data, 'r') as f:
        lines = f.readlines();
    data = np.zeros((len(lines), 4))
    fig, axs = plt.subplots(4, 1)
    for i, line in enumerate(lines):
        data[i] = [int(x) for x in line.split(' ')[:3]] + [float(line.split(' ')[3])]
    for i in range(0, 4):
        axs[i].plot(range(0, len(lines)), data[:, i])
        axs[i].set_title(_COLUMS[i])
        axs[i].grid(True)
    plt.show()

if __name__ == '__main__':
    _main()
