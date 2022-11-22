#!/usr/bin/env python3

import argparse
import sys
import matplotlib.pyplot as plt
from math import ceil, e
import numpy as np

def _main():
    parser = argparse.ArgumentParser(prog = 'amscore_plot', description = 'plots linear segmentation')
    parser.add_argument('file')
    args = parser.parse_args()
    with open(args.file, 'r') as f:
        file = f.read()
    groups = file.split('--')
    #  scores = np.empty((len(groups), 2))
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel('time')
    ax.set_ylabel('am_score')
    ax.grid(True)
    for g in groups:
        lines = g.strip().split('\n')
        time = float(lines[6].split(' ')[-2])
        #  print(lines[1].strip().split(' '))
        am_score = float(lines[1].split(' ')[-1])
        thres = lines[0].split(' ')[-1]
        #  scores[i] = am_score, time
        ax.plot(time, am_score, marker='o')
        ax.text(time, am_score, thres)
    ax.text(0, 0, "test")
    plt.show()

if __name__ == '__main__':
    _main()
