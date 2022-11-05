#!/usr/bin/env python3

import argparse
import sys
import matplotlib.pyplot as plt
from math import ceil

def _main():
    parser = argparse.ArgumentParser(prog = 'segplot', description = 'plots linear segmentation')
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()
    width = ceil(len(args.files)**0.5)
    height = ceil(len(args.files) / width)
    fig, axs = plt.subplots(height, width)
    for i, path in enumerate(args.files):
        with open(path, 'r') as f:
            lines = f.readlines();
        plotlines = []
        plotlines.append([])
        for line in lines:
            if line == '\n':
                plotlines.append([])
                continue
            plotlines[-1].append([float(line.split(' ')[0]), float(line.split(' ')[1])])

        # make segmentation lines longer
        plotlines[1][0][1] = max([p[1] for p in plotlines[0]])
        plotlines[2][0][1] = plotlines[1][0][1]
        plotlines[1][1][1] = min([p[1] for p in plotlines[0]])
        plotlines[2][1][1] = plotlines[1][1][1]

        # draw lines
        for line in plotlines:
            X = [p[0] for p in line]
            Y = [p[1] for p in line]
            if height > 1:
                current_plt = axs[i%width, int(i/width)]
            else:
                current_plt = axs[int((i+1)/width)]
            current_plt.plot(X, Y)
            #  current_plt.ylabel('Energy')
            #  current_plt.xlabel('Time')
            current_plt.set_title(path)
    plt.show()

if __name__ == '__main__':
    _main()
