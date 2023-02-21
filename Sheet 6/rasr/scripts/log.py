#!/usr/bin/env python3

import sys
import xml.etree.ElementTree as ET

def main():
    file = sys.argv[1]
    et = ET.parse(file)
    root = et.getroot()
    corpus = root.find('corpus')
    recordings = corpus.findall('recording')
    segments = []
    for rec in recordings:
        segments += rec.findall('segment')

    correct_words = [int(seg.find('correct-words').text) for seg in segments]
    errors = [int(seg.find('evaluation').find('statistic').find('cost').text) for seg in segments]
    wer = sum(errors)/sum(correct_words)
    print(f'WER: {wer}')

    timers = [float(seg.find('timer').find('elapsed').text) for seg in segments]
    real_times = [float(seg.find('real-time').text) for seg in segments]
    rtf = sum(timers) / sum(real_times)
    print(f'RTF: {rtf}')

if __name__ == "__main__":
    main()

