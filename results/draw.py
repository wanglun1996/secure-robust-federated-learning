import os
import sys
import csv
import numpy as np
import argparse
import matplotlib

matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
agg = ['average', 'trimmedmean', 'krum', 'filterl2']
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('attack')
    parser.add_argument('--lower', type=float, default=0)
    parser.add_argument('--upper', type=float, default=30)

    args = parser.parse_args()
    
    txt_location = './' + args.attack + '_%s_' + args.dataset + '.txt'
    for agg_method in agg:
        
