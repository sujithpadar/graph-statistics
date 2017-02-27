from igraph import *
import time
from pprint import pprint
import numpy as np
import pickle
from glob import glob
import sys

def main(pickleloc):
    '''

    :param pickleloc: pickle file location of lwcc lscc graphs
    :return:
    '''

    def metrics(bins):
        '''

        :param bins: bins of the histogram where x axis is the distances..1 to n
        :return: Returns, mean of the histogram, median, diameter, 90% diameter in a dictionary
        '''

        # Calculating Mean
        sum = 0
        for i in range(0, np.size(bins)):
            sum += (i + 1) * bins[i]
        mean = sum / np.sum(bins)

        # caculating diameter
        dia = np.size(bins)

        # calculating Median
        median_index = np.sum(bins) / 2
        for i in range(0, np.size(bins)):
            median_index -= bins[i]
            if median_index <= 0:
                break

        median = i + 1

        # caclating effective diameter

        eff_dia_index = np.sum(bins) * 0.9
        for i in range(0, np.size(bins)):
            eff_dia_index -= bins[i]
            if eff_dia_index <= 0:
                break
        eff_dia = i + (bins[i] + eff_dia_index) / bins[i]

        stats = {'mean': mean,
                 'median': median,
                 'dia': dia,
                 'eff_dia': eff_dia
                 }

        return stats

    pciklefnames = glob(pickleloc + "/" + pickleloc.split("/")[-1].split(".")[0].split("-Analysis")[0] + "*.pickle")

    with open(pciklefnames[0], 'rb') as handle:
        lscc = pickle.load(handle)['glscc']

    with open(pciklefnames[1], 'rb') as handle:
        lwcc = pickle.load(handle)['glwcc']

    lsccdstatsfname = pickleloc + '/lsccc-exact-dist-summ.pickle'
    lwccdstatsfname = pickleloc + '/lwccc-exact-dist-summ.pickle'
    logfile = pickleloc + '/analysis-exact-dist-summ.txt'

    t0 = time.time()
    hist = lscc.path_length_hist(directed=True)
    bins = hist.__dict__['_bins']
    lsccdstats = metrics(bins)
    t1 = time.time()

    t2 = time.time()
    hist = lwcc.path_length_hist(directed=False)
    t3 = time.time()
    bins = hist.__dict__['_bins']
    lwccdstats = metrics(bins)

    with open(lsccdstatsfname, 'wb') as handle:
        pickle.dump(lsccdstats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(lwccdstatsfname, 'wb') as handle:
        pickle.dump(lwccdstats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    printfile = open(logfile, 'w+')
    print("Largest Strongly Connected Component:", file=printfile)
    print("Computed in " + str(round((t1 - t0) / 60, 3)) + "mins", file=printfile)
    pprint(lsccdstats, printfile)
    print("---------------------------------------------" + '\n', file=printfile)

    print("Largest Weakly Connected Component:", file=printfile)
    print("Computed in " + str(round((t3 - t2) / 60, 3)) + "mins", file=printfile)
    pprint(lwccdstats, printfile)
    print("---------------------------------------------" + "\n", file=printfile)

    printfile.close()


if __name__ == "__main__":
    main(sys.argv[1])