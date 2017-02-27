from glob import glob
import pickle
import time
import numpy as np
from pprint import pprint
import sys
import random
from collections import Counter

def main(pickleloc):
    # input should be pickle folder name
    # load pickle files from there
    # on those graph pick a random sample of nodes
    # compute the pair wise distance from those nodes
    # compute statistics on those distances
    # save it in a text file

    def SampleRandomSourceApproximationDistanceStats(g):

        def getRandomPairs(onodes, psample):
            snodes = [onodes[i] for i in random.sample(range(onodes.shape[0]),round(onodes.shape[0] * psample))]
            return snodes

        # get distances for samplenodes
        def getPairDistance(nodeset,graph,originalnodes):

            counts = dict()

            for idx in np.arange(0,len(nodeset)):
                distall = graph.shortest_paths(samplenodes[idx],[item for item in originalnodes if item not in samplenodes[idx]])[0]
                distcounter = Counter(distall)
                for d,f in distcounter.items():
                    counts[d] = counts.get(d,0)+f

            return counts

        def getDistanceMetrics(bins):
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

        originalnodes = np.array(g.vs['name'])

        # pick random pairs
        samplenodes = getRandomPairs(originalnodes,0.01)

        # get sample distance counts
        sampledist = getPairDistance(samplenodes,g,originalnodes)

        # generate dictionary of distance statistics
        distsumm =  getDistanceMetrics(list(sampledist.values()))

        return distsumm


    # pickleloc = '/media/Documents/01 Aalto/03 Study/Semester 01/05 Algorithmic Methods of Data Mining - Aristides Gionis/Project/rawdata/01-wiki-vote/Wiki-Vote20161207-15-00'
    pciklefnames = glob(pickleloc + "/"+pickleloc.split("/")[-1].split(".")[0].split("-Analysis")[0]+"*.pickle")

    with open(pciklefnames[0], 'rb') as handle:
        lscc = pickle.load(handle)['glscc']

    with open(pciklefnames[1], 'rb') as handle:
        lwcc = pickle.load(handle)['glwcc']

    lsccdstatsfname = pickleloc+'/lsccc-srsa-dist-summ.pickle'
    lwccdstatsfname = pickleloc+'/lwccc-srsa-dist-summ.pickle'
    logfile = pickleloc+'/analysis-srsa-dist-summ.txt'

    t0 = time.time()
    lsccdstats = SampleRandomSourceApproximationDistanceStats(lscc)
    t1 = time.time()

    t2 = time.time()
    lwccdstats = SampleRandomSourceApproximationDistanceStats(lwcc)
    t3 = time.time()

    with open(lsccdstatsfname, 'wb') as handle:
        pickle.dump(lsccdstats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(lwccdstatsfname, 'wb') as handle:
        pickle.dump(lwccdstats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    printfile = open(logfile, 'w+')
    print("Largest Strongly Connected Component:", file=printfile)
    print("Computed in " + str(round((t1 - t0) / 60, 3)) + "mins", file=printfile)
    pprint(lsccdstats, printfile)
    print("---------------------------------------------" + "\n", file=printfile)

    print("Largest Weakly Connected Component:", file=printfile)
    print("Computed in " + str(round((t3 - t2) / 60, 3)) + "mins", file=printfile)
    pprint(lwccdstats, printfile)
    print("---------------------------------------------" + "\n", file=printfile)

    printfile.close()

if __name__ == "__main__":
    main(sys.argv[1])
