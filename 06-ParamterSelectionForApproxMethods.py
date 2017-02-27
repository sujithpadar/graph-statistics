from igraph import *
import pickle
import pandas as pd
import numpy as np
import random
from collections import Counter
from glob import glob
import time

def main(pickleloc):

    def ParameterIterator(g, undirected= False):

        def SampleRandomPairApproximationDistanceStats(g, sampleprop=0.01):
            def getRandomPairs(onodes, psample):
                from random import randint, seed
                seed(1234)

                nmax = onodes.shape[0]

                def gencoordinates(m, n):
                    seen = set()
                    x, y = randint(m, n), randint(m, n)
                    while True:
                        seen.add((x, y))
                        yield (x, y)
                        x, y = randint(m, n), randint(m, n)
                        while (x, y) in seen:
                            x, y = randint(m, n), randint(m, n)

                sindices = []
                for n in range(0, int(round(nmax * psample))):
                    sindices.append(next(gencoordinates(0, nmax - 1)))

                snodes = [(originalnodes[x], originalnodes[y]) for x, y in sindices]
                return snodes

            # get distances for samplenodes
            def getPairDistance(nodeset, graph):

                counts = dict()

                for idx in np.arange(0, len(nodeset)):
                    dist = graph.shortest_paths(samplenodes[idx][0], samplenodes[idx][1])[0][0]
                    counts[dist] = counts.get(dist, 0) + 1

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
                i = 0
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
            samplenodes = getRandomPairs(originalnodes, sampleprop)

            # get sample distance counts
            sampledist = getPairDistance(samplenodes, g)

            # generate dictionary of distance statistics
            distsumm = getDistanceMetrics(list(sampledist.values()))

            return distsumm

        def SampleRandomSourceApproximationDistanceStats(g, sampleprop=0.01):

            def getRandomPairs(onodes, psample):
                snodes = [onodes[i] for i in random.sample(range(onodes.shape[0]), int(round(onodes.shape[0] * psample)))]
                return snodes

            # get distances for samplenodes
            def getPairDistance(nodeset, graph, originalnodes):

                counts = dict()

                for idx in np.arange(0, len(nodeset)):
                    distall = graph.shortest_paths(samplenodes[idx],
                                                   [item for item in originalnodes if item not in samplenodes[idx]])[0]
                    distcounter = Counter(distall)
                    for d, f in distcounter.items():
                        counts[d] = counts.get(d, 0) + f

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
                i = 0
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
            samplenodes = getRandomPairs(originalnodes, sampleprop)

            # get sample distance counts
            sampledist = getPairDistance(samplenodes, g, originalnodes)

            # generate dictionary of distance statistics
            distsumm = getDistanceMetrics(list(sampledist.values()))

            return distsumm

        def ApproximateNeighbour(g, r=5, K=30, undirected=False):
            '''
            :param g: graph object igraph
            :param r: ANF parameter - set between 5 to 10
            :param K: ANF number of samples - higher the number more the approximation accuracy
            :return: graph distance statistics
            '''

            # data elements required
            # 2xNxK matrix for each iteration and previous iteration
            # a dictionary to save respective value of h for the iteration
            # a bitwise hashing function
            # a list of nodes
            # a list of edges

            # algorithm
            # for every node in the list generate k bitmasks : write a function to return a numpy matrix of bitmasks
            # save indexes of these nodes on numpy array
            # traverse edge list - for each tuple, add the bitmasks
            # summarise current iteration and update the summary dictionary


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

                # calculating diameter
                dia = np.size(bins)

                # calculating Median
                median_index = np.sum(bins) / 2
                i = 0
                for i in range(0, np.size(bins)):
                    median_index -= bins[i]
                    if median_index <= 0:
                        break

                median = i + 1

                # calculating effective diameter

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

            def getInitialBitmaps(nlist, r, K):
                '''

                :param nlist: list of nodes
                :param r: r for ANF algorithm
                :param K: K for ANF algorithm
                :return: anfmat : a 2xNxK matrix with ndarray[0] containing NxK bitmap initializations
                '''
                bmlength = np.int(np.ceil(np.log2(nlist.shape[0])) + r)
                anfmat = np.zeros(shape=(2, nlist.shape[0], K))

                for n in np.arange(nlist.shape[0]):
                    for k in np.arange(K):
                        bchar = '0' * bmlength

                        for i in np.arange(0, bmlength):
                            if (np.random.uniform() <= (1 / pow(2, i + 1))):
                                blist = list(bchar)
                                blist[i] = '1'
                                bchar = ''.join(blist)
                                # break

                        anfmat[0][n][k] = int(bchar, 2)
                anfmat = anfmat.astype(int)
                return anfmat

            # def bprint(mat,bmlength):
            #     for m in np.arange(mat.shape[0]):
            #         print([(format(mm,'0'+str(bmlength)+'b')) for mm in mat[m]])

            nlist = np.array(g.vs['name'])
            bmlength = np.int(np.ceil(np.log2(nlist.shape[0])) + r)

            elist = np.array([e.tuple for e in g.es])

            anfmat = getInitialBitmaps(nlist, r, K)

            # initialize
            anfmat[1] = anfmat[0]
            for e in np.arange(elist.shape[0]):
                anfmat[1][elist[e][0]] = np.bitwise_or(anfmat[1][elist[e][0]], anfmat[0][elist[e][1]])
                if undirected:
                    anfmat[1][elist[e][1]] = np.bitwise_or(anfmat[1][elist[e][1]], anfmat[0][elist[e][0]])

            # compute stats
            counts = dict()
            d = 0
            for n in np.arange(nlist.shape[0]):
                b = np.array([bin(bn).find('0', 1) - 2 for bn in anfmat[0][n]], dtype=int)
                b[np.where(b == -3)] = bmlength
                counts[d] = counts.get(d, 0) + (pow(2, b.mean()) / .77351)

            d = 1
            for n in np.arange(nlist.shape[0]):
                b = np.array([bin(bn).find('0', 1) - 2 for bn in anfmat[1][n]], dtype=int)
                b[np.where(b == -3)] = bmlength
                counts[d] = counts.get(d, 0) + (pow(2, b.mean()) / .77351)

            counts[1] = counts[1] - counts[0]

            while counts[len(counts) - 1] > 1:

                d += 1

                # initialize
                anfmat[0] = anfmat[1]
                for e in np.arange(elist.shape[0]):
                    anfmat[1][elist[e][0]] = np.bitwise_or(anfmat[1][elist[e][0]], anfmat[0][elist[e][1]])
                    if undirected:
                        anfmat[1][elist[e][1]] = np.bitwise_or(anfmat[1][elist[e][1]], anfmat[0][elist[e][0]])

                # compute stats
                for n in np.arange(nlist.shape[0]):
                    b = np.array([bin(bn).find('0', 1) - 2 for bn in anfmat[1][n]], dtype=int)
                    b[np.where(b == -3)] = bmlength
                    counts[d] = counts.get(d, 0) + (pow(2, b.mean()) / .77351)

                # remove cumulative counts
                for i in np.arange(0, d):
                    counts[d] = counts[d] - counts[i]

            # remove end elements
            del counts[len(counts) - 1]
            del counts[0]

            distsumm = getDistanceMetrics(list(counts.values()))

            return distsumm

        # create a data frame for different values of parameters
        psrpa = pd.DataFrame()
        psrsa = pd.DataFrame()
        panf = pd.DataFrame()

        for sp in np.linspace(0.01, 0.36, 6):
            t0 = time.time()
            ssrpa = SampleRandomPairApproximationDistanceStats(g, sp)
            t1 = time.time()
            ssrsa = SampleRandomSourceApproximationDistanceStats(g, sp)
            t2 = time.time()
            ssrpa['param'] = sp
            ssrsa['param'] = sp
            ssrpa['time'] =  str(round((t1 - t0) / 60, 3))
            ssrsa['time'] =  str(round((t2 - t1) / 60, 3))
            psrpa = psrpa.append(ssrpa,ignore_index=True )
            psrsa = psrsa.append(ssrsa,ignore_index=True )


        for k in np.linspace(5, 155, 6):
            t3 = time.time()
            sanf = ApproximateNeighbour(g, r=5, K=int(k), undirected=undirected)
            t4 = time.time()
            sanf['param'] = int(k)
            sanf['time'] = str(round((t4 - t3) / 60, 3))
            panf = panf.append(sanf,ignore_index=True)

        return psrpa, psrsa, panf

    # pickleloc = '/media/Documents/01 Aalto/03 Study/Semester 01/05 Algorithmic Methods of Data Mining - Aristides Gionis/Project/rawdata/01-wiki-vote/Wiki-Vote20161207-15-00'
    pciklefnames = glob(pickleloc + "/" + pickleloc.split("/")[-1].split(".")[0].split("-Analysis")[0] + "*.pickle")

    with open(pciklefnames[0], 'rb') as handle:
        lscc = pickle.load(handle)['glscc']

    with open(pciklefnames[1], 'rb') as handle:
        lwcc = pickle.load(handle)['glwcc']

    lsccpsrpastatsfname = pickleloc + '/analysis-param-lscc-srspa-summ.csv'
    lsccpsrsastatsfname = pickleloc + '/analysis-param-lscc-srsa-summ.csv'
    lsccpanfstatsfname = pickleloc + '/analysis-param-lscc-anf-summ.csv'

    lsccpsrpastats, lsccpsrsastats, lsccpanfstats = ParameterIterator(lscc)

    lsccpsrpastats.to_csv(lsccpsrpastatsfname,index=False)
    lsccpsrsastats.to_csv(lsccpsrsastatsfname,index=False)
    lsccpanfstats.to_csv(lsccpanfstatsfname,index=False)

    lwccpsrpastatsfname = pickleloc + '/analysis-param-lwcc-srspa-summ.csv'
    lwccpsrsastatsfname = pickleloc + '/analysis-param-lwcc-srsa-summ.csv'
    lwccpanfstatsfname = pickleloc + '/analysis-param-lwcc-anf-summ.csv'

    lwccpsrpastats, lwccpsrsastats, lwccpanfstats = ParameterIterator(lwcc,undirected=True)

    lwccpsrpastats.to_csv(lwccpsrpastatsfname,index=False)
    lwccpsrsastats.to_csv(lwccpsrsastatsfname,index=False)
    lwccpanfstats.to_csv(lwccpanfstatsfname,index=False)

if __name__ == "__main__":
    main(sys.argv[1])

