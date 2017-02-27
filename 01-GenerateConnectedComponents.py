from igraph import *
import time
from pprint import pprint
import pickle
from datetime import datetime
import sys

def main( graphfname):
    '''

    :param graphfname: path of csv file with graph edgelists
    :return:
    '''

    def CreateConnectedComponents(graphloc,logfile):
        '''
        :param graphloc: location of the graph file
        :param logfile: filename for logfile
        :return: lscc and lwcc dictionaries
        '''

        printfile = open(logfile, 'w+')

        g = Graph.Read_Ncol(graphloc, names=True, directed=True)

        # Main graph
        graphsumm = {'number of edges':g.ecount(),
                     'number of vertices':g.vcount()}
        print("Main Graph:", file=printfile)
        pprint(graphsumm, printfile)
        print("---------------------------------------------"+"\n", file=printfile)

        # strongly connected component
        t0 = time.time()
        lscc = g.clusters().giant()
        t1 = time.time()

        lsccsumm = {'number of edges':lscc.ecount(),
                     'number of vertices':lscc.vcount()}

        print("Largest Strongly Connected Component:", file=printfile)
        print("Generated in "+str(round((t1-t0)/60,3))+"mins", file=printfile)
        print("Summary:", file=printfile)
        pprint(lsccsumm, printfile)
        print("---------------------------------------------" + "\n", file=printfile)


        lsccdict ={'glscc':lscc,
                   'summlscc':lsccsumm,
                   }

        # weakly connected component
        t0 = time.time()
        lwcc = g.as_undirected(mode="collapse", combine_edges=None).clusters().giant()
        t1 = time.time()

        lwccsumm = {'number of edges':lwcc.ecount(),
                     'number of vertices':lwcc.vcount()}

        print("Largest Weakly Connected Component:", file=printfile)
        print("Generated in "+str(round((t1-t0)/60,3))+"mins", file=printfile)
        print("Summary:", file=printfile)
        pprint(lwccsumm, printfile)
        print("---------------------------------------------" + "\n",file=printfile)

        printfile.close()

        lwccdict ={'glwcc':lwcc,
                   'summlwcc':lwccsumm,
                   }

        return lsccdict, lwccdict


    # create output file requirements
    graphname =  graphfname.split("/")[-1].split(".")[0]
    # dirname = "/".join(graphfname.split("/")[0:-1])+"/"+graphname+datetime.now().strftime('%Y%m%d-%H-%M')
    dirname = "/".join(graphfname.split("/")[0:-1]) + "/" + graphname + '-Analysis'
    logfname = dirname+"/"+'analysis-generation-summary'+'.txt'
    lsccpfile = dirname+"/"+graphname+'-lscc-graph'+'.pickle'
    lwccpfile = dirname+"/"+graphname+'-lwcc-graph'+'.pickle'

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # extract connected components
    lsccout, lwccout = CreateConnectedComponents( graphfname,logfname)

    with open(lsccpfile, 'wb') as handle:
        pickle.dump(lsccout, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(lwccpfile, 'wb') as handle:
        pickle.dump(lwccout, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main(sys.argv[1])

