## The code uses pyhton 3 environment with following dependencies installed
igraph, pandas, numpy, glob


## There are 6 .py files as listed below with a brief description
01-GenerateConnectedComponents 			- 
take an input argument of path location of the graph *.txt file, generates largest strongly & weakly connected components and saves them as .pickle files in a "*Analysis" new folder create in the same directory as that of input *.txt file

rawdata/RansomNodePairSampleApproximation 	- 
takes input argument of "*Analysis" folder location, loads the lscc & lwcc pickles from the folder and does random node pair approximation. Stores summary in the form a .txt in the same folder along with .pickle of summary dictionary objects.

03-RandomSourceApproximation 			- 
takes input argument of "*Analysis" folder location, loads the lscc & lwcc pickles from the folder and does random node source approximation. Stores summary in the form a .txt in the same folder along with .pickle of summary dictionary objects.

04-ApproximateNeighbourhoodFunction 	- 
takes input argument of "*Analysis" folder location, loads the lscc & lwcc pickles from the folder and does random node pair approximation. Stores summary in the form a .txt in the same folder along with .pickle of summary dictionary objects.

05-ExactMetricCompuation 				- 
takes input argument of "*Analysis" folder location, loads the lscc & lwcc pickles from the folder and does exact metric computation. Stores summary in the form a .txt in the same folder along with .pickle of summary dictionary objects.

06-ParamterSelectionForApproxMethods 	- 
takes input argument of "*Analysis" folder location, loads the lscc & lwcc pickles from the folder and does paramter sumamry for the approximation schemes. Stores summary in the form a .txt in the same folder along with .pickle of summary dictionary objects.


## sample codes
python 01-GenerateConnectedComponents.py '/rawdata/soc-epinions1/soc-Epinions1.txt' &
python 02-RansomNodePairSampleApproximation.py '/rawdata/soc-epinions1/soc-Epinions1-Analysis' &
python 03-RandomSourceApproximation.py '/rawdata/soc-epinions1/soc-Epinions1-Analysis' &
python 04-ApproximateNeighbourhoodFunction.py '/rawdata/soc-epinions1/soc-Epinions1-Analysis' &
python 05-ExactMetricCompuation.py '/rawdata/soc-epinions1/soc-Epinions1-Analysis' &
python 06-ParamterSelectionForApproxMethods.py '/rawdata/soc-epinions1/soc-Epinions1-Analysis' &



