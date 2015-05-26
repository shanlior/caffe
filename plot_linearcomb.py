#!/usr/bin/python

import sys
import re
import os
import shutil
import commands
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D

def extract_linearcomb(filename):
  f=open(filename,'r')
  openFile = f.read()
  tuples = re.findall(r'Iteration #(\d*) for alpha = (\-?\d*.\d*), Test Loss = (\d*.\d*) and Train Loss = (\d*.\d*) and SmallTrain Loss = (\d*.\d*)', openFile)
  iterations = []
  alpha = []
  trainLoss = []
  testLoss = []
  smallTrainLoss = []
  for tuple in tuples:
    #if (float(tuple[1])) not in alpha:
     # print tuple
      iterations.append(float(tuple[0]))
      alpha.append(float(tuple[1]))
      testLoss.append(float(tuple[2]))
      trainLoss.append(float(tuple[3]))
      smallTrainLoss.append(float(tuple[4]))
  #fig = plt.figure(1)
  #ax = fig.add_subplot(211, projection='3d')
  #plt.title('Test Loss')
  #plt.xlabel('iteration')
  #plt.ylabel('alpha')
  #ax.plot_trisurf(iterations, alpha, (testLoss), cmap=cm.gist_rainbow, linewidth=0.2)
  #ax = fig.add_subplot(212, projection='3d')
  #plt.title('Train Loss')
  #plt.xlabel('iteration')
  #plt.ylabel('alpha')
  #ax.plot_trisurf(iterations, alpha, (trainLoss), cmap=cm.gist_rainbow, linewidth=0.2)

#  fig2 = plt.figure(2)
  accum = 0
  accumEstimate = 0
  maxIter = 1
  for i in range(0,maxIter):
	  iterNum = 8
	# for stepsize = 0.5
	  iterSize = 10
	# for stepsize = 0.1
	#  iterSize = 91
	  begin = iterNum * iterSize
	  end = (iterNum+1) * iterSize
	  coeff = np.polyfit(alpha[begin:end], trainLoss[begin:end], 2, rcond=None, full=False, w=None, cov=False)
	  smallcoeff = np.polyfit(alpha[begin:end], smallTrainLoss[begin:end], 2, rcond=None, full=False, w=None, cov=False)
	  testcoeff = np.polyfit(alpha[begin:end], testLoss[begin:end], 2, rcond=None, full=False, w=None, cov=False)
	  parabola = []
	  smallparabola = []
	  for idx in range(begin,end):
	    parabola.append(coeff[2]+coeff[1]*alpha[idx]+coeff[0]*alpha[idx]*alpha[idx])
	    smallparabola.append(smallcoeff[2]+smallcoeff[1]*alpha[idx]+smallcoeff[0]*alpha[idx]*alpha[idx])
	  plt.plot(alpha[begin:end], testLoss[begin:end],alpha[begin:end], trainLoss[begin:end],alpha[begin:end], parabola, alpha[begin:end], smallTrainLoss[begin:end],
	alpha[begin:end], smallparabola)
#	  print 'Parabola minimum: ' + str(-coeff[1]/(2*coeff[0]))
#	  print 'Estimated minimum: ' + str(-smallcoeff[1]/(2*smallcoeff[0]))
#	  print 'Test minimum: ' + str(-testcoeff[1]/(2*testcoeff[0]))
#	  print 'Solution diff: ' + str(np.abs(-testcoeff[1]/(2*testcoeff[0])+coeff[1]/(2*coeff[0])))
#	  print 'Estimate diff: ' + str(np.abs(-testcoeff[1]/(2*testcoeff[0])+smallcoeff[1]/(2*smallcoeff[0])))
          accum = accum + (np.abs(-testcoeff[1]/(2*testcoeff[0])+coeff[1]/(2*coeff[0])))
          accumEstimate = accumEstimate + (np.abs(-testcoeff[1]/(2*testcoeff[0])+smallcoeff[1]/(2*smallcoeff[0])))
	  plt.title('Loss vs. alpha')
	  plt.xlabel('alpha')
	  plt.ylabel('Loss')
	  plt.legend(['Test Loss', 'Train Loss','Parabola of Train Loss','smallTrainLoss','small Parabola'],loc='upper right', bbox_to_anchor=(1.0, 1.05),
		    fancybox=True, shadow=True)
	  plt.show()
#  print 'Accumulative: ' + str(accum)
#  print 'Estimated: ' + str(accumEstimate)
#  print 'Average Error: ' + str((accumEstimate - accum)/maxIter)

def main():
  # This command-line parsing code is provided.
  # Make a list of command line arguments, omitting the [0] element
  # which is the script itself.
  args = sys.argv[1:]

  if not args:
    print 'usage: ' + sys.argv[0] + ' <File1> <File2> <...>'
    sys.exit(1)


  for filename in args:
    extract_linearcomb(filename)
 
  
if __name__ == '__main__':
  main()
