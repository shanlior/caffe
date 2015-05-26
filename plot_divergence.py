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


"""Extract contraction from log file
   usage: [--save dir] [--noshow] [--average numofsamples] file [file ...]
   The save options - save the plots to the mentioned dir
   The noshow option disables the automatic opening of the figure
   the save flag must be on inorder to use the diff function
"""

def numLayers(filename):
  f=open(filename,'r')
  openFile = f.read()
  ls = re.findall(r'Layer #(\d*) Accumulating', openFile) 
  maxLayer = 0
  for num in ls:
    if int(num) > maxLayer:
      maxLayer = int(num)
  return maxLayer+1

def extract_accumulation(filename):
  """
  Given a file name for logFile, returns a list of tupples
  that contain (Layer#,Accumluation)
  """
  f=open(filename,'r')
  openFile = f.read()
  tuples = re.findall(r'Iteration #(\d*), Layer #(\d*) Accumulating (\d*.\d*)', openFile) 
  val_out = []
  iter_out = []
  numLayer = numLayers(filename)
  for i in range(0,numLayer):
    val_list = []
    iter_list = []
    for tuple in tuples:    
      if int(tuple[1]) == i:
        val_list.append(tuple[2])
        iter_list.append(tuple[0])
    val_out.append(val_list)
    iter_out.append(iter_list)
  return iter_out, val_out

def LayerNames(filename):
  f=open(filename,'r')
  openFile = f.read()
  tuples = re.findall(r'Layer #(\d*): (\w.*)', openFile) 
  unique = []
  [unique.append(tuple) for tuple in tuples if tuple not in unique]
  names = []
  for tuple in unique:
    names.append(tuple[1])
  return names
 
def extract_accuracy(filename):
  f=open(filename,'r')
  openFile = f.read()
  iteration = re.findall(r'Iteration (\d*), Testing net', openFile)
  ls_iter = []
  for iter in iteration:
    ls_iter.append(iter)

  tuples = re.findall(r'Test net output #0: accuracy = (\d*.\d*)', openFile) 
  ls = []
  for tuple in tuples:    
    ls.append(tuple)
  out = []
  for i in range(len(ls)):
    out.append((ls_iter[i],ls[i]))
  return out 

def extract_accuracy_merge(filename):
  f=open(filename,'r')
  openFile = f.read()
  iteration = re.findall(r'Iteration (\d*), Testing net.*\n.*Test net output #0 of Solver2Before', openFile)
  ls_iter = []
  for iter in iteration:
    ls_iter.append(iter)

  Solver1Before = re.findall(r'Test net output #0 of Solver1Before: accuracy = (\d*.\d*)', openFile) 
  Solver2Before = re.findall(r'Test net output #0 of Solver2Before: accuracy = (\d*.\d*)', openFile) 
  AfterMerge = re.findall(r'Test net output #0 of AfterMerge: accuracy = (\d*.\d*)', openFile) 
  x = []
  y = []
  z = []
  for tuple in Solver1Before:    
    x.append(tuple)
  for tuple in Solver2Before:    
    y.append(tuple)
  for tuple in AfterMerge:    
    z.append(tuple)
  out = []
  for i in range(len(ls_iter)):
    out.append((ls_iter[i],x[i],y[i],z[i]))
  return out 

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    y = []
    for inter in interval:
      y.append(float(inter))
    return np.convolve(y, window, 'same')

def main():
  # This command-line parsing code is provided.
  # Make a list of command line arguments, omitting the [0] element
  # which is the script itself.
  args = sys.argv[1:]

  if not args:
    print 'usage: [--save dir] [--noshow] [--average numofsamples] file [file ...]'
    sys.exit(1)

  # Notice the summary flag and remove it from args if it is present.
  save = False
  show = True
  average = False
  if args[0] == '--save':
    save = True
    dir = args[1]
    del args[0:2]
  if args[0] == '--noshow':
    show = False
    del args[0]
  if args[0] == '--average':
    average = True
    windowsize = args[1]
    del args[0:2]
  filenames = []
  legendnums = []
  legendnames = []
  for name in args:
    if name.endswith('.log'):
      filenames.append(os.path.splitext(name)[0])
    if (len(filenames) > 0):
      tmp = re.search('(\d\d*)', filenames[-1])
      legendnums.append(str(tmp.group(1))) 
      tmp = re.findall('_([a-zA-Z]*)_\d*',filenames[-1])
      legendnames.append(str(tmp[-1]))
  numLayer = numLayers(args[0])
  names = LayerNames(args[0])

  vector={}
  iteration={}
  for i in range(0,numLayer):
    plt.figure(i+1)
    for filename in args:
      if i==0:
        iteration[filename], vector[filename] = extract_accumulation(filename)    
      if average:
        y_av = movingaverage(vector[filename][i],windowsize)
        plt.plot(iteration[filename][i],y_av)
      else:
        plt.plot(iteration[filename][i],vector[filename][i])
    plt.title(names[i],fontsize=16,fontweight='bold')
    plt.xlabel('Iteration',fontsize=14)
    plt.xlim(xmin=0)
    plt.subplots_adjust(left=0.21)
    plt.ylabel(r'$\frac{2\sqrt{\sum_i^{N_i} |w_{1}^{(i)}-w_{2}^{(i)}|^2}}{\sqrt{\sum_i^{N_i} |w_{1}^{(i)}|^2}+\sqrt{\sum_i^{N_i} |w_{2}^{(i)}|^2}}$',fontsize=14,fontweight='bold')
    fontP = FontProperties()
    #fontP.set_size('medium')
    if (len(legendnums)>1):
      if (legendnums[0]==legendnums[1]):
        plt.legend(legendnames,loc='upper left',
           prop = fontP)
      else:
        plt.legend(legendnums,loc='upper left',
           prop = fontP)
    if save:
      if not os.path.isdir(dir): os.makedirs(dir)
      plt.savefig(dir + '/' + names[i] + '.png')
  plt.figure(numLayer+1)
  colors = iter(cm.rainbow(np.linspace(0, 1, len(args))))
  for filename in args:
#    accuracy = extract_accuracy(filename)
#    [t,z] = zip(*accuracy)
    accuracy = extract_accuracy_merge(filename)
    [t,x,y,z] = zip(*accuracy)
    plt.scatter(t,z,color=next(colors))
  plt.title('Accuracy vs. Iteration')
  plt.xlabel('Iteration')
  plt.xlim(xmin=0)
  plt.yticks(np.arange(0, 1.05, 0.05))
  plt.grid()
  plt.ylabel('Accuracy')
  plt.ylim(ymax=1)
  if (len(legendnums)>1):
    if (legendnums[0]==legendnums[1]):
      plt.legend(legendnames,loc='upper left',
         prop = fontP)
    else:
      plt.legend(legendnums,loc='upper left',
         prop = fontP)
  if save:
    if not os.path.isdir(dir): os.makedirs(dir)
    plt.savefig(dir + '/accuracy.png')
  if show: plt.show()   
  
if __name__ == '__main__':
  main()
