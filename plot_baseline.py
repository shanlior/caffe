#!/usr/bin/python

import sys
import re
import os
import shutil
import commands
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D


"""Extract contraction from log file
   usage: [--save dir] [--noshow] [-diff] file [file ...]
   The save options - save the plots to the mentioned dir
   The noshow option disables the automatic opening of the figure
   the save flag must be on inorder to use the diff function
"""

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

def extract_loss(filename):
  f=open(filename,'r')
  openFile = f.read()
  iteration = re.findall(r'Iteration (\d*), Testing net', openFile)
  ls_iter = []
  for iter in iteration:
    ls_iter.append(iter)
  tuples = re.findall(r'Test net output #1: loss = (\d*.\d*)', openFile) 
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
  iteration = re.findall(r'Iteration (\d*), Testing net.*\n.*Test net output #0 of AfterMerge', openFile)
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
      out.append((ls_iter[i],z[i]))
  return out 



def extract_loss_merge(filename):
  f=open(filename,'r')
  openFile = f.read()
  iteration = re.findall(r'Iteration (\d*), Testing net.*\n.*Test net output #0 of AfterMerge', openFile)
  ls_iter = []
  for iter in iteration:
    ls_iter.append(iter)

  Solver1Before = re.findall(r'Test net output #1 of Solver1Before: loss = (\d*.\d*)', openFile) 
  Solver2Before = re.findall(r'Test net output #1 of Solver2Before: loss = (\d*.\d*)', openFile) 
  AfterMerge = re.findall(r'Test net output #1 of AfterMerge: loss = (\d*.\d*)', openFile)
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
      out.append((ls_iter[i],z[i]))
  return out 

def extract_LR_merge(filename):
  f=open(filename,'r')
  openFile = f.read()
  out = re.findall(r'Iteration (\d*), lr = (\d*.\d*\s)', openFile)
  out2 = re.findall(r'Iteration (\d*), lr = ([-+]?[0-9]*\.?[0-9]+[eE][-+]?\d*)', openFile)
  return out + out2
    
def main():
  # This command-line parsing code is provided.
  # Make a list of command line arguments, omitting the [0] element
  # which is the script itself.
  args = sys.argv[1:]

  if not args:
    print 'usage: [file ...]'
    sys.exit(1)

  # Notice the summary flag and remove it from args if it is present.

  filenames = []
  legendnums = []
  legendnames = []
  for name in args:
    if name.endswith('.log'):
      filenames.append(os.path.splitext(name)[0])
    if (len(filenames) > 0):
      tmp = re.search('(\d\d*)', filenames[-1])
      #legendnums.append(str(tmp.group(1))) 
      tmp = re.findall('([a-zA-Z]*_[a-zA-Z]*_\d*)',filenames[-1])
      #legendnames.append(str(tmp[-1]))

  colors = iter(cm.rainbow(np.linspace(0, 1, len(args))))
  plt.figure(1)
  for filename in args:
    if filename.endswith('baseLine.log'):
      accuracy = extract_accuracy(filename)
      [t,z] = zip(*accuracy)
    else:
      accuracy = extract_accuracy_merge(filename)
      [t,z] = zip(*accuracy)
    plt.scatter(t,z,color=next(colors))
  plt.title('Accuracy vs. Iteration')
  plt.xlabel('Iteration')
  plt.xlim(xmin=0)
  plt.yticks(np.arange(0, 1.05, 0.05))
  plt.grid()
  plt.ylabel('Accuracy')
  plt.ylim(ymax=1)
  legendnames = ['Orthogonal adaptive','Baseline - Cifar Full','Shuffled - 0.1 step size','with Adagrad']
#  legendnames = ['1000 images FM+LS','50000 images FM+LS','Baseline','5000 images FM','5000 images FM+LS','50000 images, LS 1000']
  legendnames = ['LS - every 2000 iterations','LS - every 1000 iterations','LS - every 300 iterations','LS - every 100 iterations','Baseline']
  plt.legend(legendnames,loc='lower right')
  colors = iter(cm.rainbow(np.linspace(0, 1, len(args))))
  plt.figure(2)
  for filename in args:
    if filename.endswith('baseLine.log'):
      loss = extract_loss(filename)
      [t,z] = zip(*loss)
    else:
      loss = extract_loss_merge(filename)
      [t,z] = zip(*loss)
    z1=[]
    for i in range(len(z)):
      z1.append(np.log(float(z[i])))
    plt.scatter(t,z1,color=next(colors))   
  plt.title('Log Test Loss vs. Iterations')
  plt.xlabel('Iteration')
  plt.ylabel('Log Loss')
  plt.legend(legendnames,loc='upper right')
  plt.figure(3)
  colors = iter(cm.rainbow(np.linspace(0, 1, len(args))))
  for filename in args:
    tmp = extract_LR_merge(filename)
    [it,rate] = zip(*tmp)
    plt.scatter(it,rate,color=next(colors))
  plt.title('Learning Rate')
  plt.xlabel('Iteration')
  plt.ylabel('Learning Rate')
  plt.legend(legendnames,loc='upper right')
  plt.show()
  
if __name__ == '__main__':
  main()
