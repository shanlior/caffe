#!/usr/bin/env sh

TOOLS=./build/tools


$TOOLS/caffe train \
  --solver=examples/cifar10/cifar10_full_solver.prototxt --solver2=examples/cifar10/cifar10_full_solver_2.prototxt --solverComb=examples/cifar10/cifar10_full_solver_comb.prototxt --start_iteration=500 2> examples/cifar10/dual_solver_ELI_500_10000data.log




