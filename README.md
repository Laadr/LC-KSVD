# Label Consistent KSVD algorithm (LC-KSVD)

## Description
Implementation of the Label consistent KSVD algorithm proposed by Zhuolin Jiang, Zhe Lin and Larry S. Davis.

This implementation is a translation of the matlab code released by the authors on /projectlcksvd.html.

The code has been extended in order to use the related method called Discriminative KSVD proposed by Zhang, Qiang and Li, Baoxin.

## Usage
Class LCKSVD() includes the following methods:

    initialization4LCKSVD(training_feats,H_train,dictsize,iterations,sparsitythres,tol=1e-4):

    Initialization for Label consistent KSVD algorithm
    Inputs
          training_feats  -training features
          H_train         -label matrix for training feature 
          dictsize        -number of dictionary items
          iterations      -iterations
          sparsitythres   -sparsity threshold
          tol             -tolerance when performing the approximate KSVD
    Outputs
          Dinit           -initialized dictionary
          Tinit           -initialized linear transform matrix
          Winit           -initialized classifier parameters
          Q               -optimal code matrix for training features 
.

    initialization4DKSVD(training_feats,labels,dictsize,iterations,sparsitythres,Dinit=None,tol=1e-4):

    Initialization for Discriminative KSVD algorithm
    Inputs
          training_feats  -training features
          labels          -label matrix for training feature (numberred from 1 to nb of classes)
          dictsize        -number of dictionary items
          iterations      -iterations
          sparsitythres   -sparsity threshold
          Dinit           -initial guess for dictionary
          tol             -tolerance when performing the approximate KSVD
    Outputs
          Dinit           -initialized dictionary
          Winit           -initialized classifier parameters
.
    
    labelconsistentksvd(Y,Dinit,labels,Winit,iterations,sparsitythres,sqrt_beta,sqrt_alpha=0.,Q_train=None,Tinit=None, tol=1e-4):

    Label consistent KSVD algorithm
    Inputs
          Y               -training features
          Dinit           -initialized dictionary
          labels          -labels matrix for training feature (numberred from 1 to nb of classes)
          Winit           -initialized classifier parameters
          iterations      -iterations for KSVD
          sparsitythres   -sparsity threshold for KSVD
          sqrt_beta       -contribution factor
          sqrt_alpha      -contribution factor (0. for D-KSVD)
          Q_train         -optimal code matrix for training feature (use only for LC-KSVD)
          Tinit           -initialized transform matrix (use only for LC-KSVD)
    Outputs
          D               -learned dictionary
          X               -sparsed codes
          T               -learned transform matrix
          W               -learned classifier parameters
.
    
    classification(D, W, data, sparsity):

    Classification 
    Inputs
          D               -learned dictionary
          W               -learned classifier parameters
          data            -data to classify
          sparsity        -sparsity threshold
    outputs
          prediction      -predicted classification vectors. Perform sp.argmax(W.dot(gamma), axis=0) to get labels
          gamma           -learned representation
    
!! This code has not been extensively tested. If an error occures, feel free to open an issue and I will try to correct the code rapidly.

# Requirements

Scipy and scikit-learn need to be installed.

# Authors

The approximate KSVD algorithm included in the code has been written by nel215 (https://github.com/nel215/ksvd) (and very slighty modified).

Software translated and extended from matlab (http://users.umiacs.umd.edu/~zhuolin) to python by Adrien Lagrange (ad.lagrange@gmail.com), 2018.

# License

Distributed under the terms of the GNU General Public License 2.0.
