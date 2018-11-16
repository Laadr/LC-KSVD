# -*- coding: utf-8 -*-
import scipy as sp
import scipy.linalg as splin
from numpy import linalg as LA
from sklearn.linear_model import orthogonal_mp_gram


class ApproximateKSVD(object):
  def __init__(self, n_components, max_iter=10, tol=1e-6, transform_n_nonzero_coefs=None):
    """
    Parameters
    ----------
    n_components:
        Number of dictionary elements
    max_iter:
        Maximum number of iterations
    tol:
        tolerance for error
    transform_n_nonzero_coefs:
        Number of nonzero coefficients to target
    """
    self.components_               = None
    self.gamma_                    = None
    self.max_iter                  = max_iter
    self.tol                       = tol
    self.n_components              = n_components
    self.transform_n_nonzero_coefs = transform_n_nonzero_coefs

  def _update_dict(self, X, D, gamma):
    for j in range(self.n_components):
      I = gamma[j,:] > 0
      if sp.sum(I) == 0:
        continue

      D[:,j] = 0
      g = gamma[j,I]
      r = X[:,I] - D.dot(gamma[:,I])
      d = r.dot(g)
      d /= splin.norm(d)
      g = r.T.dot(d)
      D[:,j] = d
      gamma[j,I] = g.T
    return D, gamma

  def _initialize(self, X):
    if min(X.shape) < self.n_components:
      D = sp.random.randn(X.shape[0],self.n_components)
    else:
      u, s, vt = sp.sparse.linalg.svds(X, k=self.n_components)
      D = sp.dot(u,sp.diag(s))
    D /= splin.norm(D, axis=0)[sp.newaxis,:]
    return D

  def _transform(self, D, X):
    gram = D.T.dot(D)
    Xy = D.T.dot(X)

    n_nonzero_coefs = self.transform_n_nonzero_coefs
    if n_nonzero_coefs is None:
      n_nonzero_coefs = int(0.1 * X.shape[1])

    return orthogonal_mp_gram(gram, Xy, copy_Gram=False, copy_Xy=False, n_nonzero_coefs=n_nonzero_coefs)

  def fit(self, X, Dinit=None):
    """
    Use data to learn dictionary and activations.
    Parameters
    ----------
    X: data. (shape = [n_features, n_samples])
    Dinit: initialization of dictionary. (shape = [n_features, n_components])
    """
    if Dinit is None:
      D = self._initialize(X)
    else:
      D = Dinit / splin.norm(Dinit, axis=0)[sp.newaxis,:]

    for i in range(self.max_iter):
      gamma = self._transform(D, X)
      e = splin.norm(X - D.dot(gamma))
      if e < self.tol:
        break
      D, gamma = self._update_dict(X, D, gamma)

    self.components_ = D
    self.gamma_ = gamma
    return self

  def transform(self, X):
    return self._transform(self.components_, X)


class DKSVD(object):
  """
    Implementation of the Label consistent KSVD algorithm proposed by Zhuolin Jiang, Zhe Lin and Larry S. Davis.
    This implementation is a translation of the matlab code released by the authors on http://users.umiacs.umd.edu/~zhuolin/projectlcksvd.html.
    The code has been extended in order to use the related method called Discriminative KSVD proposed by Zhang, Qiang and Li, Baoxin.
    Author: Adrien Lagrange (ad.lagrange@gmail.com)
    Date: 25-10-2018
  """
  def __init__(self):
    super(DKSVD, self).__init__()

  def initialization4LCKSVD(self,training_feats,H_train,dictsize,iterations,sparsitythres,tol=1e-4):
    """
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
    """

    numClass = H_train.shape[0] # number of objects
    numPerClass = round(dictsize/float(numClass)) # initial points from each class
    Dinit = sp.empty((training_feats.shape[0],numClass*numPerClass)) # for LC-Ksvd1 and LC-Ksvd2
    dictLabel = sp.zeros((numClass,numPerClass))

    runKsvd = ApproximateKSVD(numPerClass, max_iter=iterations, tol=tol, transform_n_nonzero_coefs=sparsitythres)
    for classid in range(numClass):

      col_ids = sp.logical_and(H_train[classid,:]==1,sp.sum(training_feats**2, axis=1) > 1e-6)

      #  Initilization for LC-KSVD (perform KSVD in each class)
      Dpart = training_feats[:,col_ids][:,sp.random.choice(col_ids.sum(),numPerClass,replace=False)]
      Dpart = Dpart/splin.norm(Dpart,axis=0)
      para_data = training_feats[:,col_ids]
    
      # ksvd process
      runKsvd.fit(training_feats[:,col_ids])
      Dinit[:,numPerClass*classid:numPerClass*(classid+1)] = runKsvd.components_
    
      dictLabel[classid,numPerClass*classid:numPerClass*(classid+1)] =  1.

    T = sp.eye(dictsize) # scale factor
    Q = sp.zeros((dictsize,training_feats.shape[1])) # energy matrix
    for frameid in range(training_feats.shape[1]):
      for itemid in range(Dinit.shape[1]):
        Q[sp.ix_(dictLabel==itemid,H_train==frameid)] =1.

    # ksvd process
    runKsvd.fit(training_feats,Dinit=Dinit)
    Xtemp = runKsvd.gamma_

    # learning linear classifier parameters
    Winit = splin.pinv(Xtemp.dot(Xtemp.T)+sp.eye(Xtemp.shape[0])).dot(Xtemp).dot(H_train.T)
    Tinit = splin.pinv(Xtemp.dot(Xtemp.T)+sp.eye(Xtemp.shape[0])).dot(Xtemp).dot(Q.T)

    return Dinit,Tinit.T,Winit.T,Q

  def initialization4DKSVD(self,training_feats,labels,dictsize,iterations,sparsitythres,Dinit=None,tol=1e-4):
    """
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
    """

    H_train = sp.zeros((int(labels.max()),training_feats.shape[1]),dtype=float)
    for c in range(int(labels.max())):
      H_train[c,labels==(c+1)]=1.

    if Dinit is None:
      Dinit = training_feats[:,sp.random.choice(training_feats.shape[1],dictsize,replace=False)]

    # ksvd process
    runKsvd = ApproximateKSVD(dictsize, max_iter=iterations, tol=tol, transform_n_nonzero_coefs=sparsitythres)
    runKsvd.fit(training_feats,Dinit=Dinit)

    # learning linear classifier parameters
    Winit = splin.pinv(runKsvd.gamma_.dot(runKsvd.gamma_.T)+sp.eye(runKsvd.gamma_.shape[0])).dot(runKsvd.gamma_).dot(H_train.T)

    return Dinit,Winit.T

  def labelconsistentksvd(self,Y,Dinit,labels,Winit,iterations,sparsitythres,sqrt_beta,sqrt_alpha=0.,Q_train=None,Tinit=None, tol=1e-4):
    """
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
    """
    
    H_train = sp.zeros((int(labels.max()),Y.shape[1]),dtype=float)
    print H_train.shape
    for c in range(int(labels.max())):
      H_train[c,labels==(c+1)]=1.

    # ksvd process
    runKsvd = ApproximateKSVD(Dinit.shape[1], max_iter=iterations, tol=tol, transform_n_nonzero_coefs=sparsitythres)
    if sqrt_alpha == 0:
      runKsvd.fit(sp.vstack((Y,sqrt_beta*H_train)), Dinit=sp.vstack((Dinit,sqrt_beta*Winit)))
    else:
      runKsvd.fit(sp.vstack((Y,sqrt_alpha*Q_train,sqrt_beta*H_train)), Dinit=sp.vstack((Dinit,sqrt_alpha*Tinit,sqrt_beta*Winit)))

    # get back the desired D, T, W
    i_end_D   = Dinit.shape[0]
    if sqrt_alpha == 0:
      i_start_W = i_end_D
      i_end_W   = i_end_D+Winit.shape[0]
      D = runKsvd.components_[:i_end_D,:]
      W = runKsvd.components_[i_start_W:i_end_W,:]
      T = None
    else:
      i_start_T = i_end_D
      i_end_T   = i_end_D+Tinit.shape[0]
      i_start_W = i_end_T
      i_end_W   = i_end_T+Winit.shape[0]
      D = runKsvd.components_[:i_end_D,:]
      T = runKsvd.components_[i_start_T:i_end_T,:]
      W = runKsvd.components_[i_start_W:i_end_W,:]

    # normalization
    l2norms = splin.norm(D,axis=0)[sp.newaxis,:] + tol
    D /= l2norms
    W /= l2norms
    W /= sqrt_beta
    if sqrt_alpha != 0:
      T /= l2norms
      T /= sqrt_alpha

    return D,runKsvd.gamma_,T,W

  def classification(self, D, W, data, sparsity):
    """
    Classification 
    Inputs
          D               -learned dictionary
          W               -learned classifier parameters
          data            -data to classify
          sparsity        -sparsity threshold
    outputs
          prediction      -predicted classification vectors. Perform sp.argmax(W.dot(gamma), axis=0) to get labels
          gamma           -learned representation
    """

    # sparse coding
    G = D.T.dot(D)
    gamma = orthogonal_mp_gram(G, D.T.dot(data), copy_Gram=False, copy_Xy=False, n_nonzero_coefs=sparsity)
    # # classify process
    # prediction = sp.argmax(W.dot(gamma), axis=0)

    return W.dot(gamma),gamma
