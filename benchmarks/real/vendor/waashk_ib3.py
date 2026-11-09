"""IB3 vendored from waashk/instanceselection (Cunha et al., ACM Computing Surveys 2023).

Source: https://github.com/waashk/instanceselection, commit a857d9c88a9d92e55083f8607932502cb989d7bf,
files src/main/python/iSel/base.py and src/main/python/iSel/ib3.py, concatenated below. The repository README states
the MIT License. Changes, each marked PATCH in the code:

1. base.py: `sklearn.externals.six` was removed in scikit-learn 0.23; the metaclass is now declared directly.
2. The module-level `random` calls go through `self.rng` (default `random.Random(0)`), so runs are reproducible and
   a different random source can be injected (verify_baselines.py injects KEEL's).
3. A debug `print` of the first instance is removed.
4. When no stored instance is acceptable, the original used a random *position* in CD as an instance index. It now
   uses the instance stored at that position, as KEEL's IB3 does.
5. The original removed members from CD while iterating over it, which skips the member after each removal. It now
   iterates over a copy, so every member within the similarity radius has its record updated, as in KEEL's IB3.

With patches 4 and 5 this is the same procedure as KEEL's Preprocess/Instance_Selection/IB3 (accept when
0.9 * accuracy > class frequency, drop when accuracy < 0.7 * class frequency).
"""

# ---- base.py ----
"""Base para metodos de redução"""
import warnings
class InstanceReductionWarning(UserWarning):
    pass
warnings.simplefilter("always", InstanceReductionWarning)

from sklearn.base import BaseEstimator
from abc import ABCMeta, abstractmethod

class InstanceSelectionBase(BaseEstimator, metaclass=ABCMeta):  # PATCH 1
    @abstractmethod
    def __init__(self):
        pass


class InstanceSelectionMixin(InstanceSelectionBase):

    """Mixin class for all instance reduction techniques"""
    def select_data(self, X, y):
        """
        Procedimento para redução dos dados. 
        # Entrada X e y
        # X = 
        # y =
        """
        pass

    def fit(self, X, y):
        """
        Call reduce data procedure
        """
        self.X = X
        self.y = y

        self.select_data(X, y)

        return self


# ---- ib3.py ----
import numpy as np
import random
from sklearn.utils.validation import check_X_y
from scipy.spatial.distance import cdist

import copy
from sklearn.metrics.pairwise import pairwise_distances

class IB3(InstanceSelectionMixin):
    """ Instance Base 3 (IB3)
    Descrição:
    ==========


    Parametros:
    ===========


    Atributos:
    ==========


    Ref.
    ====
    
    """

    def __init__(self, rng=None):  # PATCH 2: seedable random source
        self.rng = random.Random(0) if rng is None else rng
        self.sample_indices_ = []
        self.nAccept = 0.9
        self.nDrop = 0.7

    def isAcceptable(self, c, cClass, recordClass, freqClass):
        if recordClass[c].sum() != 0.0:
            minAc = (recordClass[c][0] / recordClass[c].sum()) * self.nAccept
        else:
            minAc = 0.
        maxFr = freqClass[c][cClass] / freqClass[c].sum()

        if (minAc > maxFr):
            return True

        return False

    def isDrop(self, c, cClass, recordClass, freqClass):

        maxAc = recordClass[c][0] / recordClass[c].sum()
        minFr = (freqClass[c][cClass] / freqClass[c].sum()) * self.nDrop

        if (maxAc < minFr):
            return True

        return False
    


    def select_data(self, X, y):

        X, y = check_X_y(X, y, accept_sparse="csr")
        #y = y.astype('int')

        PDMatrix = pairwise_distances(X=X, metric='euclidean')


        classes = np.unique(y)
        self.classes_ = classes
        self.mask = np.zeros(y.size, dtype=bool)

        #recordmatrix
        recordClass = np.zeros(shape=(y.size, 2))

        #Class frequency
        nclasses = len(set(self.classes_))
        freqClass = np.zeros(shape=(y.size, nclasses))

        #similarity vector
        sim = np.zeros(y.size) - 1.0
        CD = []




        xidx_random_list = list(range(X.shape[0]))
        self.rng.shuffle(xidx_random_list)  # PATCH 2

        for xidx in xidx_random_list:

            if len(CD) == 0:
                CD.append(xidx)
                self.mask[xidx] = True
                freqClass[xidx][y[xidx]]+=1
            
                recordClass[xidx][0] +=1
            else:
                sim[CD] = PDMatrix[xidx][CD]
                
                acceptable = False
                bestsim = np.inf

                for c in CD:
                    if self.isAcceptable(c, y[c], recordClass,freqClass):
                        if sim[c] < bestsim:
                            acceptable = True
                            cmax = copy.copy(c)
                            bestsim = copy.copy(sim[c])

                if not acceptable:
                    cmax = CD[self.rng.randint(0, len(CD)-1)]  # PATCH 4: was a position, not an instance

                if y[xidx] != y[cmax]:
                    CD.append(xidx)
                    self.mask[xidx] = True
                    sim[xidx] = 0



                for c in CD:
                    freqClass[c][y[xidx]] +=1

                for c in list(CD):  # PATCH 5: CD.remove below must not skip the next member
                    if sim[c] <= sim[cmax]:
                        if y[c] == y[xidx]:
                            recordClass[c][0]+=1
                        else:
                            recordClass[c][1]+=1

                        if self.isDrop(c, y[c], recordClass, freqClass):
                            CD.remove(c)
                            self.mask[c] = False

        self.X_ = np.asarray(X[self.mask])
        self.y_ = np.asarray(y[self.mask])
        self.sample_indices_ = list(sorted(np.where(self.mask == True)[0]))
        self.reduction_ = 1.0 - float(len(self.y_))/len(y)
        return self.X_, self.y_

