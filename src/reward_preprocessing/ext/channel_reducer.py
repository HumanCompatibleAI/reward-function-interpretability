# Copyright 2018 The Lucid Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
This is a slightly modified version of channel_reducer.py from lucent.misc.
Here is the problem and reasoning for these changes:
channel_reducer in lucent and lucid reduces the innermost (last) dimension. This is fine
in lucid, because lucid has the convention that channels in images are the last dim.
However, in lucent the convention was changed to use the first dimension (after the
batch_size / num_samples dimension) for channels in images. A lot of code can therefore
not use this channel_reducer if we e.g. want to reduce the channel dim (e.g. in rl
vision / reward model interpret).
Consequently, I added an optional argument to choose which dimension should be reduced.

Helper for using sklearn.decomposition on high-dimensional tensors.

Provides ChannelReducer, a wrapper around sklearn.decomposition to help them
apply to arbitrary rank tensors. It saves lots of annoying reshaping.
"""

import numpy as np
from sklearn.base import BaseEstimator
import sklearn.decomposition


class ChannelReducer(object):
    """Helper for dimensionality reduction to the innermost dimension of a tensor.

    This class wraps sklearn.decomposition classes to help them apply to arbitrary
    rank tensors. It saves lots of annoying reshaping.

    See the original sklearn.decomposition documentation:
    http://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition
    """

    def __init__(self, n_components=3, reduction_alg="NMF", reduction_dim=-1, **kwargs):
        """
        Constructor for ChannelReducer.

        Inputs:
          n_components: Number of dimensions to reduce innermost dimension to.
          reduction_alg: A string or sklearn.decomposition class. Defaults to
            "NMF" (non-negative matrix factorization). Other options include:
            "PCA", "FastICA", and "MiniBatchDictionaryLearning". The name of any of
            the sklearn.decomposition classes will work, though.
          reduction_dim: Which dimension to reduce. Defaults to -1, which is the last /
            innermost dimension.
          kwargs: Additional kwargs to be passed on to the reducer.
        """

        if not isinstance(n_components, int):
            raise ValueError("n_components must be an int, not '%s'." % n_components)
        if n_components <= 0:
            raise ValueError("n_components must be strictly > 0")
        # Defensively look up reduction_alg if it is a string and give useful errors.
        algorithm_map = {}
        for name in dir(sklearn.decomposition):
            obj = sklearn.decomposition.__getattribute__(name)
            if isinstance(obj, type) and issubclass(obj, BaseEstimator):
                algorithm_map[name] = obj
        if isinstance(reduction_alg, str):
            if reduction_alg in algorithm_map:
                reduction_alg = algorithm_map[reduction_alg]
            else:
                raise ValueError(
                    "Unknown dimensionality reduction method '%s'." % reduction_alg
                )

        self.n_components = n_components
        self._reducer = reduction_alg(n_components=n_components, **kwargs)
        self._is_fit = False
        self.reduction_dim = reduction_dim

    @classmethod
    def _apply_flat(cls, f, acts, reduction_dim=-1):
        """
        Utility for applying f to inner dimension of activations.
        Flattens activations into a 2D tensor, applies f, then unflattens so that all
        dimensions except innermost are unchanged.
        keep_dim is the dimension that does not get flattened, i.e. the one that we
        want to reduce. Defaults to -1, which is the innermost aka last dimension.
        """
        orig_shape = acts.shape
        acts_flat = acts.reshape([-1, acts.shape[reduction_dim]])
        new_flat = f(acts_flat)
        if not isinstance(new_flat, np.ndarray):
            return new_flat
        new_shape = list(orig_shape)
        # All dimensions go back to the previous shape except the dimensions we didn't,
        # flatten, which is the dimension we reduced.
        new_shape[reduction_dim] = -1
        return new_flat.reshape(new_shape)

    def fit(self, acts):
        """Learn a model of dim reduction for the data. Returns the instance of the
        compositions itself."""
        self._is_fit = True
        return ChannelReducer._apply_flat(
            self._reducer.fit, acts, reduction_dim=self.reduction_dim
        )

    def fit_transform(self, acts):
        """Learn a model of dim reduction for the data. Returns the actual reduced
        data."""
        self._is_fit = True
        return ChannelReducer._apply_flat(
            self._reducer.fit_transform, acts, reduction_dim=self.reduction_dim
        )

    def transform(self, acts):
        """Return the data as it was learned."""
        return ChannelReducer._apply_flat(
            self._reducer.transform, acts, reduction_dim=self.reduction_dim
        )

    def __call__(self, acts):
        if self._is_fit:
            return self.transform(acts)
        else:
            return self.fit_transform(acts)

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name + "_" in self._reducer.__dict__:
            return self._reducer.__dict__[name + "_"]

    def __dir__(self):
        dynamic_attrs = [
            name[:-1]
            for name in dir(self._reducer)
            if name[-1] == "_" and name[0] != "_"
        ]

        return (
            list(ChannelReducer.__dict__.keys())
            + list(self.__dict__.keys())
            + dynamic_attrs
        )
