# This file is part of the dpatk package.
# Copyright (C) 2021  Corentin Martens
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Contact: corentin.martens@ulb.be

import numpy as np
from abc import ABC, abstractmethod
from numpy import ndarray
from sklearn.base import BaseEstimator
from sklearn.decomposition import FactorAnalysis, FastICA, IncrementalPCA, PCA
from typing import Optional


class ModelBase(ABC):

    """
    Base class for dynamic PET models.

    Parameters
    ----------
    nb_of_parameters : int
        The number of parameters of the model. Must be > 0.

    Attributes
    ----------
    _nb_of_parameters : int
        The number of parameters of the model.

    """

    def __init__(self, nb_of_parameters):

        self._name = None
        self._nb_of_parameters = nb_of_parameters

    @abstractmethod
    def generate(self, parameters):

        """Generates dynamic PET data from model parameter values.

        Parameters
        ----------
        parameters : ndarray
            A 2D array with cases on axis 0 and model parameter values on axis 1.

        Returns
        -------
        data : ndarray
            A 2D array with cases on axis 0 and dynamic PET samples on axis 1.

        """
        pass

    def get_name(self):

        """Returns the model name.

        Returns
        -------
        name : str
            The model name.

        """

        name = self._name

        return name

    @abstractmethod
    def transform(self, data):

        """Transforms dynamic PET data into model parameter values.

        Parameters
        ----------
        data : ndarray
            A 2D array with cases on axis 0 and dynamic PET samples on axis 1.

        Returns
        -------
        parameters : ndarray
            A 2D array with cases on axis 0 and model parameter values on axis 1.

        """
        pass


class DecompositionModelBase(ModelBase):

    """
    Base class for decomposition dynamic PET models derived from the sklearn.decomposition module.

    Parameters
    ----------
    nb_of_components : int
        The number of components (parameters) of the model. Must be > 0.

    Attributes
    ----------
    _is_fitted : bool
        Specifies whether the model is fitted.
    _model : BaseEstimator
        The sklearn internal decomposition model.
    
    """

    def __init__(self, nb_of_components: Optional[int] = None):

        super().__init__(nb_of_components)

        # Forces the class to be abstract without any abstractmethod.
        if self.__class__ is DecompositionModelBase:
            raise TypeError(f'Can\'t instantiate abstract class {DecompositionModelBase.__name__}')

        assert nb_of_components is None or nb_of_components > 0

        self._is_fitted = False
        self._model = None

    def fit(self, data):

        """Fits the model to the dynamic PET data.

        Parameters
        ----------
        data : ndarray
            A 2D array with cases on axis 0 and dynamic PET samples on axis 1.

        """

        assert data.ndim == 2
        assert data.shape[0] >= data.shape[1]
        assert data.shape[1] >= self._nb_of_parameters

        self._model.fit(data)
        self._is_fitted = True

    def generate(self, parameters):

        """Generates dynamic PET data from model parameter (component) values.

        Parameters
        ----------
        parameters : ndarray
            A 2D array with cases on axis 0 and model parameter (component) values on axis 1.

        Returns
        -------
        data : ndarray
            A 2D array with cases on axis 0 and dynamic PET samples on axis 1.

        """

        assert self._is_fitted
        assert 0 < parameters.ndim < 3
        assert parameters.shape[-1] <= self._model.n_components_

        if parameters.ndim == 1:
            parameters = parameters[np.newaxis, ...]

        components = np.zeros(parameters.shape[0], self._model.n_components_)
        components[:, :parameters.shape[-1]] = parameters
        data = self._model.inverse_transform(components)

        if data.shape[0] == 1:
            data = np.squeeze(data, axis=0)

        return data

    def get_components(self):

        """Returns the model components.

        Returns
        -------
        components : ndarray
            A 2D array with for each component (axis 0), the corresponding weights of the original data features
            (axis 1).

        """

        assert self._is_fitted

        components = self._model.components_[:self._nb_of_parameters]

        return components

    def transform(self, data):

        """Transforms dynamic PET data into model parameter (component) values.

        Parameters
        ----------
        data : ndarray
            A 2D array with cases on axis 0 and dynamic PET samples on axis 1.

        Returns
        -------
        parameters : ndarray
            A 2D array with cases on axis 0 and model parameter (component) values on axis 1.

        """

        assert self._is_fitted
        assert 0 < data.ndim < 3
        assert data.shape[-1] == self._model.components_.shape[1]

        if data.ndim == 1:
            data = data[np.newaxis, ...]

        parameters = self._model.transform(data)[:, :self._nb_of_parameters]

        if parameters.shape[0] == 1:
            parameters = np.squeeze(parameters, axis=0)

        return parameters


class FAModel(DecompositionModelBase):

    """
    A factor analysis dynamic PET model.

    Parameters
    ----------
    nb_of_components : int
        The number of components (parameters) of the model. Must be > 0.
    tolerance : float, optional
        The tolerance for convergence. Default: 1e-2.
    max_iter : int, optional
        The maximum number of iterations for convergence. Default: 1000.
    rotation : {‘varimax’, ‘quartimax’}, optional
        The factor rotation method. Default: None, no rotation.

    Attributes
    ----------
    _model : FactorAnalysis
        The sklearn internal FA model.

    """

    def __init__(self, nb_of_components: int, tolerance: float = 1e-2, max_iter: int = 1000,
                 rotation: Optional[str] = None):

        super().__init__(nb_of_components)

        assert tolerance > 0.0
        assert max_iter > 0
        assert rotation is None or rotation in ['varimax', 'quartimax']

        # All component values are computed, then returned values are truncated to _nb_parameters.
        self._model = FactorAnalysis(n_components=None, tol=tolerance, copy=True, max_iter=max_iter,
                                     noise_variance_init=None, svd_method='randomized', iterated_power=3,
                                     rotation=rotation, random_state=0)
        self._name = 'FA'


class ICAModel(DecompositionModelBase):

    """
    An independent component analysis dynamic PET model.

    Parameters
    ----------
    nb_of_components : int
        The number of components (parameters) of the model. Must be > 0.
    tolerance : float, optional
        The tolerance for convergence. Default: 1e-4.
    max_iter : int, optional
        The maximum number of iterations for convergence. Default: 200.

    Attributes
    ----------
    _model : FastICA
        The sklearn internal ICA model.

    """

    def __init__(self, nb_of_components: int, tolerance: float = 1e-4, max_iter: int = 200):

        super().__init__(nb_of_components)

        # The data must be whiten for ICA.
        self._model = FastICA(n_components=nb_of_components, algorithm='parallel', whiten=True, fun='logcosh',
                              fun_args=None, max_iter=max_iter, tol=tolerance, w_init=None, random_state=None)
        self._name = 'ICA'


class PCAModel(DecompositionModelBase):

    """
    A principal component analysis dynamic PET model.

    Parameters
    ----------
    nb_of_components : int
        The number of components (parameters) of the model. Must be > 0.
    whiten : bool, optional
        Specifies whether the data must be whiten before model fitting. Default: False.
    incremental : bool, optional
        Specifies whether the incremental algorithm must be used. Set it to True for large datasets. Default: False.
    batch_size : int, optional
        The batch size for the incremental algorithm. Ignored if incremental is False. Default: None, set to 5 times the
        number of data features.

    Attributes
    ----------
    _model : PCA
        The sklearn internal ICA model.

    """

    def __init__(self, nb_of_components: int, whiten: bool = False, incremental: bool = True,
                 batch_size: Optional[int] = None):

        super().__init__(nb_of_components)

        assert batch_size is None or batch_size > 0

        # All component values are computed, then returned values are truncated to _nb_parameters.
        if incremental:
            self._model = IncrementalPCA(n_components=None, whiten=whiten, copy=True, batch_size=batch_size)
        else:
            self._model = PCA(n_components=None, copy=True, whiten=whiten, svd_solver='auto', tol=0.0,
                              iterated_power='auto', random_state=None)
        self._name = 'PCA'
