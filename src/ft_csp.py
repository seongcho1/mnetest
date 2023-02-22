import copy as cp
# from mne.cov import _regularized_covariance
# from mne.fixes import pinv
# from mne.utils import fill_doc, _validate_type, copy_doc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin

# https://towardsdatascience.com/customizing-sklearn-pipelines-transformermixin-a54341d8d624
# https://github.com/mne-tools/mne-python/blob/main/mne/decoding/csp.py
class FT_CSP(TransformerMixin, BaseEstimator):
    """M/EEG signal decomposition using the Common Spatial Patterns (CSP).
    This class can be used as a supervised decomposition to estimate spatial
    filters for feature extraction. CSP in the context of EEG was first
    described in :footcite:`KolesEtAl1990`; a comprehensive tutorial on CSP can
    be found in :footcite:`BlankertzEtAl2008`. Multi-class solving is
    implemented from :footcite:`Grosse-WentrupBuss2008`.
    Parameters
    ----------
    n_components : int (default 4)
        The number of components to decompose M/EEG signals. This number should
        be set by cross-validation.
    reg : float | str | None (default None)
        If not None (same as ``'empirical'``, default), allow regularization
        for covariance estimation. If float (between 0 and 1), shrinkage is
        used. For str values, ``reg`` will be passed as ``method`` to
        :func:`mne.compute_covariance`.
    log : None | bool (default None)
        If ``transform_into`` equals ``'average_power'`` and ``log`` is None or
        True, then apply a log transform to standardize features, else features
        are z-scored. If ``transform_into`` is ``'csp_space'``, ``log`` must be
        None.
    cov_est : 'concat' | 'epoch' (default 'concat')
        If ``'concat'``, covariance matrices are estimated on concatenated
        epochs for each class. If ``'epoch'``, covariance matrices are
        estimated on each epoch separately and then averaged over each class.
    transform_into : 'average_power' | 'csp_space' (default 'average_power')
        If 'average_power' then ``self.transform`` will return the average
        power of each spatial filter. If ``'csp_space'``, ``self.transform``
        will return the data in CSP space.
    norm_trace : bool (default False)
        Normalize class covariance by its trace. Trace normalization is a step
        of the original CSP algorithm :footcite:`KolesEtAl1990` to eliminate
        magnitude variations in the EEG between individuals. It is not applied
        in more recent work :footcite:`BlankertzEtAl2008`,
        :footcite:`Grosse-WentrupBuss2008` and can have a negative impact on
        pattern order.
    cov_method_params : dict | None
        Parameters to pass to :func:`mne.compute_covariance`.
        .. versionadded:: 0.16
    %(rank_none)s
        .. versionadded:: 0.17
    component_order : 'mutual_info' | 'alternate' (default 'mutual_info')
        If ``'mutual_info'`` order components by decreasing mutual information
        (in the two-class case this uses a simplification which orders
        components by decreasing absolute deviation of the eigenvalues from 0.5
        :footcite:`BarachantEtAl2010`). For the two-class case, ``'alternate'``
        orders components by starting with the largest eigenvalue, followed by
        the smallest, the second-to-largest, the second-to-smallest, and so on
        :footcite:`BlankertzEtAl2008`.
        .. versionadded:: 0.21
    Attributes
    ----------
    filters_ :  ndarray, shape (n_channels, n_channels)
        If fit, the CSP components used to decompose the data, else None.
    patterns_ : ndarray, shape (n_channels, n_channels)
        If fit, the CSP patterns used to restore M/EEG signals, else None.
    mean_ : ndarray, shape (n_components,)
        If fit, the mean squared power for each component.
    std_ : ndarray, shape (n_components,)
        If fit, the std squared power for each component.
    See Also
    --------
    mne.preprocessing.Xdawn, SPoC
    References
    ----------
    .. footbibliography::
    """


    def __init__(self, n_components=4, reg=None, log=None, cov_est='concat',
                 transform_into='average_power', norm_trace=False,
                 cov_method_params=None, rank=None,
                 component_order='mutual_info'):
        # Init default CSP
        self.PRT = False
        if not isinstance(n_components, int):
            raise ValueError('n_components must be an integer.')
        self.n_components = n_components
        self.reg = reg
        self.log = log
        # Init default cov_est
        if not (cov_est == "concat" or cov_est == "epoch"):
            raise ValueError("unknown covariance estimation method")
        self.cov_est = cov_est

        self.transform_into = transform_into
        self.norm_trace = norm_trace

        self.cov_method_params = cov_method_params
        self.rank = rank

        self.component_order = component_order

    def _check_Xy(self, X, y=None):
        """Check input data."""
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be of type ndarray (got %s)."
                             % type(X))
        if y is not None:
            if len(X) != len(y) or len(y) < 1:
                raise ValueError('X and y must have the same length.')
        if X.ndim < 3:
            raise ValueError('X must have at least 3 dimensions.')

    def fit(self, X, y):
        """Estimate the CSP decomposition on epochs.
        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            The data on which to estimate the CSP.
        y : array, shape (n_epochs,)
            The class for each epoch.
        Returns
        -------
        self : instance of CSP
            Returns the modified instance.
        """

        if self.PRT:
            print("\n" + ">"*42*2)
            print(f">>>csp.fit(X, y), X.shape={X.shape}, y.shape={y.shape}<<<")
            print("<"*42*2 + "\n")

        self._check_Xy(X, y)

        if self.PRT:
            print("X.shape", X.shape)
            print("len(y)", len(y), y)

        self._classes = np.unique(y)
        if self.PRT:
            print("_classes", self._classes)
        n_classes = len(self._classes)
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2.")

        covs, sample_weights = self._compute_covariance_matrices(X, y)
        if self.PRT:
            print(f"covs.shape={covs.shape}, sample_weight={sample_weights}")

        eigen_vectors, eigen_values = self._decompose_covs(covs,
                                                           sample_weights)

        if self.PRT:
            print(f"eigenvalues.shape={eigen_values.shape}, eigenvectors.shape={eigen_vectors.shape}\n")
            print(f"eigenvalues={eigen_values}, eigenvectors.shape={eigen_vectors.shape}\n")

        #np.argsort(np.abs(eigen_values - 0.5))[::-1]
        if self.PRT:
            print(f"np.abs(eigen_values - 0.5)={np.abs(eigen_values - 0.5)}\n")
            print(f"np.argsort(np.abs(eigen_values - 0.5))={np.argsort(np.abs(eigen_values - 0.5))}\n")
            print(f"np.argsort(np.abs(eigen_values - 0.5))[::-1]={np.argsort(np.abs(eigen_values - 0.5))[::-1]}\n")

        ix = self._order_components(covs, sample_weights, eigen_vectors, eigen_values)
        if self.PRT:
            print(f"ix={ix}\n")

        eigen_vectors = eigen_vectors[:, ix]
        if self.PRT:
            print(f"eigenvectors2.shape={eigen_vectors.shape}")

        self.filters_ = eigen_vectors.T
        self.patterns_ = pinv2(eigen_vectors)

        pick_filters = self.filters_[:self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])

        # compute features (mean power)
        X = (X ** 2).mean(axis=2)

        # To standardize features
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

        return self

    def transform(self, X):
        """Estimate epochs sources given the CSP filters.
        Parameters
        ----------
        X : array, shape (n_epochs, n_channels, n_times)
            The data.
        Returns
        -------
        X : ndarray
            If self.transform_into == 'average_power' then returns the power of
            CSP features averaged over time and shape (n_epochs, n_sources)

        """
        if self.PRT:
            print("\n" + ">"*42*2)
            print(f">>>csp.transform(X), X.shape={X.shape}<<<")
            print("<"*42*2 + "\n")

        if not isinstance(X, np.ndarray):
            raise ValueError("X should be of type ndarray (got %s)." % type(X))
        if self.filters_ is None:
            raise RuntimeError('No filters available. Please first fit CSP '
                               'decomposition.')

        pick_filters = self.filters_[:self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])

        # compute features (mean band power)
        if self.transform_into == 'average_power':
            X = (X ** 2).mean(axis=2)
            log = True if self.log is None else self.log
            if log:
                X = np.log(X)
            else:
                X -= self.mean_
                X /= self.std_
        return X

    def fit_transform(self, X, y, **fit_params):  # noqa: D102
        """Fit to data, then transform it.
        Fits transformer to ``X`` and ``y`` with optional parameters
        ``fit_params``, and returns a transformed version of ``X``.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Training set.
        y : array, shape (n_samples,)
            Target values or class labels.
        **fit_params : dict
            Additional fitting parameters passed to the ``fit`` method..
        Returns
        -------
        X_new : array, shape (n_samples, n_features_new)
            Transformed array.
        """

        # fit method of arity 2 (supervised transformation)
        self.fit(X, y)
        return self.transform(X)

    def plot_patterns(
            self, info, components=None, *, average=None, ch_type=None,
            scalings=None, sensors=True, show_names=False, mask=None,
            mask_params=None, contours=6, outlines='head', sphere=None,
            image_interp='cubic',
            extrapolate='auto', border='mean', res=64,
            size=1, cmap='RdBu_r', vlim=(None, None), cnorm=None,
            colorbar=True, cbar_fmt='%3.1f', units=None, axes=None,
            name_format='CSP%01d', nrows=1, ncols='auto', show=True):
        """Plot topographic patterns of components.
        The patterns explain how the measured data was generated from the
        neural sources (a.k.a. the forward model).
        Parameters
        ----------
        %(info_not_none)s Used for fitting. If not available, consider using
            :func:`mne.create_info`.
        components : float | array of float | None
           The patterns to plot. If ``None``, all components will be shown.
        %(average_plot_evoked_topomap)s
        %(ch_type_topomap)s
        scalings : dict | float | None
            The scalings of the channel types to be applied for plotting.
            If None, defaults to ``dict(eeg=1e6, grad=1e13, mag=1e15)``.
        %(sensors_topomap)s
        %(show_names_topomap)s
        %(mask_patterns_topomap)s
        %(mask_params_topomap)s
        %(contours_topomap)s
        %(outlines_topomap)s
        %(sphere_topomap_auto)s
        %(image_interp_topomap)s
        %(extrapolate_topomap)s
            .. versionadded:: 1.3
        %(border_topomap)s
            .. versionadded:: 1.3
        %(res_topomap)s
        %(size_topomap)s
        %(cmap_topomap)s
        %(vlim_plot_topomap)s
            .. versionadded:: 1.3
        %(cnorm)s
            .. versionadded:: 1.3
        %(colorbar_topomap)s
        %(cbar_fmt_topomap)s
        %(units_topomap)s
        %(axes_evoked_plot_topomap)s
        name_format : str
            String format for topomap values. Defaults to "CSP%%01d".
        %(nrows_ncols_topomap)s
            .. versionadded:: 1.3
        %(show)s
        Returns
        -------
        fig : instance of matplotlib.figure.Figure
           The figure.
        """
        print("\n" + ">"*42*2)
        print(f">>>csp.plot_patterns(info), info.type={type(info)}, \ninfo=\n{info}<<<")
        print("<"*42*2 + "\n")

        from mne import EvokedArray

        if units is None:
            units = 'AU'
        if components is None:
            components = np.arange(self.n_components)

        # set sampling frequency to have 1 component per time point
        info = cp.deepcopy(info)
        with info._unlock():
            info['sfreq'] = 1.
        # create an evoked
        patterns = EvokedArray(self.patterns_.T, info, tmin=0)
        # the call plot_topomap
        fig = patterns.plot_topomap(
            times=components, average=average, ch_type=ch_type,
            scalings=scalings, sensors=sensors, show_names=show_names,
            mask=mask, mask_params=mask_params, contours=contours,
            outlines=outlines, sphere=sphere, image_interp=image_interp,
            extrapolate=extrapolate, border=border, res=res, size=size,
            cmap=cmap, vlim=vlim, cnorm=cnorm, colorbar=colorbar,
            cbar_fmt=cbar_fmt, units=units, axes=axes, time_format=name_format,
            nrows=nrows, ncols=ncols, show=show)
        fig.show()

    def _compute_covariance_matrices(self, X, y):
        _, n_channels, _ = X.shape

        covs = []
        sample_weights = []
        for this_class in self._classes:
            cov, weight = self._concat_cov(X[y == this_class])

            if self.norm_trace:
                cov /= np.trace(cov)

            covs.append(cov)
            sample_weights.append(weight)

        if self.PRT:
            print(f"===_compute_covariance_matrices===, sample_weights={sample_weights}")
        return np.stack(covs), np.array(sample_weights)

    def _concat_cov(self, x_class, ddof=1):
        """Concatenate epochs before computing the covariance."""
        _, n_channels, _ = x_class.shape
        if self.PRT:
            print(f"_concat_cov: x_class.shape0={x_class.shape}")
        x_class = np.transpose(x_class, [1, 0, 2])
        if self.PRT:
            print(f"_concat_cov: x_class.shape1={x_class.shape}")
        x_class = x_class.reshape(n_channels, -1)
        if self.PRT:
            print(f"_concat_cov: x_class.shape2={x_class.shape}")
        # cov0 = _regularized_covariance(
        #     x_class, reg=self.reg, method_params=self.cov_method_params,
        #     rank=self.rank)
        #cov1 = np.cov(x_class, bias=False, ddof=1)
        #cov1 = np.cov(x_class, bias=1)
        cov1 = np.cov(x_class)
        if self.PRT:
            print(f"_concat_cov: x_class.shape3={x_class.shape}, x_class.T.shape={x_class.T.shape}, x_class.T.conj().shape={x_class.T.conj().shape}")

        cov = np.dot(x_class, x_class.T.conj()) / float(x_class.shape[1] - ddof)
        if self.PRT:
            print(f"_concat_cov: cov.shape={cov.shape}")

        weight = x_class.shape[0]

        if self.PRT:
            print(f"==_concat_cov==, n_channels={n_channels}, weight={weight}")
            print(f"compare cov, {cov[0][0]}, cov1 {cov1[0][0]}, {abs(cov[0][0] - cov1[0][0]) < 0.0000001}")
        return cov, weight

    def _decompose_covs(self, covs, sample_weights):
        from scipy import linalg
        n_classes = len(covs)
        if n_classes == 2:
            eigen_values, eigen_vectors = linalg.eigh(covs[0], covs.sum(0))
        else:
            raise NotImplementedError('not implemented for the case of more than 2 classes')
        return eigen_vectors, eigen_values


    def _normalize_eigenvectors(self, eigen_vectors, covs, sample_weights):
        # Here we apply an euclidean mean. See pyRiemann for other metrics
        mean_cov = np.average(covs, axis=0, weights=sample_weights)

        for ii in range(eigen_vectors.shape[1]):
            tmp = np.dot(np.dot(eigen_vectors[:, ii].T, mean_cov),
                         eigen_vectors[:, ii])
            eigen_vectors[:, ii] /= np.sqrt(tmp)
        return eigen_vectors

    def _order_components(self, covs, sample_weights, eigen_vectors, eigen_values):
        n_classes = len(self._classes)
        if n_classes > 2:
            raise NotImplementedError('not implemented for the case of more than 2 classes')
        elif n_classes == 2:
            ix = np.argsort(np.abs(eigen_values - 0.5))[::-1]

        return ix

def pinv2(a, rtol=1e-05):  # rtol=None):
    """Compute a pseudo-inverse of a matrix."""
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    del a
    maxS = np.max(s)
    if rtol is None:
        rtol = max(vh.shape + u.shape) * np.finfo(u.dtype).eps
    rank = np.sum(s > maxS * rtol)
    u = u[:, :rank]
    u /= s[:rank]
    return (u @ vh[:rank]).conj().T
