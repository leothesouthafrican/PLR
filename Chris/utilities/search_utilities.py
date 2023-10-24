import numpy as np
from collections.abc import Iterable, Mapping

class RandomizedSearch:
    """
    RandomizedSearchCV does not have a callback (as we used with BayesSearchCV) or easy workaround,
    so we implement our own basic random search class...
    """
    def __init__(self, pipeline, param_distributions, scorer, scoring='dbcv', n_iter=10, random_seed=None):

        if not isinstance(param_distributions, (Mapping, Iterable)):
            raise TypeError(
                "Parameter distribution is not a dict or a list,"
                f" got: {param_distributions!r} of type "
                f"{type(param_distributions).__name__}"
            )

        if isinstance(param_distributions, Mapping):
            # wrap dictionary in a singleton list to support either dict
            # or list of dicts
            param_distributions = [param_distributions]

        for dist in param_distributions:
            if not isinstance(dist, dict):
                raise TypeError(
                    "Parameter distribution is not a dict ({!r})".format(dist)
                )
            for key in dist:
                if not isinstance(dist[key], Iterable) and not hasattr(
                        dist[key], "rvs"
                ):
                    raise TypeError(
                        f"Parameter grid for parameter {key!r} is not iterable "
                        f"or a distribution (value={dist[key]})"
                    )

        self.pipeline = pipeline
        self.param_distributions = param_distributions
        self.scoring = scoring
        self.scorer = scorer
        self.n_iter = n_iter
        self.rng = np.random.RandomState(random_seed)
        self.results_ = {
            'x_iters': [],
            'fun': None,
            'x': None
        }

    def fit(self, X, callback=None):
        for i in range(self.n_iter):
            params = self.choose_params()
            self.pipeline.set_params(**params)

            self.results_['x_iters'].append(params)
            all_scores = self.scorer(self.pipeline, X, score='all')
            this_score = all_scores[self.scoring]

            if not np.isnan(this_score) and (
                    self.results_['fun'] is None or this_score > self.results_['fun']
            ):
                self.results_['fun'] = this_score
                self.results_['x'] = params

            callback(self.results_, params, all_scores)

    def choose_params(self):

        dist = self.rng.choice(self.param_distributions)
        # Always sort the keys of a dictionary, for reproducibility
        items = sorted(dist.items())
        params = dict()
        for k, v in items:
            if hasattr(v, "rvs"):
                params[k] = v.rvs(random_state=self.rng)
            else:
                params[k] = v[self.rng.randint(len(v))]
        return params
