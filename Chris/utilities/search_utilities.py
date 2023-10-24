import numpy as np

class RandomizedSearch:
    """
    RandomizedSearchCV does not have a callback (as we used with BayesSearchCV) or easy workaround,
    so we implement our own basic random search class...
    """
    def __init__(self, pipeline, param_distributions, scoring='dbcv', n_iter=10, random_seed=None):

        self.pipeline = pipeline
        self.param_distributions = param_distributions
        self.scoring = scoring
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
            all_scores = cv_score(pipeline, X, score='all')
            this_score = all_scores[self.scoring]

            if not np.isnan(this_score) and (
                    self.results_['fun'] is None or this_score > self.results_['fun']
            ):
                self.results_['fun'] = this_score
                self.results_['x'] = params

            callback(self.results_, params, all_scores)

    def choose_params(self):

        dist = rng.choice(self.param_distributions)
        # Always sort the keys of a dictionary, for reproducibility
        items = sorted(dist.items())
        params = dict()
        for k, v in items:
            if hasattr(v, "rvs"):
                params[k] = v.rvs(random_state=rng)
            else:
                params[k] = v[rng.randint(len(v))]
        return params
