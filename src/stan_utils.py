from hashlib import md5
import logging
import os
import pickle
import pystan

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel('INFO')


def stan_model_cache(model_file):
    with open(model_file, 'r') as f:
        model_code = f.read()

    code_hash = md5(model_code.encode('ascii')).hexdigest()
    pickle_file = os.path.join(
        os.path.dirname(model_file),
        'cached-model-{}.pkl'.format(code_hash)
    )
    try:
        stan_model = pickle.load(open(pickle_file, 'rb'))
    except FileNotFoundError:
        stan_model = pystan.StanModel(file=model_file)
        with open(pickle_file, 'wb') as f:
            pickle.dump(stan_model, f)
    else:
        logger.info("Using cached StanModel")
    return stan_model
