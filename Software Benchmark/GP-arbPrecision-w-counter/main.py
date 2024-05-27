
from src.FlexGP import GaussianProcessRegressor, utils
from multiprocessing import Pool
import traceback
import random
import pandas as pd
from sklearn.datasets import fetch_california_housing

class NullWriter(object):
    def write(self, arg):
        pass

def run_experiment(args):
    config, california_housing = args
    print(config)
    print("Start Experiment: " + config["name"])
    try:
        GP = GaussianProcessRegressor(config, california_housing)
    except Exception as e:
        print("Error, skipping experiment " + config["name"])
        print(e)
        return
    try:
       GP.fit()
       GP.predict(testset=False)
       GP.predict(testset=True)
       GP.eval()
    except:
       print("Error, skipping experiment " + config["name"])
       GP.failed(traceback)


if __name__ == '__main__':
    experiments = utils.load_config_files()
    print(experiments)
    print("Experiments: " + str(len(experiments)))

    california_housing = fetch_california_housing(as_frame=True)
    california_housing = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)

    # Erstellen Sie einen Pool von 8 Prozessen
    with Pool(8) as p:
        random.shuffle(experiments)
        p.map(run_experiment, [(config, california_housing) for config in experiments])

