from solvation_predictor.solubility.calculate_solubility import *
from solvation_predictor import inp
import datetime

if __name__ == "__main__":
    inp = inp.CommonArgs().parse_args()
    logging = create_logger()
    logger = logging.debug
    logger(f"Start training on {datetime.time()}")

    # df = pd.read_csv('/Users/u0161682/Documents/GitLab/SolProp_ML/Data/solubility_test.csv')

    # predictions = predict_property(csv_path=None,
    #                                df=df,
    #                                ghsolv=True,
    #                                gsolv=True,
    #                                hsolv=False,
    #                                saq=True,
    #                                solute_parameters=False,
    #                                reduced_number=False,
    #                                validate_data_list=['solute', 'solvent1'],
    #                                export_csv='./../model_predictions_test.csv',
    #                                logger='/Users/u0161682/Documents/GitLab/SolProp_ML/Model_predictions/logger.log')

    df = pd.read_csv('/Users/u0161682/Documents/GitLab/SolProp_ML/Data/Solubility/converted_solubilities.csv')

    results = calculate_solubility(path=None,
                                   df=df,
                                   validate_data_list=['solute', 'solvent', 'reference_solvent', 'temperature'],
                                   calculate_aqueous=True,
                                   calculate_Hdiss_T_dep=False,
                                   reduced_number=False,
                                   export_csv='./../solubility_predictions_test.csv',
                                   export_detailed_csv=True,
                                   solv_crit_prop_dict=None,
                                   logger='/Users/u0161682/Documents/GitLab/SolProp_ML/ModelPredictions/logger.log')
    logger(f"End training on {datetime.time()}")
