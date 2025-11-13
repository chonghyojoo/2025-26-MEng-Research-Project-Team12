import os

from solvation_predictor.inp import TrainArgs, Gsolv, GsolvAqueous, Hsolv, LogSaq, Solute, GsolvHsolv
from solvation_predictor.train.train import load_checkpoint, load_scaler


class SolubilityModels:
    def __init__(
        self,
        reduced_number,
        load_ghsolv,
        load_g,
        load_h,
        load_saq,
        load_solute,
        logger=None,
        verbose=True,
    ):
        """
        Loads the required models for solvation free energy, enthalpy, and aqueous solubility.
            :param reduced_number: if true, only 3 models are considered per property to make predictions faster
            :param load_g: load models for solvation free energy
            :param load_h: load models for solvation enthalpy
            :param load_saq: load models for aqueous solubility
            :param load_solute: load models for solute parameters
            :param logger: logger file
            :param verbose: whether to show logger info or not
        """
        self.ghsolv_models = None
        self.g_models = None
        self.g_aq_models = None
        self.h_models = None
        self.saq_models = None
        self.solute_models = None
        self.logger = logger.info if logger is not None else print

        if load_ghsolv or load_g or load_h or load_saq:
            self.ghsolv_models = (
                self.load_models("GHsolvQMExpmix", GsolvHsolv, reduced_number=reduced_number, verbose=verbose)
                if load_ghsolv
                else None
            )
            # self.g_models = (
            #     self.load_models("GsolvExpQM", Gsolv, reduced_number=reduced_number, verbose=verbose)
            #     if load_g
            #     else None
            # )
            self.g_aq_models = (
                self.load_models("GHsolvQMExppure", GsolvAqueous, reduced_number=reduced_number, verbose=verbose)
                if load_g
                else None
            )
            # self.h_models = (
            #     self.load_models("HsolvExpQM", Hsolv, reduced_number=reduced_number, verbose=verbose)
            #     if load_h
            #     else None
            # )
            self.saq_models = (
                self.load_models("Saq", LogSaq, reduced_number=reduced_number, verbose=verbose)
                if load_saq
                else None
            )
            # self.solute_models = (
            #     self.load_models("Solute", Solute, reduced_number=reduced_number, verbose=verbose)
            #     if load_solute
            #     else None
            # )

    def load_models(self, property_name, inp, reduced_number=False, verbose=True):
        """
        Loads the models for the given property and corresponding input arguments.
            :param property_name:
            :param inp:
            :param reduced_number:
            :param verbose:
        """
        number = 10 if not reduced_number else 3

        paths = [
            os.path.join("trained_models", property_name, "model" + str(i) + ".pt")
            for i in range(number)
        ]

        if verbose:
            self.logger(f"Loading {number} {property_name} models.")

        input_arguments = inp().parse_args()
        scalers = []
        models = []
        for p in paths:
            input_arguments.model_path = p
            scaler = load_scaler(p, from_package=True)
            model = load_checkpoint(p, input_arguments, from_package=True)
            scalers.append(scaler)
            models.append(model)
        return input_arguments, scalers, models
