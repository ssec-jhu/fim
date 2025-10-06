"""MaterialModel class for unified FIM pipeline
Description: Encapsulates material model type and parameters,
and delegates computation to model-specific functions.
"""

import logging

from fim.refactor.vws_models import (
    calculate_VWS_hgo,
    calculate_VWS_linear,
    calculate_VWS_nh,
    sensitivity_full_hgo,
    sensitivity_full_linear,
    sensitivity_nh,
)


class MaterialModel:
    """Encapsulates logic for different material models and their parameter management."""

    def __init__(self, model_name: str, parameters: dict):
        self.name = model_name
        self.params = parameters
        self.model_func = self._get_material_model(model_name)

    def _get_material_model(self, model_name):
        logging.info("Selecting material model: %s", model_name)
        if model_name == "linear":
            return calculate_VWS_linear
        if model_name == "hgo":
            return calculate_VWS_hgo
        if model_name == "nh":
            return calculate_VWS_nh
        raise ValueError(f"Unsupported model name: {model_name}")

    def get_parameter(self, key, default=None):
        return self.params.get(key, default)

    def sensitivity_analysis_linear(self, tensor_displacement_list, X, Y, Z, volume_matrix, L, H, deviation=0.05):
        if self.name != "linear":
            raise NotImplementedError("Sensitivity analysis for 'linear' model only.")

        E1 = self.get_parameter("E1")
        E2 = self.get_parameter("E2")
        v12 = self.get_parameter("v12")
        v23 = self.get_parameter("v23")
        Gt = self.get_parameter("Gt")
        Force = self.get_parameter("Force")

        return sensitivity_full_linear(
            tensor_displacement_list, E1, E2, v12, v23, Gt, X, Y, Z, Force, volume_matrix, L, H, deviation
        )

    def sensitivity_analysis_hgo(self, tensor_displacement_list, X, Y, Z, volume_matrix, L, H, deviation=0.05):
        if self.name != "hgo":
            raise NotImplementedError("Sensitivity analysis for 'hgo' model only.")

        C10 = self.get_parameter("C10")
        D1 = self.get_parameter("D1")
        k1 = self.get_parameter("k1")
        k2 = self.get_parameter("k2")
        kappa = self.get_parameter("kappa")
        Force = self.get_parameter("Force")

        return sensitivity_full_hgo(
            tensor_displacement_list, X, Y, Z, C10, D1, k1, k2, kappa, volume_matrix, Force, L, H, deviation
        )

    def sensitivity_analysis_nh(self, tensor_displacement_list, X, Y, Z, volume_matrix, L, H, deviation=0.05):
        if self.name != "nh":
            raise NotImplementedError("Sensitivity analysis for 'nh' model only.")

        C10 = self.get_parameter("C10")
        D1 = self.get_parameter("D1")
        Force = self.get_parameter("Force")

        return sensitivity_nh(tensor_displacement_list, X, Y, Z, C10, D1, volume_matrix, Force, L, H, deviation)

    def evaluate_virtual_fields(self, displacement_field, X, Y, Z, Force, volume_matrix):
        return self.model_func(displacement_field, X, Y, Z, Force, volume_matrix, self.params)

    def info(self):
        print(f"Model: {self.name}\nParameters:")
        for k, v in self.params.items():
            print(f"  {k}: {v}")

    def to_dict(self):
        return {"model_name": self.name, "parameters": self.params}
