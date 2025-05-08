"""MaterialModel class for unified FIM pipeline
Description: Encapsulates material model type and parameters,
and delegates computation to model-specific functions.
"""

import logging

from vws_models import calculate_VWS_hgo, calculate_VWS_linear, sensitivity_full


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
        raise ValueError(f"Unsupported model name: {model_name}")

    def get_parameter(self, key, default=None):
        return self.params.get(key, default)

    def sensitivity_analysis(self, tensor_displacement_list, X, Y, Z, Force, cube_size, L, H, deviation=0.05):
        if self.name != "linear":
            raise NotImplementedError("Sensitivity analysis is only available for the 'linear' model.")

        E1 = self.get_parameter("E1")
        E2 = self.get_parameter("E2")
        v12 = self.get_parameter("v12")
        v23 = self.get_parameter("v23")
        Gt = self.get_parameter("Gt")

        return sensitivity_full(
            tensor_displacement_list, E1, E2, v12, v23, Gt, X, Y, Z, Force, cube_size, L, H, deviation
        )

    def evaluate_virtual_fields(self, displacement_field, X, Y, Z, Force, cube_size):
        return self.model_func(displacement_field, X, Y, Z, Force, cube_size, self.params)

    def info(self):
        print(f"Model: {self.name}\nParameters:")
        for k, v in self.params.items():
            print(f"  {k}: {v}")

    def to_dict(self):
        return {"model_name": self.name, "parameters": self.params}
