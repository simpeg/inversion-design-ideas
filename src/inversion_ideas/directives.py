"""
Directives to modify the objective function between iterations of an inversion.
"""


class MultiplierCooler:
    def __init__(self, cooling_factor):
        self.cooling_factor = cooling_factor

    def __call__(self, objective_function):
        if not hasattr(objective_function, "multiplier"):
            raise TypeError()
        objective_function.multiplier /= self.cooling_factor
