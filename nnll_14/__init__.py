#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->

# pylint: disable=import-outside-toplevel


class DualWeight:
    def __init__(self, weight_1=0.0, weight_2=0.0):
        self.combined_weight = weight_1 + (weight_2 * 0.01)

    def get_weight_1(self):
        return int(self.combined_weight)  # Extract integer part

    def get_weight_2(self):
        return (self.combined_weight - int(self.combined_weight)) * 100  # Extract fractional part

    def increment_weight_1(self, value):
        self.combined_weight += value
        self.combined_weight = int(self.combined_weight) + (self.combined_weight - int(self.combined_weight))  # Ensure no overflow into

    def increment_weight_2(self, value):
        fractional_part = (self.combined_weight - int(self.combined_weight)) * 100 + value
        self.combined_weight = int(self.combined_weight) + (fractional_part / 100)

    def decrement_weight_1(self, value):
        self.increment_weight_1(-value)

    def decrement_weight_2(self, value):
        self.increment_weight_2(-value)


dual_weight = DualWeight(5.0, 75.0)
print("Initial Weights - Weight 1:", dual_weight.get_weight_1(), "Weight 2:", dual_weight.get_weight_2())
dual_weight.increment_weight_1(3)
dual_weight.decrement_weight_2(10)

print("adjusted weights- weight 1:", dual_weight.get_weight_1(), "weight 2:", dual_weight.get_weight_2())
