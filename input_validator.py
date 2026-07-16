
import pandas as pd


class InputValidator:

    def __init__(self, feature_engineer):

        self.feature_engineer = feature_engineer

    def validate(self, inputs):

        validated = {}

        for feature, value in inputs.items():

            # -------------------------
            # Empty value
            # -------------------------
            while str(value).strip() == "":

                print(f"\n❌ {feature} cannot be empty.")

                value = input(f"Enter {feature}: ")

            # -------------------------
            # Numeric Validation
            # -------------------------
            if feature in self.feature_engineer.column_types["numeric"]:

                while True:

                    try:

                        validated[feature] = float(value)

                        break

                    except ValueError:

                        print(f"\n❌ '{value}' is not a valid number.")

                        value = input(f"Enter {feature}: ")

            # -------------------------
            # Date Validation
            # -------------------------
            elif feature in self.feature_engineer.column_types["date"]:

                while True:

                    try:

                        pd.to_datetime(value)

                        validated[feature] = value

                        break

                    except Exception:

                        print(f"\n❌ '{value}' is not a valid date.")

                        print("Example: 2024-05-09")

                        value = input(f"Enter {feature}: ")

            # -------------------------
            # Everything Else
            # -------------------------
            else:

                validated[feature] = str(value).strip()

        return validated