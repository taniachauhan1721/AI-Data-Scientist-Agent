from google import genai
import json
from config import API_KEY

class LLMAssistant:

    

    def __init__(self):

        self.client = genai.Client(
            api_key=API_KEY
        )




    def extract_features(self, user_prompt, feature_names):

      prompt = f"""
You are a machine learning assistant.

Extract ONLY the feature values mentioned.

Dataset features:

{feature_names}

Return ONLY valid JSON.

Example:

User:
Age is 45, Male, Cholesterol 230

Output:

{{
"Age":45,
"Sex":"Male",
"Cholesterol":230
}}

If a feature is not mentioned,
do not include it.

Return JSON only.
"""

       
       
      response = self.client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt + "\n\nUser:\n" + user_prompt
      )

      text = response.text.strip()

      text = text.replace("```json", "")
      text = text.replace("```", "")

      return json.loads(text)

    def find_missing_important_features(
        self,
        extracted_features: dict,
        important_features: list
        ):
        """Compare extracted features with the important features
    and return only the missing ones.
    """

        missing = []

        extracted = {
        key.lower().strip()
        for key in extracted_features.keys()
        }

        for feature in important_features:

          if feature.lower().strip() not in extracted:
            missing.append(feature)

        return missing

    def ask_missing_features(
        self,
        missing_features,
        feature_options
    ):
      """
    Ask the user only for missing important features.

    Small categorical features -> numbered menu.
    Large categorical features -> type the value.
    Numeric features -> type the value.
    """

      user_inputs = {}

      print("\nI need a few more important details.\n")

      for feature in missing_features:

        print(f"\n{feature}")

        # Feature has predefined options
        if feature in feature_options:

            options = feature_options[feature]

            # ----------------------------------
            # Small categorical feature
            # ----------------------------------
            if len(options) <= 10:

                for i, option in enumerate(options, start=1):
                    print(f"{i}. {option}")

                while True:

                    choice = input("Choose option number: ")

                    try:
                        choice = int(choice)

                        if 1 <= choice <= len(options):
                            user_inputs[feature] = options[choice - 1]
                            break

                    except:
                        pass

                    print("Invalid choice. Try again.")

            # ----------------------------------
            # Large categorical feature
            # ----------------------------------
            else:

                value = input(f"Enter {feature}: ")

                # Optional: match ignoring case
                for option in options:

                    if option.lower() == value.lower():

                        value = option
                        break

                user_inputs[feature] = value

        else:

            value = input(f"Enter {feature}: ")
            user_inputs[feature] = value

      return user_inputs

    def build_prediction_input(
        self,
        extracted_features,
        user_inputs
        ):
        """
        Merge extracted features with values
         entered by the user.
         """

        final_input = extracted_features.copy()

        final_input.update(user_inputs)

        return final_input