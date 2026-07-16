import pandas as pd


class PredictionPreprocessor:

    def __init__(
        self,
        feature_engineer,
        training_columns
    ):

        self.feature_engineer = feature_engineer
        self.training_columns = training_columns

    def create_dataframe(self, input_dict):

       return pd.DataFrame([input_dict])
    


    def encode_categorical_columns(self, df):

     for feature, encoder in self.feature_engineer.encoders.items():

        # Skip if user didn't provide this feature
        if feature not in df.columns:
            continue

        # -------------------------
        # One-Hot Encoding
        # -------------------------
        if encoder["type"] == "onehot":

            value = str(df.at[0, feature])

            # Remove original column
            df.drop(columns=[feature], inplace=True)

            # Create all dummy columns
            for column in encoder["categories"]:

                df[column] = 0

            dummy_name = f"{feature}_{value}"

            if dummy_name in df.columns:
                df[dummy_name] = 1

        # -------------------------
        # Frequency Encoding
        # -------------------------
        elif encoder["type"] == "frequency":

            value = df.at[0, feature]

            freq = encoder["map"].get(value, 0)

            df[f"{feature}_freq_enc"] = freq

            df.drop(columns=[feature], inplace=True)

     return df
    

    
    def scale_numeric_columns(self, df):

      if self.feature_engineer.scaler is None:
        return df

    # Use exactly the columns the scaler was fitted on
      scaled_cols = list(
        self.feature_engineer.scaler.feature_names_in_
    )

    # Ensure every required column exists
      for col in scaled_cols:

        if col not in df.columns:
            df[col] = 0

      df[scaled_cols] = self.feature_engineer.scaler.transform(
        df[scaled_cols]
    )

      return df

    def create_interaction_features(self, df):

      for col1, col2 in self.feature_engineer.interaction_columns:

        if col1 in df.columns and col2 in df.columns:

            feature_name = f"{col1}_x_{col2}"

            df[feature_name] = (
                df[col1] * df[col2]
            )

      return df
    

    def add_missing_columns(self, df):

      for column in self.training_columns:

        if column not in df.columns:

            df[column] = 0

      return df
    
    def reorder_columns(self, df):

      df = df[self.training_columns]

      return df
    


    
    def transform(self, input_dict):

      df = self.create_dataframe(input_dict)

      # Convert numeric columns to numbers
      for col in self.feature_engineer.column_types["numeric"]:

       if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="raise"
        )

      df=self.encode_categorical_columns(df)
      df = self.add_missing_columns(df)
      df = self.create_interaction_features(df)
      df = self.scale_numeric_columns(df)

      


      df = self.reorder_columns(df)

      return df