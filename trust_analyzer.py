


import pandas as pd
import numpy as np


class TrustAnalyzer:

    def __init__(self, agent):

        self.agent = agent
        self.df = agent.df
        
        

        self.trust_score = 100
        self.warnings = []

    


    def missing_value_analysis(self):

      print("\n" + "=" * 60)
      print("📊 MISSING VALUE ANALYSIS")
      print("=" * 60)

    # Total missing values
      total_missing = self.df.isnull().sum().sum()

    # Total cells in dataset
      total_cells = self.df.shape[0] * self.df.shape[1]

    # Missing percentage
      missing_percentage = (total_missing / total_cells) * 100

      print(f"Total Missing Values : {total_missing}")
      print(f"Missing Percentage   : {missing_percentage:.2f}%")

      print("\nColumn-wise Missing Values")

      missing_columns = self.df.isnull().sum()

      for column, value in missing_columns.items():

       if value > 0:

            percentage = (value / len(self.df)) * 100

            print(f"{column} : {value} ({percentage:.2f}%)")

    # -----------------------------
    # Severity Detection
    # -----------------------------

      if missing_percentage == 0:

        severity = "None"

      elif missing_percentage < 5:

        severity = "Low"

      elif missing_percentage < 15:

        severity = "Moderate"

      else:

        severity = "High"

      print(f"\nSeverity : {severity}")

    # -----------------------------
    # Trust Score
    # -----------------------------

      if severity == "Low":

        self.trust_score -= 5

      elif severity == "Moderate":

        self.trust_score -= 10

      elif severity == "High":

        self.trust_score -= 20

    # -----------------------------
    # Recommendation
    # -----------------------------

      print("\nRecommendation")

      if severity == "None":

        print("✅ No missing values detected.")

      elif severity == "Low":

        print("Fill missing values before training.")

      elif severity == "Moderate":

        print("Use appropriate imputation techniques (Mean, Median or Mode).")

      else:

        print("Large amount of missing data detected.")
        print("Consider collecting more data or removing unreliable columns.")

    # -----------------------------
    # Store Warning
    # -----------------------------

      if severity != "None":

        self.warnings.append(
            f"Missing Values ({severity}) : {missing_percentage:.2f}%"
        )



    def duplicate_analysis(self):

      print("\n" + "=" * 60)
      print("📑 DUPLICATE ANALYSIS")
      print("=" * 60)

    # Total duplicate rows
      duplicate_rows = self.df.duplicated().sum()

    # Duplicate percentage
      duplicate_percentage = (duplicate_rows / len(self.df)) * 100

      print(f"Duplicate Rows        : {duplicate_rows}")
      print(f"Duplicate Percentage  : {duplicate_percentage:.2f}%")

    # -----------------------------
    # Severity Detection
    # -----------------------------

      if duplicate_rows == 0:

        severity = "None"

      elif duplicate_percentage < 5:

        severity = "Low"

      elif duplicate_percentage < 15:

        severity = "Moderate"

      else:

        severity = "High"

      print(f"\nSeverity : {severity}")

    # -----------------------------
    # Trust Score Impact
    # -----------------------------

      if severity == "Low":

        self.trust_score -= 5

      elif severity == "Moderate":

        self.trust_score -= 10

      elif severity == "High":

        self.trust_score -= 20

    # -----------------------------
    # Recommendation
    # -----------------------------

      print("\nRecommendation")

      if severity == "None":

        print("✅ No duplicate rows detected.")

      elif severity == "Low":

        print("Remove duplicate rows before training.")

      elif severity == "Moderate":

        print("Dataset contains a moderate number of duplicates.")
        print("Remove duplicate records before model training.")

      else:

        print("Large number of duplicate rows detected.")
        print("Dataset quality is poor.")
        print("Remove duplicate records before further analysis.")

    # -----------------------------
    # Store Warning
    # -----------------------------

      if severity != "None":

        self.warnings.append(
            f"Duplicate Rows ({severity}) : {duplicate_percentage:.2f}%"
        )


    def consistency_analysis(self):

      print("\n" + "=" * 60)
      print("📝 DATA CONSISTENCY ANALYSIS")
      print("=" * 60)

      inconsistent_columns = 0

    # Check only categorical columns
      categorical_columns = self.df.select_dtypes(include="object").columns

      if len(categorical_columns) == 0:

        print("No categorical columns found.")
        return

      for column in categorical_columns:

        original_values = self.df[column].dropna().astype(str)

        cleaned_values = original_values.str.strip().str.lower()

        # Compare original unique values with cleaned unique values
        if original_values.nunique() != cleaned_values.nunique():

            inconsistent_columns += 1

            print(f"\n⚠ Inconsistent values detected in column: {column}")

            print("\nOriginal Unique Values:")

            for value in sorted(original_values.unique()):
                print(f"   {value}")

            print("\nRecommendation:")
            print("Standardize values before encoding.")

    # -----------------------------------
    # Severity
    # -----------------------------------

      if inconsistent_columns == 0:

        severity = "None"

      elif inconsistent_columns <= 2:

        severity = "Low"

      elif inconsistent_columns <= 5:

        severity = "Moderate"

      else:

        severity = "High"

      print(f"\nSeverity : {severity}")

    # -----------------------------------
    # Trust Score
    # -----------------------------------

      if severity == "Low":

        self.trust_score -= 5

      elif severity == "Moderate":

        self.trust_score -= 10

      elif severity == "High":

        self.trust_score -= 20

    # -----------------------------------
    # Recommendation
    # -----------------------------------

      print("\nRecommendation")

      if severity == "None":

        print("✅ Dataset is consistent.")

      elif severity == "Low":

        print("Standardize inconsistent categories before preprocessing.")

      elif severity == "Moderate":

        print("Several categorical inconsistencies detected.")
        print("Clean categorical values before training.")

      else:

        print("Many inconsistent categorical values detected.")
        print("Dataset should be standardized before analysis.")

    # -----------------------------------
    # Store Warning
    # -----------------------------------

      if severity != "None":

        self.warnings.append(
            f"Data Consistency ({severity})"
        )

    def check_date_formats(self):

      print("\n📅 DATE FORMAT ANALYSIS")

      date_columns_found = False

      for column in self.df.columns:

        if "date" in column.lower():

            date_columns_found = True

            values = self.df[column].dropna().astype(str)

            formats_found = set()

            for value in values:

                value = value.strip()

                if "/" in value:

                    formats_found.add("DD/MM/YYYY or MM/DD/YYYY")

                elif "-" in value:

                    parts = value.split("-")

                    if len(parts[0]) == 4:

                        formats_found.add("YYYY-MM-DD")

                    else:

                        formats_found.add("DD-MM-YYYY")

                elif "," in value:

                    formats_found.add("Month DD, YYYY")

                else:

                    formats_found.add("Unknown Format")

            print(f"\nColumn : {column}")

            print("Formats Found:")

            for fmt in formats_found:

                print(f"• {fmt}")

            if len(formats_found) > 1:

                print("\n⚠ Mixed date formats detected.")

                self.trust_score -= 5

                self.warnings.append(
                    f"Mixed Date Formats ({column})"
                )

                print("Recommendation:")
                print("Convert all dates to one standard format.")

            else:

                print("✅ Date format is consistent.")

      if not date_columns_found:

        print("No date columns detected.")

    def run_analysis(self):

        print("\n" + "=" * 60)
        print("🔍 AI DATASET TRUST ANALYZER")
        print("=" * 60)
        self.missing_value_analysis()
        self.duplicate_analysis()
        self.consistency_analysis()
        self.check_date_formats()