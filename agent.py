from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder
from feature_engineering import FeatureEngineer
from prediction_preprocessor import PredictionPreprocessor
from input_validator import InputValidator
from llm_assistant import LLMAssistant
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression , LinearRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    root_mean_squared_error
)
from evaluation_metrics import (
    calculate_classification_metrics,
    calculate_regression_metrics
)
from sklearn.tree import DecisionTreeClassifier ,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor
from trust_analyzer import TrustAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


class DataScienceAgent:
    def __init__(self, file_path):

      print("=" * 50)
      print("📂 LOADING DATASET")
      print("=" * 50)

      encodings = [
        "utf-8",
        "latin1",
        "cp1252",
        "ISO-8859-1"
      ]

      for encoding in encodings:

        try:

            self.df = pd.read_csv(file_path, encoding=encoding)

            print(f"✅ Dataset loaded successfully using {encoding} encoding.")

            break

        except UnicodeDecodeError:

            print(f"❌ {encoding} encoding failed.")

      else:

        raise Exception(
            "Unable to read the dataset. Unsupported or corrupted file."
        )

      self.original_df = self.df.copy()
    

    def show_overview(self):
        print(self.df.head())
        print("\nShape:", self.df.shape)
        print("\nColumns:", list(self.df.columns))

    def missing_values(self):
        print("\nMissing Values:")
        print(self.df.isnull().sum())

    def summary(self):
        print("\nSummary Statistics:")
        print(self.df.describe())





    def detect_columns(self):
        numeric_cols = self.df.select_dtypes(include="number").columns.tolist()
        categorical_cols = self.df.select_dtypes(exclude="number").columns.tolist()

        print("\n📊 Numeric Columns:")
        print(numeric_cols)

        print("\n📝 Categorical Columns:")
        print(categorical_cols)
    def data_quality(self):
        print("\n🧹 DATA QUALITY REPORT")

    # Duplicate rows
        duplicates = self.df.duplicated().sum()
        print(f"Duplicate Rows: {duplicates}")

    # Missing values
        missing = self.df.isnull().sum().sum()
        print(f"Total Missing Values: {missing}")

    # Dataset dimensions
        print(f"Rows: {self.df.shape[0]}")
        print(f"Columns: {self.df.shape[1]}")
    
    def visualize_data(self):
        numeric_cols = self.df.select_dtypes(include="number").columns

        for column in numeric_cols:
             plt.figure(figsize=(6,4))
             sns.histplot(self.df[column], kde=True)
             plt.title(f"Distribution of {column}")
             plt.show()

    def correlation_heatmap(self):
        numeric_df = self.df.select_dtypes(include="number")

        plt.figure(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.show()

    def detect_outliers(self):
        numeric_cols = self.df.select_dtypes(include="number").columns

        for column in numeric_cols:
           plt.figure(figsize=(6,4))
           sns.boxplot(x=self.df[column])
           plt.title(f"Boxplot of {column}")
     
           plt.show()
 

    def preprocessing_suggestions(self):
            print("\n🔧 PREPROCESSING SUGGESTIONS")

            missing = self.df.isnull().sum()

            for column, count in missing.items():

               if count > 0:

                  print(f"\nColumn: {column}")
                  print(f"Missing Values: {count}")

                  if self.df[column].dtype == "object":
                     print("Suggestion: Fill missing values using Mode (most frequent value).")

                  else:
                     print("Suggestion: Fill missing values using Median.")  
     


    def clean_data(self):
        print("\n🧹 CLEANING DATASET")

    # Remove duplicate rows
        duplicates = self.df.duplicated().sum()

        if duplicates > 0:
            self.df = self.df.drop_duplicates()
            print(f"✅ Removed {duplicates} duplicate rows.")
        else:
            print("✅ No duplicate rows found.")
    
    # Count missing values
        missing = self.df.isnull().sum()

# Loop through each column
        for column, count in missing.items():

          if count > 0:

      
             
            if is_numeric_dtype(self.df[column]):

                median_value = self.df[column].median()
                self.df[column] = self.df[column].fillna(median_value)

                print(f"✅ Filled missing values in '{column}' using Median.")

            else:

                mode_value = self.df[column].mode()[0]
                self.df[column] = self.df[column].fillna(mode_value)

                print(f"✅ Filled missing values in '{column}' using Mode.")
        self.df.to_csv("cleaned_data.csv", index=False)
        print("✅ Cleaned dataset saved as cleaned_data.csv")

    def feature_engineering_suggestions(self):

        print("\n🧠 FEATURE ENGINEERING SUGGESTIONS")

        for column in self.df.columns:

           if is_numeric_dtype(self.df[column]):

               print(f"\n{column}")
               print("➡️ Recommendation: Feature Scaling may be required.")

           else:

               unique_values = self.df[column].nunique()

               print(f"\n{column}")

               if unique_values <= 2:
                  print("➡️ Recommendation: Label Encoding")

               else:
                  print("➡️ Recommendation: One-Hot Encoding")    
    
    def select_target(self):

      print(self.df.columns)

      print("\n🎯 TARGET COLUMN SELECTION")

      print("Available Columns:")

      for column in self.df.columns:
        print("-", column)

      target = input("\nEnter the target column: ")

      self.target = target
      self.target_column = target

      print(f"\n✅ Target Column Selected: {target}")  



    def split_features_target(self):

      print("\n📂 SPLITTING FEATURES AND TARGET")

    # Create Features (X) and Target (y)
      self.X = self.df.drop(columns=[self.target])
      self.y = self.df[self.target]

    # Identifier columns to remove
      identifier_columns = [
        "id",
        "name",
        "email",
        "phone",
        "mobile",
        "employee_id",
        "student_id",
        "roll_no"
      ]

    # Remove identifier columns if present
      columns_to_drop = []

      for column in self.X.columns:
        if column.lower() in identifier_columns:
            columns_to_drop.append(column)

      if columns_to_drop:
        self.X = self.X.drop(columns=columns_to_drop)

        print("\n⏭️ Removed Identifier Columns:")
        for column in columns_to_drop:
            print(f"   - {column}")

      print(f"\n✅ Features Shape: {self.X.shape}")
      print(f"✅ Target Shape: {self.y.shape}")



    def detect_problem_type(self):

        print("\n🧠 DETECTING PROBLEM TYPE")

        unique_values = self.y.nunique()
        total_rows = len(self.y)
        unique_ratio = unique_values / total_rows

        print(f"Target Column : {self.target}")
        print(f"Data Type     : {self.y.dtype}")
        print(f"Unique Values : {unique_values}")
        print(f"Total Rows    : {total_rows}")
        print(f"Unique Ratio  : {unique_ratio:.2f}")

        if not is_numeric_dtype(self.y):
           self.problem_type = "Classification"

        elif unique_ratio < 0.05:
            self.problem_type = "Classification"

        else:
            self.problem_type = "Regression"

        print(f"\n✅ Detected Problem Type: {self.problem_type}")
    
    def train_test_split_data(self):

        print("\n✂️ SPLITTING TRAINING AND TESTING DATA")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        self.X,
        self.y,
        test_size=0.2,
        random_state=42
        )

        print(f"✅ Training Features : {self.X_train.shape}")
        print(f"✅ Testing Features  : {self.X_test.shape}")
        print(f"✅ Training Target   : {self.y_train.shape}")
        print(f"✅ Testing Target    : {self.y_test.shape}")




    def train_model(self):

      print("\n🤖 TRAINING MODELS")

      if self.problem_type == "Classification":

        print("\n🤖 TRAINING CLASSIFICATION MODELS")

        models = {}

    # Logistic Regression
        logistic_model = LogisticRegression(max_iter=1000)
        logistic_model.fit(self.X_train, self.y_train)
        logistic_predictions = logistic_model.predict(self.X_test)

  
        metrics = calculate_classification_metrics(
          self.y_test,
          logistic_predictions
)

        models["Logistic Regression"] = (
          logistic_model,
          metrics
)

        print(f"Logistic Regression : {metrics['accuracy']:.4f}")


    # Decision Tree
        decision_tree = DecisionTreeClassifier(random_state=42)
        decision_tree.fit(self.X_train, self.y_train)
        decision_predictions = decision_tree.predict(self.X_test)
        metrics = calculate_classification_metrics(
          self.y_test,
          decision_predictions
)

        models["Decision Tree"] = (
          decision_tree,
          metrics
)
        print(f"✅ Decision Tree Accuracy : {metrics['accuracy']:.4f}")

    # Random Forest
        random_forest = RandomForestClassifier(random_state=42)
        random_forest.fit(self.X_train, self.y_train)
        random_predictions = random_forest.predict(self.X_test)
        metrics = calculate_classification_metrics(
         self.y_test,
         random_predictions
)

        models["Random Forest"] = (
         random_forest,
          metrics
)

        print(f"✅ Random Forest Accuracy : {metrics['accuracy']:.4f}")

    # Best Model
        best_model_name = max(
           models,
           key=lambda x: models[x][1]["accuracy"]
)
        best_model, best_metrics = models[best_model_name]

        self.best_model = best_model
        self.best_model_name = best_model_name
        self.best_metrics = best_metrics
        self.best_score = best_metrics["accuracy"]
       

        print("\n🏆 BEST MODEL")
        print(f"Model : {best_model_name}")
        print(f"Accuracy  : {best_metrics['accuracy']:.4f}")
        print(f"Precision : {best_metrics['precision']:.4f}")
        print(f"Recall    : {best_metrics['recall']:.4f}")
        print(f"F1 Score  : {best_metrics['f1']:.4f}")
        # Save Best Model
        joblib.dump(
          self.best_model,
         "best_model.pkl"
         )

        print("💾 Best model saved as best_model.pkl")
        


      else:

        print("\n🤖 TRAINING REGRESSION MODELS")

        models = {}

    
      

        linear_model = LinearRegression()

        linear_model.fit(self.X_train, self.y_train)

        linear_predictions = linear_model.predict(self.X_test)

        metrics = calculate_regression_metrics(
          self.y_test,
          linear_predictions
)

        models["Linear Regression"] = (
         linear_model,
         metrics
)

        print(f"Linear Regression R² : {metrics['r2']:.4f}")

    # Decision Tree
        decision_tree = DecisionTreeRegressor(random_state=42)
        decision_tree.fit(self.X_train, self.y_train)
        decision_predictions = decision_tree.predict(self.X_test)
        metrics = calculate_regression_metrics(self.y_test, decision_predictions)
        models["Decision Tree"] = (decision_tree, metrics)

        print(f" Decison Tree R² : {metrics['r2']:.4f}")

    # Random Forest
        random_forest = RandomForestRegressor(random_state=42)
        random_forest.fit(self.X_train, self.y_train)
        random_predictions = random_forest.predict(self.X_test)
        metrics = calculate_regression_metrics(self.y_test, random_predictions)
        models["Random Forest"] = (random_forest, metrics)

        print(f" Random Forest R²  : {metrics['r2']:.4f}")

    # Best Model
        best_model_name = max(
         models,
         key=lambda x: models[x][1]["r2"]
)
        best_model, best_metrics = models[best_model_name]

        self.best_model = best_model
        self.best_model_name = best_model_name
        self.best_metrics = best_metrics
        self.best_score = best_metrics["r2"]

        print("\n🏆 BEST MODEL")
        print(f"Model : {best_model_name}")
        print(f"R² Score : {best_metrics['r2']:.4f}")
        print(f"MAE      : {best_metrics['mae']:.4f}")
        print(f"RMSE     : {best_metrics['rmse']:.4f}")
       
        # Save Best Model
        joblib.dump(
          self.best_model,
         "best_model.pkl"
         )

        print("💾 Best model saved as best_model.pkl")
        

    

    

    def generate_report(self):

       print("\n" + "="*50)
       print("📋 AI MODEL REPORT")
       print("="*50)

       print(f"Problem Type : {self.problem_type}")
       print(f"Best Model : {self.best_model_name}")

       if self.problem_type == "Classification":

         print(f"Accuracy  : {self.best_metrics['accuracy']:.4f}")
         print(f"Precision : {self.best_metrics['precision']:.4f}")
         print(f"Recall    : {self.best_metrics['recall']:.4f}")
         print(f"F1 Score  : {self.best_metrics['f1']:.4f}")

         print("\nInterpretation:")

         if self.best_score >= 0.90:
            print("Excellent model performance.")

         elif self.best_score >= 0.75:
            print("Good model performance.")

         elif self.best_score >= 0.60:
            print("Average model performance.")

         else:
            print("Poor model performance.")

       else:

          print(f"R² Score : {self.best_metrics['r2']:.4f}")
          print(f"MAE      : {self.best_metrics['mae']:.4f}")
          print(f"RMSE     : {self.best_metrics['rmse']:.4f}")

          print("\nInterpretation:")

          if self.best_score >= 0.90:
            print("Excellent regression performance.")

          elif self.best_score >= 0.75:
            print("Good regression performance.")

          elif self.best_score >= 0.50:
            print("Average regression performance.")

          elif self.best_score >= 0:
            print("Weak regression performance.")

          else:
            print("Poor model performance.")
            print("The model performs worse than predicting the average target value.")

       print("\nRecommendation:")

       if self.problem_type == "Classification":

          if self.best_score >= 0.75:
            print("This model is suitable for prediction.")
          else:
            print("Improve preprocessing or collect more data.")

       else:

          if self.best_score >= 0.50:
            print("This model can be used for prediction.")
          else:
            print("Collect more data or improve feature engineering.")

       print("="*50)

       
    
    def feature_importance(self):

     print("\n📊 FEATURE IMPORTANCE")

     if hasattr(self.best_model, "feature_importances_"):

        importance = self.best_model.feature_importances_

        feature_importance = pd.DataFrame({
            "Feature": self.X_train.columns,
            "Importance": importance
        })

        feature_importance = feature_importance.sort_values(
            by="Importance",
            ascending=False
        )

        print(feature_importance)

        self.feature_importance_df = feature_importance

        # ---------------------------------------
        # Top ORIGINAL features for LLM
        # ---------------------------------------
        self.important_features = []
        used_original = set()

        engineered_suffixes = [
            "_day",
            "_month",
            "_year",
            "_days_since_min",
            "_char_count",
            "_word_count",
            "_length",
            "_freq_enc"
        ]

        for feature in feature_importance["Feature"]:

            original = feature

            # ---------------------------------------
            # Reverse One-Hot Encoding
            # ---------------------------------------
            for col, encoder in self.feature_engineer.encoders.items():

                if encoder["type"] == "onehot":

                    if feature.startswith(col + "_"):

                        original = col
                        break

                elif encoder["type"] == "frequency":

                    if feature == f"{col}_freq_enc":

                        original = col
                        break

            # ---------------------------------------
            # Reverse Engineered Features
            # ---------------------------------------
            for suffix in engineered_suffixes:

                if original.endswith(suffix):

                    original = original[:-len(suffix)]
                    break

            # ---------------------------------------
            # Ignore interaction features
            # ---------------------------------------
            if "_x_" in original:
                continue

            # ---------------------------------------
            # Keep unique original features only
            # ---------------------------------------
            if original not in used_original:

                self.important_features.append(original)
                used_original.add(original)

            if len(self.important_features) == 10:
                break

        print("\n✅ Top Original Features Used By LLM:")
        print(self.important_features)

     else:

        print("⚠️ Feature importance is not available for this model.")

        self.feature_importance_df = None
        self.important_features = []


   


    """def predict_new_data(self):

      print("\n🤖 PREDICTION ON NEW DATA")

      new_data = {}

      for column in self.original_columns:

        value = input(f"Enter value for {column}: ")

        try:
            value = float(value)
        except:
            pass

        new_data[column] = value

      new_df = pd.DataFrame([new_data])

      prediction = self.best_model.predict(new_df)

      print("\n===================================")
      print("🎯 PREDICTION RESULT")
      print("===================================")

      if self.problem_type == "Classification":
        print(f"Predicted Class : {prediction[0]}")

      else:
        print(f"Predicted Value : {prediction[0]:.2f}")

    print("===================================")"""


    def get_feature_options(self):
      """
    Get possible values for categorical features from
    the FeatureEngineer encoders.
    """

      feature_options = {}

      for feature, encoder in self.encoders.items():

        if encoder["type"] == "onehot":

            values = []

            for column in encoder["categories"]:

                # Example:
                # sex_Male -> Male
                # cp_0 -> 0

                value = column.replace(feature + "_", "")

                values.append(value)

            feature_options[feature] = values

        elif encoder["type"] == "frequency":

            feature_options[feature] = list(
                encoder["map"].keys()
            )

      return feature_options

    def predict_with_llm(self):

      print("\n🤖 AI Prediction Assistant")

      user_prompt = input("\nDescribe your data:\n")

      assistant = LLMAssistant()

    # Extract features
      extracted = assistant.extract_features(
        user_prompt,
        self.important_features
      )

      print("\nExtracted Features:")
      print(extracted)

    # Find missing important features
      missing = assistant.find_missing_important_features(
        extracted,
        self.important_features
      )

      if missing:

        options = self.get_feature_options()

        answers = assistant.ask_missing_features(
            missing,
            options
        )

        extracted = assistant.build_prediction_input(
            extracted,
            answers
        )

        validator = InputValidator(
        self.feature_engineer
        )

        extracted = validator.validate(extracted)

      

      print("\nFinal Input")
      print("\n📋 User Inputs\n")

      for key, value in extracted.items():

          print(f"{key:<25}: {value}")

      # -----------------------------------
      # Convert to DataFrame
      # -----------------------------------
    

      # -----------------------------------
      # Preprocess exactly like training
      # -----------------------------------
      preprocessor = PredictionPreprocessor(
        self.feature_engineer,
        self.feature_engineer.final_feature_order
    )

      processed = preprocessor.transform(extracted)

      # -----------------------------------
      # Predict
      # -----------------------------------
      prediction = self.best_model.predict(processed)

      print("\n" + "=" * 55)
      print("🤖 AI PREDICTION REPORT")
      print("=" * 55)

      print(f"🏆 Model Used      : {self.best_model_name}")
      print(f"🎯 Target Column   : {self.target_column}")

      if self.problem_type.lower() == "classification":

       print(f"✅ Predicted Class : {prediction[0]}")

      if hasattr(self.best_model, "predict_proba"):

        probability = self.best_model.predict_proba(processed)

        confidence = max(probability[0]) * 100

        print(f"📊 Confidence      : {confidence:.2f}%")

      else:

        print(f"📈 Predicted Value : {prediction[0]}")

      print("=" * 55)


    

    def run_analysis(self):
       self.show_overview()
       self.missing_values()
       self.summary()
       self.detect_columns()
       self.data_quality()
       

       trust = TrustAnalyzer(self)
       trust.run_analysis() 
       
       self.preprocessing_suggestions()
       self.feature_engineering_suggestions()
       self.clean_data()
       
     
       
       self.select_target()

       feature_engineer = FeatureEngineer(
       df=self.df,
       target_column=self.target,
       scaling="standard",
       create_interactions=True,
       verbose=True      # change to False later if you don't want logs
)

       self.df = feature_engineer.run_feature_engineering()

       print("\n========== FEATURE ENGINEER ==========")

       print("\nEncoders:")
       print(feature_engineer.encoders)

       print("\nScaler:")
       print(feature_engineer.scaler)

       print("\nColumn Types:")
       print(feature_engineer.column_types)

       print("\nDropped Columns:")
       print(feature_engineer.dropped_columns)

       print("\n========== DEBUG ==========")

       print("\nScaled Columns:")
       print(feature_engineer.scaled_columns)

       print("\nFinal Feature Order:")
       print(feature_engineer.final_feature_order)

       print("\nScaler Feature Names:")
       print(feature_engineer.scaler.feature_names_in_)

      

       print("======================================\n")
       # Save encoders for later
       self.feature_engineer = feature_engineer
       self.encoders = feature_engineer.encoders
       self.interaction_columns = feature_engineer.interaction_columns
       self.scaled_columns = feature_engineer.scaled_columns

       feature_engineer.summary()


       self.split_features_target()
       self.detect_problem_type()

       self.train_test_split_data()
       print("\n========== TRAINING COLUMNS ==========")
       print(self.X_train.columns.tolist())
       print("======================================")
       
       self.train_model()

       self.generate_report()
       self.feature_importance()
       
       self.predict_with_llm()
    

            
agent = DataScienceAgent("datasets/Sample - Superstore.csv")

agent.run_analysis()