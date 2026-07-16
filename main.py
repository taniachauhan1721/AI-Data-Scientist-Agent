
print("hello, data scientist")
rint("\n" + "=" * 60)
        print("🔍 CHECKING DATA BEFORE TRAINING")
        print("=" * 60)

        print("\nRemaining Object Columns:")
        print(self.X_train.select_dtypes(include="object").columns.tolist())

        print("\nRemaining Missing Values:")
        print(self.X_train.isnull().sum().sum())

        print(f"\nTraining Shape : {self.X_train.shape}")


 self.detect_columns()
       self.data_quality()
       self.visualize_data()
       self.correlation_heatmap()
       self.detect_outliers()

       trust = TrustAnalyzer(self)