


from __future__ import annotations

import re
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder


class FeatureEngineer:
    def __init__(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        id_columns: Optional[List[str]] = None,
        date_columns: Optional[List[str]] = None,
        scaling: str = "standard",       # "standard", "minmax", or None
        max_onehot_cardinality: int = 15,
        low_variance_threshold: float = 1e-4,
        high_correlation_threshold: float = 0.95,
        max_missing_ratio: float = 0.9,
        drop_duplicate_columns: bool = True,
        drop_constant_columns: bool = True,
        process_text: bool = True,
        create_interactions: bool = False,
        verbose: bool = True,
    ):
        self.df = df.copy()
        self.target_column = target_column
        self.id_columns_hint = id_columns or []
        self.date_columns_hint = date_columns or []
        self.scaling = scaling
        self.max_onehot_cardinality = max_onehot_cardinality
        self.low_variance_threshold = low_variance_threshold
        self.high_correlation_threshold = high_correlation_threshold
        self.max_missing_ratio = max_missing_ratio
        self.drop_duplicate_columns = drop_duplicate_columns
        self.drop_constant_columns = drop_constant_columns
        self.process_text = process_text
        self.create_interactions = create_interactions
        self.verbose = verbose

        # populated as the pipeline runs
        self.column_types: Dict[str, List[str]] = {}
        self.log: List[str] = []
        self.dropped_columns: List[str] = []
        self.encoders: Dict[str, Any] = {}
        self.scaler = None
        self.scaled_columns = []
        
    

         # ==================================================
        self.numeric_medians: Dict[str, float] = {}
        self.categorical_modes: Dict[str, Any] = {}
        self.date_min_values: Dict[str, pd.Timestamp] = {}

    # Numeric columns that were scaled
        self.numeric_columns_to_scale: List[str] = []

    # Stores interaction feature pairs
        self.interaction_columns: List[tuple] = []

    # Final feature order expected by the model
        self.final_feature_order: List[str] = []

    # ------------------------------------------------------------------ #
    # utilities
    # ------------------------------------------------------------------ #
    def _say(self, msg: str):
        self.log.append(msg)
        if self.verbose:
            print(f"[FeatureEngineer] {msg}")

    def _feature_cols(self):
        """All columns except the target."""
        return [c for c in self.df.columns if c != self.target_column]

    # ------------------------------------------------------------------ #
    # 1. detect column types
    # ------------------------------------------------------------------ #
    def detect_column_types(self) -> Dict[str, List[str]]:
        df = self.df
        n_rows = len(df)

        id_cols, date_cols, numeric_cols, categorical_cols, text_cols = [], [], [], [], []

        id_like_pattern = re.compile(r"(^id$|_id$|^id_|uuid|guid|index)", re.IGNORECASE)

        for col in self._feature_cols():
            series = df[col]

            # explicit hints take priority
            if col in self.id_columns_hint:
                id_cols.append(col)
                continue
            if col in self.date_columns_hint:
                date_cols.append(col)
                continue

            is_stringlike = pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)

            # datetime dtype or parseable datetime-like strings
            if pd.api.types.is_datetime64_any_dtype(series):
                date_cols.append(col)
                continue
            if is_stringlike:
                sample = series.dropna().astype(str).head(20)
                if len(sample) > 0:
                    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                    if parsed.notna().mean() > 0.8:
                        date_cols.append(col)
                        continue

            # identifier-like: name pattern OR (near-)unique object/int column
            
            
            # -----------------------------
# Identifier Detection
# -----------------------------

            nunique = series.nunique(dropna=True)
            unique_ratio = nunique / max(n_rows, 1)

            column_name = col.lower().strip()

            if (
              column_name == "id"
               or column_name.endswith(" id")
               or column_name.endswith("_id")
               or "uuid" in column_name
                       or "guid" in column_name
               ):
             id_cols.append(col)
             continue

# Detect completely unique string identifiers
            if is_stringlike and unique_ratio > 0.98:
               id_cols.append(col)
               continue

# Detect nearly unique string columns
            if is_stringlike and unique_ratio > 0.95:
                id_cols.append(col)
                continue

            # numeric
            if pd.api.types.is_numeric_dtype(series):
                numeric_cols.append(col)
                continue

            # categorical vs free text
            if is_stringlike or isinstance(series.dtype, pd.CategoricalDtype):
                avg_len = series.dropna().astype(str).str.len().mean() if series.notna().any() else 0
                if nunique > 50 and avg_len > 30:
                    text_cols.append(col)
                else:
                    categorical_cols.append(col)
                continue

            # fallback
            categorical_cols.append(col)

        self.column_types = {
            "id": id_cols,
            "date": date_cols,
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "text": text_cols,
        }
        self._say(
            f"Detected columns -> id: {len(id_cols)}, date: {len(date_cols)}, "
            f"numeric: {len(numeric_cols)}, categorical: {len(categorical_cols)}, text: {len(text_cols)}"
        )
        return self.column_types

    # ------------------------------------------------------------------ #
    # 2. remove identifier columns
    # ------------------------------------------------------------------ #
    def remove_identifier_columns(self):

        id_cols = self.column_types.get("id", [])

        if len(id_cols) == 0:
          self._say("No identifier columns detected.")
          return self.df

    # Drop only columns that still exist
        id_cols = [col for col in id_cols if col in self.df.columns]

        self.df.drop(columns=id_cols, inplace=True)

        self.dropped_columns.extend(id_cols)

        self._say(f"Removed {len(id_cols)} identifier columns: {id_cols}")

        return self.df

    # ------------------------------------------------------------------ #
    # 2b. remove other "unuseful" columns: constant, near-empty, duplicate
    # ------------------------------------------------------------------ #
    def remove_unuseful_columns(self):
        """
        Drops columns that carry little or no predictive signal, beyond
        pure identifiers:
          - constant / single-value columns (zero information)
          - columns with a missing-value ratio above `max_missing_ratio`
          - exact duplicate columns (same values as another column, incl.
            duplicates that only differ by column name)
        """
        feature_cols = [c for c in self._feature_cols() if c in self.df.columns]
        to_drop = []
        reasons: Dict[str, str] = {}

        # --- constant / single-value columns ---
        if self.drop_constant_columns:
            for col in feature_cols:
                if col in to_drop:
                    continue
                if self.df[col].nunique(dropna=False) <= 1:
                    to_drop.append(col)
                    reasons[col] = "constant (only one distinct value)"

        # --- near-empty columns ---
        n_rows = len(self.df)
        for col in feature_cols:
            if col in to_drop:
                continue
            missing_ratio = self.df[col].isna().mean() if n_rows else 0
            if missing_ratio > self.max_missing_ratio:
                to_drop.append(col)
                reasons[col] = f"{missing_ratio:.0%} missing values"

        # --- exact duplicate columns (keep the first occurrence) ---
        if self.drop_duplicate_columns:
            remaining = [c for c in feature_cols if c not in to_drop]
            seen: Dict[str, str] = {}
            for col in remaining:
                # a hashable fingerprint of the column's values
                fingerprint = pd.util.hash_pandas_object(self.df[col], index=False).values.tobytes()
                if fingerprint in seen:
                    to_drop.append(col)
                    reasons[col] = f"duplicate of column '{seen[fingerprint]}'"
                else:
                    seen[fingerprint] = col

        if to_drop:
            self.df.drop(columns=to_drop, inplace=True)
            self.dropped_columns.extend(to_drop)
            # keep column_types dicts in sync
            for key in self.column_types:
                self.column_types[key] = [c for c in self.column_types[key] if c not in to_drop]
            for col in to_drop:
                self._say(f"Removed column '{col}' ({reasons[col]})")
        else:
            self._say("No constant, near-empty, or duplicate columns found")
        return self.df

    # ------------------------------------------------------------------ #
    # 3. process dates
    # ------------------------------------------------------------------ #
    def process_dates(self):
       date_cols = self.column_types.get("date", [])

       for col in date_cols:

        if col not in self.df.columns:
            continue

        parsed = pd.to_datetime(self.df[col], errors="coerce", format="mixed")

        # Save minimum date during training
        if col not in self.date_min_values:
            self.date_min_values[col] = parsed.min()

        min_date = self.date_min_values[col]

        self.df[f"{col}_year"] = parsed.dt.year
        self.df[f"{col}_month"] = parsed.dt.month
        self.df[f"{col}_day"] = parsed.dt.day
        self.df[f"{col}_dayofweek"] = parsed.dt.dayofweek
        self.df[f"{col}_is_weekend"] = parsed.dt.dayofweek.isin([5, 6]).astype(int)
        self.df[f"{col}_quarter"] = parsed.dt.quarter

        self.df[f"{col}_days_since_min"] = (
            parsed - min_date
        ).dt.days

        self.df.drop(columns=[col], inplace=True)
        self.dropped_columns.append(col)

       if date_cols:
        new_numeric = [
            c for c in self.df.columns
            if any(c.startswith(f"{d}_") for d in date_cols)
        ]

        self.column_types["numeric"] = list(
            set(self.column_types.get("numeric", []) + new_numeric)
        )

        self._say(
            f"Expanded {len(date_cols)} date column(s) into {len(new_numeric)} numeric features"
        )

       return self.df
    # ------------------------------------------------------------------ #
    # 3b. process free-text columns
    # ------------------------------------------------------------------ #
    def process_text_columns(self):
        """
        Free-text columns (long, high-cardinality strings) aren't useful
        as raw categoricals and were previously just left untouched.
        Either extract simple statistics from them (length, word count)
        or drop them, then remove the original text column either way.
        """
        text_cols = [c for c in self.column_types.get("text", []) if c in self.df.columns]
        if not text_cols:
            return self.df

        created = []
        for col in text_cols:
            text = self.df[col].astype(str).fillna("")
            if self.process_text:
                self.df[f"{col}_char_count"] = text.str.len()
                self.df[f"{col}_word_count"] = text.str.split().apply(len)
                created.extend([f"{col}_char_count", f"{col}_word_count"])
            self.df.drop(columns=[col], inplace=True)
            self.dropped_columns.append(col)

        self.column_types["numeric"] = list(set(self.column_types.get("numeric", []) + created))
        if self.process_text:
            self._say(f"Converted {len(text_cols)} free-text column(s) into length/word-count features")
        else:
            self._say(f"Dropped {len(text_cols)} free-text column(s) (process_text=False)")
        return self.df

    # ------------------------------------------------------------------ #
    # 4. handle missing values
    def handle_missing_values(self):

      numeric_cols = [c for c in self.column_types.get("numeric", []) if c in self.df.columns]
      categorical_cols = [c for c in self.column_types.get("categorical", []) if c in self.df.columns]

      for col in numeric_cols:

        if self.df[col].isna().any():

            missing_flag = f"{col}_was_missing"
            self.df[missing_flag] = self.df[col].isna().astype(int)

            # Learn median during training
            if col not in self.numeric_medians:
                self.numeric_medians[col] = self.df[col].median()

            self.df[col] = self.df[col].fillna(self.numeric_medians[col])

            self.column_types["numeric"].append(missing_flag)

      for col in categorical_cols:

        if self.df[col].isna().any():

            if col not in self.categorical_modes:
                mode_series = self.df[col].mode(dropna=True)
                self.categorical_modes[col] = (
                    mode_series.iloc[0] if not mode_series.empty else "Missing"
                )

            self.df[col] = self.df[col].fillna(self.categorical_modes[col])

      self._say(
        "Handled missing values (median for numeric, mode for categorical, +missing-flags)"
      )

      return self.df
    # 5. encode categorical columns
    # ------------------------------------------------------------------ #
    def encode_categorical_columns(self):

      categorical_cols = [
        c for c in self.column_types.get("categorical", [])
        if c in self.df.columns
      ]

      encoded_numeric = []

      for col in categorical_cols:

        nunique = self.df[col].nunique(dropna=True)

        # -----------------------------
        # One Hot Encoding
        # -----------------------------
        if nunique <= self.max_onehot_cardinality:

            dummies = pd.get_dummies(
                self.df[col],
                prefix=col,
                dummy_na=False,
                dtype=int
            )

            self.df = pd.concat(
                [self.df.drop(columns=[col]), dummies],
                axis=1
            )

            # Save categories for prediction
            self.encoders[col] = {
                "type": "onehot",
                "categories": list(dummies.columns)
            }

            encoded_numeric.extend(dummies.columns.tolist())

        # -----------------------------
        # Frequency Encoding
        # -----------------------------
        else:

            freq_map = self.df[col].value_counts(normalize=True)

            new_col = f"{col}_freq_enc"

            self.df[new_col] = (
                self.df[col]
                .map(freq_map)
                .fillna(0)
            )

            self.df.drop(columns=[col], inplace=True)

            self.encoders[col] = {
                "type": "frequency",
                "map": freq_map.to_dict()
            }

            encoded_numeric.append(new_col)

      self.column_types["numeric"] = list(
        set(
            self.column_types.get("numeric", [])
            + encoded_numeric
        )
      )

      self._say(
        f"Encoded {len(categorical_cols)} categorical column(s)"
      )

      return self.df

    # ------------------------------------------------------------------ #
    # 6. scale numeric columns
    # ------------------------------------------------------------------ #
    def scale_numeric_columns(self):

      if self.scaling is None:
        self._say("Skipping scaling (scaling=None)")
        return self.df

      numeric_cols = [
        c for c in self.column_types.get("numeric", [])
        if c in self.df.columns
     ]
      
     

    # Don't scale binary columns
      numeric_cols = [
        c for c in numeric_cols
        if self.df[c].nunique() > 2
     ]
      
      self.scaled_columns = numeric_cols.copy()

      if not numeric_cols:
        return self.df

    # Save which columns were scaled
      self.numeric_columns_to_scale = numeric_cols.copy()

    # Create scaler only once
      if self.scaler is None:

        if self.scaling == "standard":
            self.scaler = StandardScaler()

        elif self.scaling == "minmax":
            self.scaler = MinMaxScaler()

        else:
            raise ValueError(
                "scaling must be 'standard', 'minmax', or None"
            )

        self.df[numeric_cols] = self.scaler.fit_transform(
            self.df[numeric_cols]
        )

      else:
        # Reuse fitted scaler during prediction
        self.df[numeric_cols] = self.scaler.transform(
            self.df[numeric_cols]
        )

      self._say(
        f"Scaled {len(numeric_cols)} numeric column(s) using {self.scaling} scaling"
    )

      return self.df

    # ------------------------------------------------------------------ #
    # 7. remove low-variance features
    # ------------------------------------------------------------------ #
    def remove_low_variance_features(self):
        numeric_cols = [c for c in self.column_types.get("numeric", []) if c in self.df.columns]
        to_drop = [c for c in numeric_cols if self.df[c].var(ddof=0) < self.low_variance_threshold]
        if to_drop:
            self.df.drop(columns=to_drop, inplace=True)
            self.dropped_columns.extend(to_drop)
            self._say(f"Removed {len(to_drop)} low-variance column(s): {to_drop}")
        return self.df

    # ------------------------------------------------------------------ #
    # 8. remove highly-correlated features
    # ------------------------------------------------------------------ #
    def remove_high_correlation_features(self):
        numeric_cols = [c for c in self.column_types.get("numeric", []) if c in self.df.columns]
        if len(numeric_cols) < 2:
            return self.df

        corr_matrix = self.df[numeric_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > self.high_correlation_threshold)]

        if to_drop:
            self.df.drop(columns=to_drop, inplace=True)
            self.dropped_columns.extend(to_drop)
            self._say(f"Removed {len(to_drop)} highly-correlated column(s) (> {self.high_correlation_threshold}): {to_drop}")
        return self.df

    # ------------------------------------------------------------------ #
    # 9. (optional) interaction features
    # ------------------------------------------------------------------ #
    def create_interaction_features(self, top_k: int = 5):

      if not self.create_interactions:
        return self.df

      numeric_cols = [
        c for c in self.column_types.get("numeric", [])
        if c in self.df.columns
      ]

    # -----------------------------
    # TRAINING
    # -----------------------------
      if not self.interaction_columns:

        if (
            self.target_column
            and self.target_column in self.df.columns
            and pd.api.types.is_numeric_dtype(self.df[self.target_column])
        ):

            corr = (
                self.df[numeric_cols + [self.target_column]]
                .corr()[self.target_column]
                .abs()
            )

            top_cols = (
                corr.drop(self.target_column)
                .sort_values(ascending=False)
                .head(top_k)
                .index
                .tolist()
            )

        else:
            top_cols = numeric_cols[:top_k]

        # Save interaction pairs
        for i in range(len(top_cols)):
            for j in range(i + 1, len(top_cols)):
                self.interaction_columns.append(
                    (top_cols[i], top_cols[j])
                )

    # -----------------------------
    # CREATE FEATURES
    # -----------------------------
      created = []

      for col1, col2 in self.interaction_columns:

        if (
            col1 in self.df.columns
            and col2 in self.df.columns
        ):

            feature_name = f"{col1}_x_{col2}"

            self.df[feature_name] = (
                self.df[col1] * self.df[col2]
            )

            created.append(feature_name)

       
      self.column_types["numeric"] = list(
        set(
            self.column_types.get("numeric", [])
            + created
        )
      )

      self._say(
        f"Created {len(created)} interaction feature(s)"
    )

      return self.df

    # ------------------------------------------------------------------ #
    # full pipeline
    # ------------------------------------------------------------------ #
    def run_feature_engineering(self) -> pd.DataFrame:

      self._say("=== Starting automated feature engineering pipeline ===")
  
      self.detect_column_types()
      self.remove_identifier_columns()
      self.remove_unuseful_columns()
      self.process_dates()
      self.process_text_columns()
      self.handle_missing_values()
      self.encode_categorical_columns()
      self.remove_low_variance_features()
      self.remove_high_correlation_features()

    # ---------------------------------------
    # Save original feature columns
    # ---------------------------------------
      self.original_feature_columns = [
        c for c in self.df.columns
        if c != self.target_column
    ]

      self.create_interaction_features()
      self.scale_numeric_columns()

    # ---------------------------------------
    # Save final feature order
    # ---------------------------------------
      self.final_feature_order = [
        c for c in self.df.columns
        if c != self.target_column
    ]

      self._say(f"=== Done. Final shape: {self.df.shape} ===")

      return self.df
    
      # ------------------------------------------------------------------ #
    def summary(self):
        print("\n--- Feature Engineering Summary ---")
        print(f"Final shape: {self.df.shape}")
        print(f"Dropped columns ({len(self.dropped_columns)}): {self.dropped_columns}")
        print(f"Encoders fitted for: {list(self.encoders.keys())}")
        print(f"Scaler used: {self.scaling}")
        print("Steps log:")
        for line in self.log:
            print(f"  - {line}")



