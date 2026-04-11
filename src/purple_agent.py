
"""
Purple Agent - ML Engineering Agent for MLE-Bench

This agent solves ML competitions from MLE-Bench by:
1. Receiving and extracting competition.tar.gz
2. Analyzing the task type (classification, regression, etc.)
3. Using LLM to decide optimal strategy based on data characteristics
4. Preprocessing data with advanced techniques
5. Training multiple ML models and selecting the best
6. Generating submission.csv predictions
"""
import io
import json
import logging
import os
import re
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.svm import SVC, SVR

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM Decision Maker
# ---------------------------------------------------------------------------

class LLMDecisionMaker:
    """
    Uses an LLM to analyze task metadata and recommend the best modeling
    strategy.  Falls back to rule-based logic when no LLM is available.

    Priority order for LLM backends:
      1. Explicit ``llm_fn`` callable (if passed to constructor)
      2. OpenAI API (if ``OPENAI_API_KEY`` env var is set)
      3. Heuristic fallback (rule-based)
    """

    SYSTEM_PROMPT = """\
You are an expert machine-learning engineer acting as an advisor for an
automated ML competition agent.  You will receive a structured description
of a tabular ML competition task.  Your job is to:

1. Confirm or refine the task type (binary_classification /
   multiclass_classification / regression).
2. Recommend the top 3 model families to try, ordered from most to least
   promising for this specific dataset.
3. Suggest preprocessing steps (scaling, encoding, feature engineering).
4. Provide a short justification.

Respond with a valid JSON object only (no markdown fences, no extra text).
The schema must be:
{
  "task_type": "binary_classification" | "multiclass_classification" | "regression",
  "models": ["model_name_1", "model_name_2", "model_name_3"],
  "preprocessing": ["step_1", "step_2"],
  "feature_engineering": ["idea_1", "idea_2"],
  "justification": "short explanation"
}

Available model names (use these exact strings):
  - "lightgbm", "xgboost", "catboost"
  - "gradient_boosting", "random_forest", "extra_trees"
  - "logistic_regression", "ridge", "lasso", "elastic_net"
  - "svm", "knn"
  - "stacking_ensemble", "voting_ensemble"
"""

    # Default OpenAI model
    DEFAULT_MODEL = "gpt-5.4"
    ALLOWED_TASK_TYPES = {
        "binary_classification",
        "multiclass_classification",
        "regression",
    }
    ALLOWED_MODELS = (
        "lightgbm",
        "xgboost",
        "catboost",
        "gradient_boosting",
        "random_forest",
        "extra_trees",
        "logistic_regression",
        "ridge",
        "lasso",
        "elastic_net",
        "svm",
        "knn",
        "stacking_ensemble",
        "voting_ensemble",
    )

    def __init__(
        self,
        llm_fn: Optional[Callable[[str], Optional[str]]] = None,
        openai_api_key: Optional[str] = None,
        openai_model: Optional[str] = None,
        openai_base_url: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        llm_fn : callable(prompt: str) -> str | None
            If provided, this function will be called with the LLM prompt.
            It must return the LLM's text response.  If None or returns
            None, the heuristic fallback is used.
        openai_api_key : str | None
            OpenAI API key.  If not provided, falls back to ``OPENAI_API_KEY``
            environment variable.
        openai_model : str | None
            OpenAI model name.  Defaults to ``gpt-4.1`` (latest available).
        openai_base_url : str | None
            Custom OpenAI base URL (for proxies or compatible APIs).
        """
        self.llm_fn = llm_fn
        self._openai_client: Optional[OpenAI] = None
        self._openai_model = openai_model or self.DEFAULT_MODEL

        if HAS_OPENAI and (openai_api_key or os.environ.get("OPENAI_API_KEY")):
            kwargs: dict[str, Any] = {}
            if openai_api_key:
                kwargs["api_key"] = openai_api_key
            if openai_base_url:
                kwargs["base_url"] = openai_base_url
            try:
                self._openai_client = OpenAI(**kwargs)
                logger.info("OpenAI client initialized (model=%s)", self._openai_model)
            except Exception as exc:
                logger.warning("Failed to create OpenAI client: %s", exc)

    # -- public API ----------------------------------------------------------

    def recommend(self, task_info: dict[str, Any]) -> dict[str, Any]:
        """
        Return a recommendation dict based on task analysis.
        Priority: explicit llm_fn → OpenAI client → heuristic fallback.
        """
        fallback = self._heuristic_recommend(task_info)
        prompt = self._build_prompt(task_info)

        # 1. Try explicit llm_fn
        if self.llm_fn is not None:
            try:
                response = self.llm_fn(prompt)
                if response:
                    parsed = self._parse_llm_response(response)
                    if parsed:
                        normalized = self._normalize_recommendation(
                            parsed, fallback=fallback
                        )
                        logger.info(
                            "LLM (custom fn) recommendation: %s", normalized
                        )
                        return normalized
            except Exception as exc:
                logger.warning("Custom LLM call failed, trying next backend: %s", exc)

        # 2. Try OpenAI
        if self._openai_client is not None:
            try:
                return self._openai_recommend(task_info, fallback=fallback, prompt=prompt)
            except Exception as exc:
                logger.warning("OpenAI call failed, using heuristic fallback: %s", exc)

        # 3. Heuristic fallback
        logger.info("Using heuristic recommendation")
        return fallback

    # -- OpenAI helpers ------------------------------------------------------

    def _openai_recommend(
        self,
        task_info: dict[str, Any],
        fallback: dict[str, Any],
        prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get recommendation from OpenAI API."""
        if prompt is None:
            prompt = self._build_prompt(task_info)
        response = self._openai_client.chat.completions.create(
            model=self._openai_model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content
        parsed = self._parse_llm_response(text)
        if parsed:
            normalized = self._normalize_recommendation(parsed, fallback=fallback)
            logger.info(
                "OpenAI (model=%s) recommendation: %s",
                self._openai_model,
                normalized,
            )
            return normalized
        raise ValueError(f"OpenAI returned non-JSON response: {text[:200]}")

    # -- LLM helpers ---------------------------------------------------------

    @staticmethod
    def _build_prompt(task_info: dict[str, Any]) -> str:
        task_type = task_info.get("task_type", "unknown")
        missing_ratio = task_info.get("missing_ratio", 0.0)
        if missing_ratio is None:
            missing_ratio = 0.0
        lines = [
            "You are helping an AutoML pipeline for tabular competitions.",
            "Return only JSON with the exact schema from the system message.",
            "Prefer robust models that handle missing values and mixed feature types.",
            "",
            f"Task type (inferred): {task_type}",
            f"Target column: {task_info.get('target_column', 'unknown')}",
            f"ID column: {task_info.get('id_column', 'unknown')}",
            f"Number of rows (train): {task_info.get('n_rows', 'unknown')}",
            f"Number of columns: {task_info.get('n_cols', 'unknown')}",
            f"Numeric columns: {task_info.get('n_numeric', 'unknown')}",
            f"Categorical columns: {task_info.get('n_categorical', 'unknown')}",
            f"Missing value ratio: {missing_ratio:.2%}",
            f"Target cardinality: {task_info.get('target_cardinality', 'unknown')}",
            f"Class balance: {task_info.get('class_balance', 'unknown')}",
            "",
            "Constraints:",
            "- Provide 3-5 models sorted best-to-worst.",
            "- Models must come from the allowed model list only.",
            "- Include preprocessing and feature_engineering as concise step names.",
        ]
        desc = task_info.get("description", "")
        if desc:
            lines.append(f"\nCompetition description:\n{desc[:2000]}")
        lines.append("\nRespond with strict JSON only.")
        return "\n".join(lines)

    @staticmethod
    def _parse_llm_response(text: str) -> Optional[dict[str, Any]]:
        """Try to extract a JSON object from LLM response."""
        # Strip markdown fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text.strip())
        text = re.sub(r"\s*```\s*$", "", text.strip())
        # Find first `{` and last `}`
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    @classmethod
    def _is_model_available(cls, model_name: str) -> bool:
        if model_name == "lightgbm":
            return HAS_LIGHTGBM
        if model_name == "xgboost":
            return HAS_XGBOOST
        if model_name == "catboost":
            return HAS_CATBOOST
        return model_name in cls.ALLOWED_MODELS

    @classmethod
    def _default_models_for_task(cls, task_type: str) -> list[str]:
        if task_type == "regression":
            defaults = [
                "lightgbm",
                "catboost",
                "xgboost",
                "extra_trees",
                "random_forest",
                "ridge",
            ]
        elif task_type == "multiclass_classification":
            defaults = [
                "lightgbm",
                "catboost",
                "xgboost",
                "extra_trees",
                "random_forest",
                "gradient_boosting",
            ]
        else:
            defaults = [
                "lightgbm",
                "catboost",
                "xgboost",
                "extra_trees",
                "random_forest",
                "logistic_regression",
            ]
        return [m for m in defaults if cls._is_model_available(m)]

    @staticmethod
    def _clean_string_list(raw: Any) -> list[str]:
        if not isinstance(raw, list):
            return []
        cleaned: list[str] = []
        for item in raw:
            if isinstance(item, str):
                value = item.strip()
                if value:
                    cleaned.append(value)
        return cleaned

    @classmethod
    def _normalize_recommendation(
        cls, recommendation: dict[str, Any], fallback: dict[str, Any]
    ) -> dict[str, Any]:
        fallback_task = fallback.get("task_type", "binary_classification")
        raw_task = recommendation.get("task_type")
        task_type = raw_task if raw_task in cls.ALLOWED_TASK_TYPES else fallback_task

        models: list[str] = []
        for m in cls._clean_string_list(recommendation.get("models")):
            if m in ("stacking_ensemble", "voting_ensemble"):
                continue
            if m in cls.ALLOWED_MODELS and m not in models:
                models.append(m)

        if len(models) < 3:
            for m in cls._default_models_for_task(task_type):
                if m not in models:
                    models.append(m)
                if len(models) >= 5:
                    break

        if not models:
            models = fallback.get("models", ["gradient_boosting", "random_forest"])

        preprocessing = cls._clean_string_list(recommendation.get("preprocessing"))
        if not preprocessing:
            preprocessing = fallback.get(
                "preprocessing", ["median_impute", "label_encoding", "robust_scaling"]
            )

        feature_eng = cls._clean_string_list(recommendation.get("feature_engineering"))
        if not feature_eng:
            feature_eng = fallback.get("feature_engineering", [])

        justification = recommendation.get("justification")
        if not isinstance(justification, str) or not justification.strip():
            justification = fallback.get("justification", "Auto-normalized recommendation.")

        return {
            "task_type": task_type,
            "models": models[:5],
            "preprocessing": preprocessing[:6],
            "feature_engineering": feature_eng[:6],
            "justification": justification.strip(),
        }

    # -- heuristic fallback --------------------------------------------------

    @staticmethod
    def _heuristic_recommend(task_info: dict[str, Any]) -> dict[str, Any]:
        """Rule-based recommendation when LLM is not available."""
        task_type = task_info.get("task_type", "binary_classification")
        n_rows = int(task_info.get("n_rows", 0) or 0)
        n_cols = int(task_info.get("n_cols", 0) or 0)
        n_categorical = int(task_info.get("n_categorical", 0) or 0)
        missing_ratio = float(task_info.get("missing_ratio", 0.0) or 0.0)

        preprocessing: list[str] = []
        if missing_ratio > 0.08:
            preprocessing.append("median_impute")
        if n_categorical > 0:
            preprocessing.append("label_encoding")
        preprocessing.append("robust_scaling")

        feature_eng: list[str] = []
        if n_categorical >= 3:
            feature_eng.append("frequency_encoding")
        if n_cols >= 8:
            feature_eng.append("numeric_interactions")

        base_models = LLMDecisionMaker._default_models_for_task(task_type)
        if n_rows > 120000:
            models = [m for m in base_models if m in ("lightgbm", "xgboost", "catboost", "random_forest", "extra_trees")]
        elif n_cols > 120:
            models = [m for m in base_models if m in ("lightgbm", "extra_trees", "random_forest", "ridge", "elastic_net")]
        else:
            models = base_models

        if task_type != "regression":
            models = [m for m in models if m not in ("ridge", "lasso", "elastic_net")]
        if task_type == "regression":
            models = [m for m in models if m != "logistic_regression"]

        if len(models) < 3:
            for m in base_models:
                if m not in models:
                    models.append(m)
                if len(models) >= 5:
                    break

        if not models:
            models = ["gradient_boosting", "random_forest"]

        return {
            "task_type": task_type,
            "models": models[:5],
            "preprocessing": preprocessing,
            "feature_engineering": feature_eng,
            "justification": (
                "Heuristic recommendation based on data size, feature mix, "
                "and missing-value profile."
            ),
        }


# ---------------------------------------------------------------------------
# Data Preprocessor
# ---------------------------------------------------------------------------

class DataPreprocessor:
    """Advanced data preprocessing pipeline."""

    def __init__(self):
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler: Optional[RobustScaler] = None
        self.medians: dict[str, float] = {}
        self.modes: dict[str, Any] = {}
        self.target_encoder: Optional[LabelEncoder] = None
        self.feature_cols: list[str] = []
        self.id_col: Optional[str] = None
        self.target_col: Optional[str] = None
        self.train_numeric_medians: dict[str, float] = {}
        self.binary_numeric_target: bool = False

    # -- public API ----------------------------------------------------------

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: Optional[str],
        id_col: Optional[str],
        task_type: str = "binary_classification",
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """
        Fit preprocessing on train, transform both train and test.
        Returns (X_train, y_train, X_test) as numpy arrays.
        """
        self.target_col = target_col
        self.id_col = id_col

        self.feature_cols = [
            c for c in train_df.columns if c != target_col and c != id_col
        ]

        train = train_df[self.feature_cols].copy()
        test = test_df.reindex(columns=self.feature_cols).copy()

        # 1. Parse dates / convert types
        train, test = self._parse_dates(train, test)

        # 2. Identify column types
        num_cols, cat_cols = self._classify_columns(train)

        # 3. Impute missing values
        train, test = self._impute(train, test, num_cols, cat_cols)

        # 4. Clip outliers on numeric
        train, test = self._clip_outliers(train, test, num_cols)

        # 5. Feature engineering
        train, test = self._feature_engineering(train, test, num_cols, cat_cols)

        # Re-detect after engineering
        num_cols, cat_cols = self._classify_columns(train)

        # 6. Encode categoricals
        train, test = self._encode_categoricals(train, test, cat_cols)

        # 7. Scale numerics
        train, test = self._scale_numerics(train, test, num_cols)

        # 8. Ensure fully numeric
        train = train.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
        test = test.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)

        X_train = train.values
        X_test = test.values

        # Target encoding
        y_train = None
        if target_col and target_col in train_df.columns:
            raw_target = train_df[target_col].copy()
            if "classification" in task_type:
                clean_target = raw_target.dropna()
                self.binary_numeric_target = (
                    clean_target.nunique() == 2
                    and pd.api.types.is_numeric_dtype(clean_target)
                    and set(pd.to_numeric(clean_target, errors="coerce").dropna().unique()).issubset({0.0, 1.0})
                )
                self.target_encoder = LabelEncoder()
                y_train = self.target_encoder.fit_transform(
                    raw_target.astype(str).fillna("missing")
                )
            else:
                y_train = pd.to_numeric(raw_target, errors="coerce")
                median_val = y_train.median()
                y_train = y_train.fillna(median_val).values

        return X_train, y_train, X_test

    def inverse_transform_target(self, predictions: np.ndarray) -> np.ndarray:
        """Inverse-transform predictions back to original target labels."""
        if self.target_encoder is not None:
            return self.target_encoder.inverse_transform(
                predictions.astype(int)
            )
        return predictions

    def should_output_probabilities(self, task_type: str) -> bool:
        """
        Return True only when binary target is numeric 0/1.
        This avoids returning probabilities for categorical labels.
        """
        return task_type == "binary_classification" and self.binary_numeric_target

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _parse_dates(
        train: pd.DataFrame, test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Convert datetime-like columns to numeric features."""
        for col in list(train.columns):
            if train[col].dtype == "object":
                try:
                    dt = pd.to_datetime(train[col], errors="coerce", utc=False)
                    parse_ratio = dt.notna().mean()
                    if parse_ratio >= 0.7:
                        train[col + "_year"] = dt.dt.year
                        train[col + "_month"] = dt.dt.month
                        train[col + "_dayofweek"] = dt.dt.dayofweek
                        test[col + "_year"] = pd.to_datetime(
                            test[col], errors="coerce", utc=False
                        ).dt.year
                        test[col + "_month"] = pd.to_datetime(
                            test[col], errors="coerce", utc=False
                        ).dt.month
                        test[col + "_dayofweek"] = pd.to_datetime(
                            test[col], errors="coerce", utc=False
                        ).dt.dayofweek
                        train.drop(columns=[col], inplace=True)
                        test.drop(columns=[col], inplace=True)
                except Exception:
                    pass
        return train, test

    @staticmethod
    def _classify_columns(
        df: pd.DataFrame,
    ) -> tuple[list[str], list[str]]:
        """Separate numeric and categorical columns."""
        cat_cols = []
        num_cols = []
        for c in df.columns:
            dtype = df[c].dtype
            if dtype == "object" or dtype.name in ("string", "str", "category"):
                cat_cols.append(c)
            elif pd.api.types.is_numeric_dtype(dtype):
                num_cols.append(c)
            else:
                # Boolean or other — treat as categorical
                cat_cols.append(c)
        return num_cols, cat_cols

    def _impute(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        num_cols: list[str],
        cat_cols: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Impute missing values."""
        for col in num_cols:
            median_val = train[col].median()
            self.medians[col] = median_val
            train[col] = train[col].fillna(median_val)
            test[col] = test[col].fillna(median_val)

        for col in cat_cols:
            mode_val = train[col].mode()
            mode_val = mode_val.iloc[0] if len(mode_val) else "missing"
            self.modes[col] = mode_val
            train[col] = train[col].fillna(mode_val)
            test[col] = test[col].fillna(mode_val)

        return train, test

    @staticmethod
    def _clip_outliers(
        train: pd.DataFrame,
        test: pd.DataFrame,
        num_cols: list[str],
        clip_pct: float = 0.01,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Winsorize extreme values using percentiles."""
        for col in num_cols:
            low = train[col].quantile(clip_pct)
            high = train[col].quantile(1 - clip_pct)
            train[col] = train[col].clip(low, high)
            test[col] = test[col].clip(low, high)
        return train, test

    def _feature_engineering(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        num_cols: list[str],
        cat_cols: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Add simple interaction / summary features."""
        del cat_cols  # reserved for future categorical feature engineering
        present_num = [c for c in num_cols if c in train.columns]
        self.train_numeric_medians = {
            c: float(train[c].median()) for c in present_num
        }

        for df in (train, test):
            current_num = [c for c in present_num if c in df.columns]
            if len(current_num) >= 2:
                df["_num_mean"] = df[current_num].mean(axis=1)
                df["_num_std"] = df[current_num].std(axis=1).fillna(0)
            if len(current_num) >= 3:
                df["_num_max"] = df[current_num].max(axis=1)
                df["_num_min"] = df[current_num].min(axis=1)

            if len(present_num) > 3:
                df["_n_above_median"] = sum(df[c] > self.train_numeric_medians[c] for c in current_num)

            # Pairwise products of first few numeric cols (cap to avoid explosion)
            few = current_num[:5]
            for i in range(min(3, len(few))):
                for j in range(i + 1, min(3, len(few))):
                    df[f"{few[i]}_x_{few[j]}"] = df[few[i]] * df[few[j]]
        return train, test

    def _encode_categoricals(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        cat_cols: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Label-encode categorical columns, handling unseen categories."""
        for col in cat_cols:
            le = LabelEncoder()
            train_vals = train[col].astype(str).fillna("missing")
            test_vals = test[col].astype(str).fillna("missing")
            le.fit(train_vals)
            train[col] = le.transform(train_vals)
            mapping = {value: idx for idx, value in enumerate(le.classes_)}
            test[col] = test_vals.map(mapping).fillna(-1).astype(int)
            self.label_encoders[col] = le
        return train, test

    def _scale_numerics(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        num_cols: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply RobustScaler to numeric columns."""
        if num_cols:
            self.scaler = RobustScaler()
            train[num_cols] = self.scaler.fit_transform(train[num_cols])
            test[num_cols] = self.scaler.transform(test[num_cols])
        return train, test


# ---------------------------------------------------------------------------
# Task Analyzer (enhanced)
# ---------------------------------------------------------------------------

class TaskAnalyzer:
    """Analyze competition data to determine task type and strategy."""

    @staticmethod
    def analyze_competition(competition_dir: Path) -> dict[str, Any]:
        """
        Analyze competition directory to determine:
        - Task type (classification, regression, etc.)
        - Data type (tabular, text, image)
        - Target column(s)
        - Appropriate modeling strategy
        """
        description = TaskAnalyzer._read_description(competition_dir)
        data_files = TaskAnalyzer._find_data_files(competition_dir)
        train_file = TaskAnalyzer._find_train_file(competition_dir, data_files)

        task_info = {
            "description": description,
            "data_files": data_files,
            "task_type": "binary_classification",
            "data_type": "tabular",
            "target_column": None,
            "id_column": None,
            "strategy": "gradient_boosting",
        }

        if train_file and train_file.suffix == ".csv":
            task_info.update(TaskAnalyzer._analyze_csv(train_file))

        return task_info

    @staticmethod
    def _read_description(competition_dir: Path) -> Optional[str]:
        """Read competition description if available."""
        for name in ("description.md", "description.txt", "overview.md"):
            desc_path = competition_dir / name
            if desc_path.exists():
                try:
                    return desc_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    return desc_path.read_text(errors="ignore")
        return None

    @staticmethod
    def _find_data_files(competition_dir: Path) -> list[Path]:
        """Find all data files in competition directory."""
        for sub in ("data", "input", ""):
            data_dir = competition_dir / sub if sub else competition_dir
            if data_dir.exists():
                return sorted(list(data_dir.iterdir()), key=lambda p: p.name.lower())
        return sorted(list(competition_dir.glob("*")), key=lambda p: p.name.lower())

    @staticmethod
    def _find_train_file(
        competition_dir: Path, data_files: list[Path]
    ) -> Optional[Path]:
        """Find training data file."""
        del competition_dir
        for f in data_files:
            if f.name.lower().startswith("train") and f.suffix.lower() == ".csv":
                return f
        for f in data_files:
            name = f.name.lower()
            if f.suffix.lower() == ".csv" and "sample_submission" not in name:
                return f
        return None

    @staticmethod
    def _detect_id_column(df: pd.DataFrame) -> Optional[str]:
        """Infer an ID column from common naming and uniqueness patterns."""
        if df.empty:
            return None

        id_candidates = ["id", "ID", "Id", "PassengerId", "passenger_id", "row_id"]
        for candidate in id_candidates:
            if candidate in df.columns:
                return candidate

        n_rows = len(df)
        for col in df.columns:
            col_lower = col.lower()
            unique_ratio = df[col].nunique(dropna=False) / max(1, n_rows)
            if "id" in col_lower and unique_ratio > 0.95:
                return col
        return None

    @staticmethod
    def _detect_target_column(df: pd.DataFrame, id_col: Optional[str]) -> Optional[str]:
        """Infer target column with explicit-name and weak-statistical heuristics."""
        if df.empty:
            return None

        target_candidates = [
            "target", "Target", "TARGET",
            "label", "Label",
            "survived", "Survived",
            "class", "Class",
            "output", "Output",
            "saleprice", "SalePrice",
            "response", "y", "Y",
        ]
        for candidate in target_candidates:
            if candidate in df.columns and candidate != id_col:
                return candidate

        candidate_cols = [c for c in df.columns if c != id_col]
        if not candidate_cols:
            return None

        if len(candidate_cols) == 1:
            return candidate_cols[0]

        n_rows = len(df)
        scored: list[tuple[float, str]] = []
        for col in candidate_cols:
            series = df[col]
            unique_ratio = series.nunique(dropna=True) / max(1, n_rows)
            missing_ratio = series.isna().mean()
            score = 0.0

            # Targets are usually less unique than row identifiers.
            score += (1.0 - unique_ratio) * 0.7
            # Slight preference for columns with lower missingness.
            score += (1.0 - missing_ratio) * 0.2

            col_lower = col.lower()
            if any(token in col_lower for token in ("target", "label", "class", "output", "response", "sale", "price")):
                score += 1.5

            scored.append((score, col))

        scored.sort(reverse=True)
        return scored[0][1]

    @staticmethod
    def _infer_task_type(target_values: pd.Series) -> tuple[str, Any, Any]:
        """Infer task type from target distribution."""
        non_na = target_values.dropna()
        if non_na.empty:
            return "binary_classification", 2, "N/A"

        n_unique = int(non_na.nunique())
        n_rows = len(non_na)
        value_counts = non_na.value_counts()

        if n_unique == 2:
            return "binary_classification", 2, dict(value_counts.head(10).to_dict())

        is_numeric_target = pd.api.types.is_numeric_dtype(non_na)
        unique_ratio = n_unique / max(1, n_rows)
        regression_like = is_numeric_target and (
            n_unique >= max(20, int(n_rows * 0.2)) or unique_ratio >= 0.25
        )

        if regression_like:
            return "regression", "continuous", "N/A"

        if n_unique <= 20:
            return "multiclass_classification", n_unique, dict(value_counts.head(10).to_dict())

        # Fallback: non-numeric high cardinality is still usually classification.
        if is_numeric_target:
            return "regression", "continuous", "N/A"
        return "multiclass_classification", n_unique, "N/A"

    @staticmethod
    def _analyze_csv(train_file: Path) -> dict[str, Any]:
        """Analyze CSV file to determine task characteristics."""
        try:
            df = pd.read_csv(train_file, nrows=5000)
        except Exception as e:
            logger.warning(f"Failed to analyze CSV: {e}")
            return {}

        info: dict[str, Any] = {
            "n_rows": len(df),
            "n_cols": len(df.columns),
        }

        id_col = TaskAnalyzer._detect_id_column(df)
        target_col = TaskAnalyzer._detect_target_column(df, id_col=id_col)
        info["id_column"] = id_col
        info["target_column"] = target_col

        feature_cols = [c for c in df.columns if c not in {target_col, id_col}]
        cat_cols = [
            c for c in feature_cols
            if df[c].dtype == "object" or df[c].dtype.name in ("string", "str", "category")
        ]
        num_cols = [
            c for c in feature_cols
            if pd.api.types.is_numeric_dtype(df[c].dtype)
        ]
        info["n_categorical"] = len(cat_cols)
        info["n_numeric"] = len(num_cols)

        missing_total = df.isna().sum().sum()
        total_cells = df.shape[0] * df.shape[1]
        info["missing_ratio"] = missing_total / total_cells if total_cells else 0.0

        if target_col and target_col in df.columns:
            task_type, target_cardinality, class_balance = TaskAnalyzer._infer_task_type(df[target_col])
            info["task_type"] = task_type
            info["target_cardinality"] = target_cardinality
            info["class_balance"] = class_balance

        return info


# ---------------------------------------------------------------------------
# Model Trainer (enhanced with model selection)
# ---------------------------------------------------------------------------

class ModelTrainer:
    """Train ML models and generate predictions."""

    # Registry of all available models
    _CLASSIFIERS = {}
    _REGRESSORS = {}

    @classmethod
    def _register_models(cls):
        """Populate model registry."""
        if cls._CLASSIFIERS:
            return  # already registered

        cls._CLASSIFIERS = {
            "gradient_boosting": lambda: GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                subsample=0.8, random_state=42,
            ),
            "random_forest": lambda: RandomForestClassifier(
                n_estimators=300, max_depth=None, min_samples_split=5,
                random_state=42, n_jobs=-1,
            ),
            "extra_trees": lambda: ExtraTreesClassifier(
                n_estimators=300, max_depth=None, random_state=42, n_jobs=-1,
            ),
            "logistic_regression": lambda: LogisticRegression(
                max_iter=2000, solver="lbfgs", random_state=42,
            ),
            "svm": lambda: SVC(
                probability=True, kernel="rbf", random_state=42,
            ),
            "knn": lambda: KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
        }

        cls._REGRESSORS = {
            "gradient_boosting_regressor": lambda: GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                subsample=0.8, random_state=42,
            ),
            "random_forest_regressor": lambda: RandomForestRegressor(
                n_estimators=300, max_depth=None, min_samples_split=5,
                random_state=42, n_jobs=-1,
            ),
            "extra_trees_regressor": lambda: ExtraTreesRegressor(
                n_estimators=300, max_depth=None, random_state=42, n_jobs=-1,
            ),
            "ridge": lambda: Ridge(alpha=1.0, random_state=42),
            "lasso": lambda: Lasso(alpha=1.0, max_iter=5000, random_state=42),
            "elastic_net": lambda: ElasticNet(
                alpha=1.0, l1_ratio=0.5, max_iter=5000, random_state=42,
            ),
            "svr": lambda: SVR(kernel="rbf"),
            "knn_regressor": lambda: KNeighborsRegressor(n_neighbors=7, n_jobs=-1),
        }

        # Conditional heavy models
        if HAS_LIGHTGBM:
            cls._CLASSIFIERS["lightgbm"] = lambda: lgb.LGBMClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                num_leaves=31, random_state=42, verbose=-1, n_jobs=-1,
            )
            cls._REGRESSORS["lightgbm_regressor"] = lambda: lgb.LGBMRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                num_leaves=31, random_state=42, verbose=-1, n_jobs=-1,
            )
            # Aliases
            cls._CLASSIFIERS["lightgbm"] = cls._CLASSIFIERS["lightgbm"]
            cls._REGRESSORS["lightgbm"] = cls._REGRESSORS.get("lightgbm_regressor")

        if HAS_XGBOOST:
            cls._CLASSIFIERS["xgboost"] = lambda: xgb.XGBClassifier(
                n_estimators=500, max_depth=5, learning_rate=0.05,
                random_state=42, eval_metric="logloss", n_jobs=-1,
            )
            cls._REGRESSORS["xgboost"] = lambda: xgb.XGBRegressor(
                n_estimators=500, max_depth=5, learning_rate=0.05,
                random_state=42, n_jobs=-1,
            )

        if HAS_CATBOOST:
            cls._CLASSIFIERS["catboost"] = lambda: CatBoostClassifier(
                iterations=500, depth=6, learning_rate=0.05,
                random_state=42, verbose=False, thread_count=-1,
            )
            cls._REGRESSORS["catboost"] = lambda: CatBoostRegressor(
                iterations=500, depth=6, learning_rate=0.05,
                random_state=42, verbose=False, thread_count=-1,
            )

    @classmethod
    def get_model(cls, name: str, task_type: str):
        """Get a model instance by name."""
        cls._register_models()
        if "classification" in task_type:
            # Support both classifier and regressor names
            if name in cls._CLASSIFIERS:
                return cls._CLASSIFIERS[name]()
            # Map generic names
            mapping = {
                "lightgbm": "lightgbm",
                "xgboost": "xgboost",
                "catboost": "catboost",
                "random_forest": "random_forest",
                "extra_trees": "extra_trees",
                "gradient_boosting": "gradient_boosting",
                "logistic_regression": "logistic_regression",
                "svm": "svm",
                "knn": "knn",
                "ridge": "logistic_regression",
                "lasso": "logistic_regression",
                "elastic_net": "logistic_regression",
            }
            mapped = mapping.get(name, "gradient_boosting")
            return cls._CLASSIFIERS[mapped]()
        else:
            if name in cls._REGRESSORS:
                return cls._REGRESSORS[name]()
            mapping = {
                "lightgbm": "lightgbm_regressor",
                "xgboost": "xgboost",
                "catboost": "catboost",
                "random_forest": "random_forest_regressor",
                "extra_trees": "extra_trees_regressor",
                "gradient_boosting": "gradient_boosting_regressor",
                "ridge": "ridge",
                "lasso": "lasso",
                "elastic_net": "elastic_net",
                "svm": "svr",
                "knn": "knn_regressor",
            }
            mapped = mapping.get(name, "gradient_boosting_regressor")
            if mapped in cls._REGRESSORS:
                return cls._REGRESSORS[mapped]()
            return GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42,
            )

    @classmethod
    def _scoring(cls, task_type: str):
        """Return scoring metric for cross-validation."""
        if task_type == "binary_classification":
            return "roc_auc"
        elif task_type == "multiclass_classification":
            return "accuracy"
        else:
            return "neg_root_mean_squared_error"

    @classmethod
    def _cv_split(cls, task_type: str, y, n_splits: int = 5):
        """Create CV splitter."""
        n_samples = len(y)
        if n_samples < 3:
            raise ValueError("Need at least 3 samples for cross-validation")

        if "classification" in task_type:
            y_series = pd.Series(y)
            min_class_count = int(y_series.value_counts().min())
            if min_class_count < 2:
                raise ValueError("Each class must have at least 2 samples for CV")
            safe_splits = min(n_splits, min_class_count, n_samples)
            safe_splits = max(2, safe_splits)
            return StratifiedKFold(
                n_splits=safe_splits, shuffle=True, random_state=42
            )

        safe_splits = min(n_splits, n_samples)
        safe_splits = max(2, safe_splits)
        return KFold(n_splits=safe_splits, shuffle=True, random_state=42)

    @classmethod
    def _default_candidates(cls, task_type: str) -> list[str]:
        if "classification" in task_type:
            defaults = ["lightgbm", "catboost", "xgboost", "extra_trees", "random_forest", "gradient_boosting"]
        else:
            defaults = ["lightgbm", "catboost", "xgboost", "extra_trees", "random_forest", "ridge"]
        available: list[str] = []
        for name in defaults:
            try:
                cls.get_model(name, task_type)
                available.append(name)
            except Exception:
                continue
        return available or (["gradient_boosting"] if "classification" in task_type else ["gradient_boosting_regressor"])

    @classmethod
    def _sanitize_candidate_names(
        cls, candidate_names: list[str], task_type: str
    ) -> list[str]:
        cls._register_models()
        unique_names: list[str] = []
        seen: set[str] = set()
        for name in candidate_names or []:
            if not isinstance(name, str):
                continue
            if name in ("stacking_ensemble", "voting_ensemble"):
                continue
            if name in seen:
                continue
            try:
                cls.get_model(name, task_type)
            except Exception:
                continue
            unique_names.append(name)
            seen.add(name)

        if not unique_names:
            unique_names = cls._default_candidates(task_type)
        return unique_names

    @classmethod
    def select_best_model(
        cls,
        X_train: np.ndarray,
        y_train: np.ndarray,
        task_type: str,
        candidate_names: list[str],
        n_folds: int = 5,
    ) -> tuple[str, Any, float]:
        """
        Evaluate candidate models via cross-validation and return the best.
        Returns (name, fitted_model, cv_score).
        """
        cls._register_models()
        scoring = cls._scoring(task_type)
        cv = cls._cv_split(task_type, y_train, n_splits=n_folds)

        best_name = None
        best_model = None
        best_score = -np.inf

        unique_names = cls._sanitize_candidate_names(candidate_names, task_type)

        for name in unique_names:
            try:
                model = cls.get_model(name, task_type)
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
                mean_score = scores.mean()

                logger.info(f"Model {name}: CV {scoring} = {mean_score:.4f} (+/- {scores.std():.4f})")

                if mean_score > best_score:
                    best_score = mean_score
                    best_name = name
            except ValueError as exc:
                # Typical failures: not enough class members for fold count.
                logger.warning("Model %s skipped due to CV constraint: %s", name, exc)
                continue
            except Exception as exc:
                logger.warning(f"Model {name} failed: {exc}")
                continue

        if best_name is None:
            for name in unique_names:
                try:
                    model = cls.get_model(name, task_type)
                    model.fit(X_train, y_train)
                    best_name = name
                    best_model = model
                    best_score = 0.0
                    break
                except Exception:
                    continue

        if best_name is None:
            # Ultimate fallback
            best_name = "gradient_boosting" if "classification" in task_type else "gradient_boosting_regressor"
            best_model = cls.get_model(best_name, task_type)
            best_model.fit(X_train, y_train)
            best_score = 0.0
        elif best_model is None:
            best_model = cls.get_model(best_name, task_type)
            best_model.fit(X_train, y_train)

        return best_name, best_model, best_score

    @classmethod
    def build_ensemble(
        cls,
        X_train: np.ndarray,
        y_train: np.ndarray,
        task_type: str,
        top_models: list[str],
    ) -> Any:
        """Build a stacking or voting ensemble from top models."""
        cls._register_models()
        top_models = cls._sanitize_candidate_names(top_models, task_type)
        if len(top_models) < 2:
            fallback_name = top_models[0] if top_models else cls._default_candidates(task_type)[0]
            model = cls.get_model(fallback_name, task_type)
            model.fit(X_train, y_train)
            return model

        estimators = []
        for name in top_models[:5]:
            try:
                m = cls.get_model(name, task_type)
                estimators.append((name, m))
            except Exception:
                continue

        if not estimators:
            fallback_name = cls._default_candidates(task_type)[0]
            model = cls.get_model(fallback_name, task_type)
            model.fit(X_train, y_train)
            return model

        if len(estimators) == 1:
            _, m = estimators[0]
            m.fit(X_train, y_train)
            return m

        try:
            cv = cls._cv_split(task_type, y_train, n_splits=5)
        except ValueError:
            fallback_name = estimators[0][0]
            fallback_model = cls.get_model(fallback_name, task_type)
            fallback_model.fit(X_train, y_train)
            return fallback_model

        if "classification" in task_type:
            ensemble = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(max_iter=2000, random_state=42),
                cv=cv,
                n_jobs=-1,
            )
        else:
            ensemble = StackingRegressor(
                estimators=estimators,
                final_estimator=Ridge(alpha=1.0),
                cv=cv,
                n_jobs=-1,
            )
        try:
            ensemble.fit(X_train, y_train)
            return ensemble
        except Exception as exc:
            logger.warning("Stacking failed (%s). Falling back to voting ensemble.", exc)
            if "classification" in task_type:
                can_soft_vote = all(hasattr(model, "predict_proba") for _, model in estimators)
                voting = VotingClassifier(
                    estimators=estimators,
                    voting="soft" if can_soft_vote else "hard",
                    n_jobs=-1,
                )
            else:
                voting = VotingRegressor(estimators=estimators, n_jobs=-1)
            voting.fit(X_train, y_train)
            return voting

    # -- public API ----------------------------------------------------------

    @classmethod
    def train_and_predict(
        cls,
        competition_dir: Path,
        task_info: dict[str, Any],
        use_ensemble: bool = False,
        n_folds: int = 5,
    ) -> pd.DataFrame:
        """
        Full pipeline: preprocess → select best model → predict.
        """
        data_dir = competition_dir / "data"
        if not data_dir.exists():
            data_dir = competition_dir

        train_file = None
        test_file = None
        for f in data_dir.glob("*.csv"):
            name_lower = f.name.lower()
            if name_lower.startswith("train"):
                train_file = f
            elif name_lower.startswith("test") and "sample_submission" not in name_lower:
                test_file = f
        if not train_file:
            csv_files = list(data_dir.glob("*.csv"))
            if csv_files:
                train_file = csv_files[0]
        if not train_file:
            raise ValueError("No training data found")

        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file) if test_file is not None else None
        if test_df is None:
            test_df = train_df.copy()

        target_col = task_info.get("target_column")
        id_col = task_info.get("id_column")
        task_type = task_info.get("task_type", "binary_classification")
        if not target_col or target_col not in train_df.columns:
            inferred_target = TaskAnalyzer._detect_target_column(train_df, id_col=id_col)
            if inferred_target:
                target_col = inferred_target
                task_info["target_column"] = target_col
        if not id_col or id_col not in train_df.columns:
            inferred_id = TaskAnalyzer._detect_id_column(train_df)
            if inferred_id:
                id_col = inferred_id
                task_info["id_column"] = id_col

        # Preprocessing
        preprocessor = DataPreprocessor()
        X_train, y_train, X_test = preprocessor.fit_transform(
            train_df, test_df, target_col, id_col, task_type,
        )
        if y_train is None:
            raise ValueError(
                f"Target column '{target_col}' was not found in training data."
            )

        # LLM-based or heuristic model selection
        llm = LLMDecisionMaker(
            llm_fn=task_info.get("llm_fn"),
            openai_api_key=task_info.get("openai_api_key"),
            openai_model=task_info.get("openai_model"),
            openai_base_url=task_info.get("openai_base_url"),
        )
        recommendation = llm.recommend(task_info)
        candidate_names = cls._sanitize_candidate_names(
            recommendation.get("models", []), task_type
        )

        if use_ensemble and len(candidate_names) >= 2:
            model = cls.build_ensemble(X_train, y_train, task_type, candidate_names[:5])
            model_name = "stacking_ensemble"
        else:
            model_name, model, _ = cls.select_best_model(
                X_train, y_train, task_type, candidate_names, n_folds=n_folds,
            )

        logger.info(f"Best model: {model_name}")

        # Predict
        is_bool_target = False
        if "classification" in task_type:
            # Check if original target was boolean-like
            if target_col and target_col in train_df.columns:
                target_vals = train_df[target_col].dropna()
                is_bool_target = target_vals.dtype == "bool" or set(target_vals.unique()).issubset({True, False, "True", "False"})

            if task_type == "binary_classification":
                if preprocessor.should_output_probabilities(task_type) and hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_test)[:, 1]
                    if is_bool_target:
                        predictions = (proba >= 0.5).astype(int)
                    else:
                        predictions = proba
                else:
                    predictions = model.predict(X_test)
                    predictions = preprocessor.inverse_transform_target(predictions)
            else:
                predictions = model.predict(X_test)
                predictions = preprocessor.inverse_transform_target(predictions)
        else:
            predictions = model.predict(X_test)

        # Submission DataFrame
        submission_df = pd.DataFrame()
        if id_col and id_col in test_df.columns:
            submission_df[id_col] = test_df[id_col]

        # Use correct target name if known
        target_name = target_col if target_col else "target"
        submission_df[target_name] = predictions

        return submission_df


# ---------------------------------------------------------------------------
# PurpleAgent (top-level)
# ---------------------------------------------------------------------------

class PurpleAgent:
    """
    Purple ML Agent - solves ML competitions from MLE-Bench.

    Receives competition.tar.gz, trains models, returns submission.csv.
    """

    def __init__(
        self,
        llm_fn: Optional[Callable[[str], Optional[str]]] = None,
        openai_api_key: Optional[str] = None,
        openai_model: Optional[str] = None,
        openai_base_url: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        llm_fn : callable(prompt: str) -> str | None
            Optional custom LLM callable. Takes priority over OpenAI.
        openai_api_key : str | None
            OpenAI API key.  If not set, uses ``OPENAI_API_KEY`` env var.
        openai_model : str | None
            OpenAI model name.  Defaults to ``gpt-4.1`` (latest stable).
        openai_base_url : str | None
            Custom OpenAI-compatible base URL (for proxies, Azure, Ollama, etc.).
        """
        self.work_dir: Optional[Path] = None
        self.task_info: dict[str, Any] = {}
        self.llm_fn = llm_fn
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.openai_base_url = openai_base_url

    def extract_competition_data(self, tar_bytes: bytes) -> Path:
        """
        Extract competition.tar.gz to temporary directory.
        Returns path to extracted directory.
        """
        self.work_dir = Path(tempfile.mkdtemp(prefix="purple_agent_"))

        tar_buffer = io.BytesIO(tar_bytes)
        with tarfile.open(fileobj=tar_buffer, mode="r:gz") as tar:
            # Use safe extraction mode when supported (Python 3.12+).
            try:
                tar.extractall(path=self.work_dir, filter="data")
            except TypeError:
                tar.extractall(path=self.work_dir)

        logger.info(f"Extracted competition data to {self.work_dir}")

        # Resolve actual data directory
        if (self.work_dir / "home" / "data").exists():
            return self.work_dir / "home"
        elif (self.work_dir / "data").exists():
            return self.work_dir
        else:
            subdirs = [d for d in self.work_dir.iterdir() if d.is_dir()]
            if subdirs and (subdirs[0] / "data").exists():
                return subdirs[0] / "data"
            return self.work_dir

    def analyze_task(self, competition_dir: Path) -> dict[str, Any]:
        """Analyze the competition task."""
        self.task_info = TaskAnalyzer.analyze_competition(competition_dir)
        # Inject LLM config for ModelTrainer
        self.task_info["llm_fn"] = self.llm_fn
        self.task_info["openai_api_key"] = self.openai_api_key
        self.task_info["openai_model"] = self.openai_model
        self.task_info["openai_base_url"] = self.openai_base_url
        logger.info(f"Task analysis complete: {self.task_info}")
        return self.task_info

    def solve_task(
        self,
        competition_dir: Path,
        use_ensemble: bool = True,
        n_folds: int = 5,
    ) -> pd.DataFrame:
        """Solve the ML competition task."""
        submission_df = ModelTrainer.train_and_predict(
            competition_dir,
            self.task_info,
            use_ensemble=use_ensemble,
            n_folds=n_folds,
        )
        logger.info(f"Generated submission with {len(submission_df)} predictions")
        return submission_df

    def create_submission_bytes(self, submission_df: pd.DataFrame) -> bytes:
        """Convert submission DataFrame to CSV bytes."""
        csv_bytes = submission_df.to_csv(index=False).encode("utf-8")
        return csv_bytes

    def solve_competition(
        self,
        tar_bytes: bytes,
        use_ensemble: bool = True,
        n_folds: int = 5,
    ) -> bytes:
        """Complete pipeline: extract, analyze, solve, return submission CSV."""
        competition_dir = self.extract_competition_data(tar_bytes)
        self.analyze_task(competition_dir)
        submission_df = self.solve_task(
            competition_dir, use_ensemble=use_ensemble, n_folds=n_folds,
        )
        return self.create_submission_bytes(submission_df)

    def cleanup(self):
        """Clean up temporary files."""
        if self.work_dir and self.work_dir.exists():
            import shutil
            shutil.rmtree(self.work_dir, ignore_errors=True)
            self.work_dir = None