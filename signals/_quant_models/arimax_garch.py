"""Quantitative models for the ARIMAX-GARCH trading signal strategy.

This module defines the core quantitative logic for the ARIMAX-GARCH model,
including stationarity checks, parameter optimization, and the main model class
that handles data preparation, fitting, and forecasting.
"""

from arch import arch_model  # pylint: disable=E0611
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from typing import Any, Optional, Tuple

from colorama import init

init(autoreset=True)

# Path setup for imports
current_dir = Path(__file__).resolve().parent
# Check if the parent directory is already in the path to avoid duplicates
if str(current_dir.parent.parent) not in sys.path:
    sys.path.insert(0, str(current_dir.parent.parent))

from components._generate_indicator_features import generate_indicator_features
from components.config import (
    ARIMAX_DEFAULT_ORDER, ARIMAX_MAX_D, ARIMAX_MAX_P, ARIMAX_MAX_Q,
    COL_BB_LOWER, COL_BB_UPPER, COL_CLOSE, COL_VOLUME, GARCH_P, GARCH_Q,
    MIN_DATA_POINTS
)
from utilities._logger import setup_logging

logger = setup_logging('arimax_garch', log_level=logging.DEBUG)


def check_stationarity(series: Any, name: str = "Series") -> bool:
    """Checks the stationarity of a time series using the Augmented Dickey-Fuller test.

    A stationary series is one whose statistical properties such as mean, variance,
    autocorrelation, etc. are all constant over time.

    Args:
        series: The time series data to check.
        name: The name of the series for logging purposes.

    Returns:
        True if the series is stationary (p-value <= 0.05), False otherwise.
    """
    try:
        result = adfuller(series.dropna())
        logger.analysis(f"Stationarity test for {name}: ADF Statistic={result[0]:.6f}, p-value={result[1]:.6f}")
        if result[1] <= 0.05:
            logger.analysis(f"  Result: Stationary (p-value <= 0.05)")
            return True
        else:
            logger.analysis(f"  Result: Not Stationary (p-value > 0.05)")
            return False
    except Exception as e:
        logger.warning(f"Could not perform stationarity test for {name}: {e}")
        return False


def find_optimal_arimax_order(y: Any,
                            exog: Optional[Any] = None) -> Tuple[int, int, int]:
    """Finds the optimal (p, d, q) order for an ARIMAX model.

    The function iterates through a predefined range of p, d, and q values,
    fitting an ARIMAX model for each combination and selecting the order that
    minimizes the Akaike Information Criterion (AIC).

    Args:
        y: The endogenous variable (time series).
        exog: The exogenous variables (optional).

    Returns:
        A tuple (p, d, q) representing the optimal order found.
    """
    best_aic = float('inf')
    best_order = ARIMAX_DEFAULT_ORDER
    logger.model("Searching for optimal ARIMAX (p, d, q) order...")

    for p in range(ARIMAX_MAX_P + 1):
        for d in range(ARIMAX_MAX_D + 1):
            for q in range(ARIMAX_MAX_Q + 1):
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=ConvergenceWarning)
                        model = ARIMA(y, exog=exog, order=(p, d, q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                except Exception as e:
                    # This can happen for non-invertible MA parameters, etc.
                    # We can safely ignore these and continue the search.
                    logger.debug(f"Skipping order {(p, d, q)} due to error: {e}")
                    continue

    logger.model(f"Optimal ARIMAX order found: {best_order} with AIC: {best_aic:.2f}")
    return best_order


class ArimaxGarchModel:
    """A class that encapsulates the ARIMAX-GARCH modeling process.

    This class provides methods to prepare data, fit the combined ARIMAX and GARCH
    models, and generate forecasts.
    """

    def __init__(self):
        """Initializes the ArimaxGarchModel instance."""
        self.arimax_model: Optional[ARIMA] = None
        self.garch_model: Optional[arch_model] = None
        self.arimax_results: Optional[Any] = None
        self.garch_results: Optional[Any] = None
        self.exog_cols: Optional[list[str]] = None
        self.model_config: dict = {}

    def prepare_data(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """Prepares the input DataFrame by generating technical indicators.

        Args:
            df_input: The raw input DataFrame with at least a 'close' column.

        Returns:
            A new DataFrame with added features and cleaned of NaN values.
            Returns an empty DataFrame if preparation fails.
        """
        if df_input.empty or COL_CLOSE not in df_input.columns:
            logger.error("Input DataFrame is empty or missing the 'close' column.")
            return pd.DataFrame()

        df_with_features = generate_indicator_features(df_input.copy())
        if df_with_features.empty:
            logger.error("Feature generation returned an empty DataFrame.")
            return pd.DataFrame()

        # Calculate additional features used in the model
        df_with_features['Returns'] = df_with_features[COL_CLOSE].pct_change()
        df_with_features['Log_Returns'] = np.log(
            df_with_features[COL_CLOSE] / df_with_features[COL_CLOSE].shift(1))
        df_with_features['Volatility'] = df_with_features['Returns'].rolling(
            window=20).std()

        if COL_VOLUME in df_with_features.columns:
            volume_ma = df_with_features[COL_VOLUME].rolling(window=20).mean()
            
            # Avoid division by zero
            df_with_features['Volume_Ratio'] = (
                df_with_features[COL_VOLUME] /
                volume_ma).replace([np.inf, -np.inf], 1).fillna(1)

        if COL_BB_UPPER in df_with_features.columns and COL_BB_LOWER in df_with_features.columns:
            bb_range = df_with_features[COL_BB_UPPER] - \
                df_with_features[COL_BB_LOWER]
            
            # Avoid division by zero
            df_with_features['BB_Position'] = ((
                df_with_features[COL_CLOSE] - df_with_features[COL_BB_LOWER]
            ) / bb_range).replace([np.inf, -np.inf], 0.5).fillna(0.5)

        df_clean = df_with_features.dropna()

        if len(df_clean) < MIN_DATA_POINTS:
            logger.error(
                f"Insufficient data after cleaning ({len(df_clean)} rows), "
                f"requires at least {MIN_DATA_POINTS}.")
            return pd.DataFrame()

        logger.data(
            f"Data preparation successful, returning {len(df_clean)} rows.")
        return df_clean

    def fit_arimax_model(self, df: pd.DataFrame, target_col: str = COL_CLOSE) -> bool:
        """Fits the ARIMAX model to the prepared data.

        This method selects exogenous variables, finds the optimal order,
        and fits the model, handling convergence issues by retrying with a
        more robust solver.

        Args:
            df: The prepared DataFrame with features.
            target_col: The name of the target variable column.

        Returns:
            True if the model was fitted successfully, False otherwise.
        """
        if df.empty:
            logger.error("Cannot fit ARIMAX model: DataFrame is empty.")
            return False

        # statsmodels requires a DatetimeIndex for some forecasting functionality.
        # If not present, we create a dummy index to ensure compatibility.
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.debug(
                "Input lacks a DatetimeIndex. Assigning a dummy date range for "
                "statsmodels compatibility.")
            df = df.set_index(
                pd.date_range(start='2000-01-01',
                                periods=len(df),
                                freq='H'))

        y = df[target_col]

        self.exog_cols = [
            'rsi', 'macd', 'BB_Position', 'Volume_Ratio', 'Volatility'
        ]
        available_exog_cols = [
            col for col in self.exog_cols if col in df.columns
        ]

        exog = df[available_exog_cols] if available_exog_cols else None
        if exog is not None:
            logger.model(
                f"Using {len(available_exog_cols)} exogenous variables: {available_exog_cols}"
            )
        else:
            logger.warning(
                "No exogenous variables available; fitting a standard ARIMA model."
            )

        check_stationarity(y, target_col)
        optimal_order = find_optimal_arimax_order(y, exog)

        logger.model(f"Fitting ARIMAX model with order {optimal_order}...")
        self.arimax_model = ARIMA(y, exog=exog, order=optimal_order)

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                self.arimax_results = self.arimax_model.fit()

                # If the last warning was a ConvergenceWarning, retry with a more robust solver.
                if w and issubclass(w[-1].category, ConvergenceWarning):
                    logger.warning(
                        "ConvergenceWarning detected. Retrying with 'nm' "
                        "(Nelder-Mead) solver.")
                    self.arimax_results = self.arimax_model.fit(method='nm')

        except Exception as e:
            logger.error(
                f"An unexpected error occurred during ARIMAX model fitting: {e}",
                exc_info=True)
            return False

        assert self.arimax_results is not None
        logger.success("ARIMAX model fitted successfully.")
        logger.model(f"ARIMAX Summary:\n{self.arimax_results.summary()}")
        return True

    def fit_garch_model(self, use_arimax_residuals: bool = True) -> bool:
        """Fits the GARCH model to the ARIMAX residuals.

        This method checks for ARCH effects using the Ljung-Box test before
        fitting the GARCH model. It includes a fallback to a simpler GARCH(1,1)
        model if the primary model fails to converge.

        Args:
            use_arimax_residuals: If True, uses residuals from the fitted ARIMAX
                model. Currently, only this option is supported.

        Returns:
            True if the model was fitted successfully, False otherwise.
        """
        if not use_arimax_residuals:
            logger.error(
                "GARCH fitting on returns is not supported in this version.")
            return False

        if self.arimax_results is None:
            logger.error(
                "Cannot fit GARCH: ARIMAX model has not been fitted yet.")
            return False

        residuals = self.arimax_results.resid.dropna()
        if residuals.empty:
            logger.error(
                "ARIMAX residuals are empty. Cannot fit GARCH model.")
            return False

        logger.model("Fitting GARCH model on ARIMAX residuals...")

        # Ljung-Box test on squared residuals to check for ARCH effects.
        # A significant p-value suggests that a GARCH model is appropriate.
        lb_test = acorr_ljungbox(residuals**2, lags=[10], return_df=True)
        p_value = lb_test['lb_pvalue'].iloc[-1]
        logger.analysis(
            f"Ljung-Box test for ARCH effects: p-value={p_value:.4f}")
        if p_value > 0.05:
            logger.warning(
                "Ljung-Box test is not significant (p > 0.05). "
                "GARCH model may not be appropriate.")

        try:
            self.garch_model = arch_model(residuals,  # pylint: disable=E1136
                                          vol='GARCH',
                                          p=GARCH_P,
                                          q=GARCH_Q,
                                          mean='Zero')
            assert self.garch_model is not None
            self.garch_results = self.garch_model.fit(disp='off')
        except Exception as e:
            logger.warning(
                f"Primary GARCH({GARCH_P},{GARCH_Q}) fitting failed: {e}. "
                "Trying fallback GARCH(1,1)...")
            try:
                # Fallback to a standard and robust GARCH(1,1) model.
                self.garch_model = arch_model(residuals,  # pylint: disable=E1136
                                              vol='GARCH',
                                              p=1,
                                              q=1,
                                              mean='Zero')
                assert self.garch_model is not None
                self.garch_results = self.garch_model.fit(disp='off')
            except Exception as e_fallback:
                logger.error(
                    f"GARCH fallback model also failed to fit: {e_fallback}",
                    exc_info=True)
                return False

        assert self.garch_results is not None
        logger.success("GARCH model fitted successfully.")
        logger.model(f"GARCH Summary:\n{self.garch_results.summary()}")
        return True

    def forecast(
        self,
        steps: int = 5,
        exog_forecast: Optional[Any] = None
    ) -> pd.DataFrame:
        """Generates n-step ahead forecasts for price and volatility.

        Args:
            steps: The number of steps to forecast ahead.
            exog_forecast: A DataFrame containing future values of the
                exogenous variables. If None, the last known values are used
                and held constant for the forecast horizon.

        Returns:
            A DataFrame containing the price forecast, confidence intervals,
            and volatility forecast for each step. Returns an empty DataFrame
            on failure.
        """
        if self.arimax_results is None or self.garch_results is None:
            logger.error(
                "Cannot forecast: one or both models have not been fitted.")
            return pd.DataFrame()

        logger.model(f"Generating {steps}-step forecast...")

        # If no future exogenous variables are provided, assume they remain constant.
        # This is a strong assumption and may impact forecast accuracy.
        if exog_forecast is None and self.arimax_results.model.exog is not None:
            last_exog = self.arimax_results.model.exog[-1:].values
            exog_forecast = np.tile(last_exog, (steps, 1))

        try:
            # Generate forecasts from both models
            arimax_forecast_obj = self.arimax_results.get_forecast(
                steps=steps, exog=exog_forecast)
            garch_forecast_obj = self.garch_results.forecast(horizon=steps,
                                                             reindex=False)

            # Extract values
            price_forecast = arimax_forecast_obj.predicted_mean
            conf_int = arimax_forecast_obj.conf_int()
            volatility_forecast = np.sqrt(
                garch_forecast_obj.variance.values.flatten())

            forecast_df = pd.DataFrame(
                {
                    'Price_Forecast': price_forecast.values,
                    'Price_Lower': conf_int.iloc[:, 0].values,
                    'Price_Upper': conf_int.iloc[:, 1].values,
                    'Volatility_Forecast': volatility_forecast
                },
                index=price_forecast.index)

            logger.success(
                f"Forecast generated successfully for {steps} steps.")
            return forecast_df

        except Exception as e:
            logger.error(
                f"An unexpected error occurred during forecasting: {e}",
                exc_info=True)
            return pd.DataFrame()