import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        df,
        target_col,
        past_features,
        future_features,
        context_length,
        prediction_length,
    ):
        """
        df: pandas DataFrame containing the time series data
        target_col: column name for the target variable
        past_features: list of past covariate column names
        future_features: list of future covariate column names
        context_length: how many time steps from the past are used for context
        prediction_length: how many future time steps are predicted
        """
        self.df = df
        self.target_col = target_col
        self.past_features = past_features
        self.future_features = future_features
        self.context_length = context_length
        self.prediction_length = prediction_length

    def __len__(self):
        # Each entry corresponds to a sample in the dataset
        # Adjust the length depending on your dataset and sequence lengths
        return len(self.df) - self.context_length - self.prediction_length

    def __getitem__(self, idx):
        # Get the context (past) and future sequences for the current sample
        past_target = (
            self.df[self.target_col]
            .iloc[idx : idx + self.context_length]
            .values
        )
        future_target = (
            self.df[self.target_col]
            .iloc[
                idx
                + self.context_length : idx
                + self.context_length
                + self.prediction_length
            ]
            .values
        )

        past_time_feat = (
            self.df[self.past_features]
            .iloc[idx : idx + self.context_length]
            .values
        )
        future_time_feat = (
            self.df[self.future_features]
            .iloc[
                idx
                + self.context_length : idx
                + self.context_length
                + self.prediction_length
            ]
            .values
        )

        past_observed_values = torch.ones_like(
            torch.tensor(past_target, dtype=torch.float32)
        )

        return {
            "past_target": torch.tensor(past_target, dtype=torch.float32),
            "future_target": torch.tensor(future_target, dtype=torch.float32),
            "past_time_feat": torch.tensor(
                past_time_feat, dtype=torch.float32
            ),
            "future_time_feat": torch.tensor(
                future_time_feat, dtype=torch.float32
            ),
            "past_observed_values": past_observed_values,
        }
