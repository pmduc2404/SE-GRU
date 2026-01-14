from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch
import pandas as pd


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series data.
    """

    def __init__(self, data, sequence_length, target_column, feature_columns):
        self.data = data.reset_index(drop=True)
        self.sequence_length = sequence_length
        self.target_column = target_column
        self.feature_columns = feature_columns

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        x = self.data.iloc[idx:idx + self.sequence_length][self.feature_columns].values
        y = self.data.iloc[idx + self.sequence_length][self.target_column]
        return torch.FloatTensor(x), torch.FloatTensor([y])


def prepare_data(
    file_path,
    sequence_length=48,
    target_column='Lake water level (m)',
    train_ratio=0.8
):

    print("Loading CSV data...")
    df = pd.read_csv(file_path)

    # Sort by time
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)

    print(f"Total samples: {len(df)}")

    # Define feature columns (exclude Time & target)
    feature_columns = [col for col in df.columns if col not in ['Time', target_column]]

    print(f"Feature columns: {feature_columns}")
    print(f"Target column: {target_column}")

    # === Time-based split (NO SHUFFLE) ===
    train_size = int(train_ratio * len(df))
    train_df = df.iloc[:train_size].reset_index(drop=True)
    test_df = df.iloc[train_size:].reset_index(drop=True)

    print(f"Train size: {len(train_df)}")
    print(f"Test size : {len(test_df)}")

    # === Scaling (FIT ON TRAIN ONLY) ===
    scaler_features = StandardScaler()
    scaler_target = StandardScaler()

    # Scale features
    train_df[feature_columns] = scaler_features.fit_transform(
        train_df[feature_columns]
    )
    test_df[feature_columns] = scaler_features.transform(
        test_df[feature_columns]
    )

    # Scale target separately
    train_df[target_column] = scaler_target.fit_transform(
        train_df[[target_column]]
    )
    test_df[target_column] = scaler_target.transform(
        test_df[[target_column]]
    )

    # === Create datasets ===
    train_dataset = TimeSeriesDataset(
        train_df, sequence_length, target_column, feature_columns
    )

    test_dataset = TimeSeriesDataset(
        test_df, sequence_length, target_column, feature_columns
    )

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Test dataset length : {len(test_dataset)}")

    return train_dataset, test_dataset, scaler_features, scaler_target
