import warnings
import pandas as pd

def preprocess_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None)
    df.columns = [
        "block_id",
        "frame_idx",
        "E_mu_Z",
        "E_mu_phys_est",
        "E_mu_X",
        "E_nu1_X",
        "E_nu2_X",
        "E_nu1_Z",
        "E_nu2_Z",
        "N_mu_X",
        "M_mu_XX",
        "M_mu_XZ",
        "M_mu_X",
        "N_mu_Z",
        "M_mu_ZZ",
        "M_mu_Z",
        "N_nu1_X",
        "M_nu1_XX",
        "M_nu1_XZ",
        "M_nu1_X",
        "N_nu1_Z",
        "M_nu1_ZZ",
        "M_nu1_Z",
        "N_nu2_X",
        "M_nu2_XX",
        "M_nu2_XZ",
        "M_nu2_X",
        "N_nu2_Z",
        "M_nu2_ZZ",
        "M_nu2_Z",
        "nTot",
        "bayesImVoltage",
        "opticalPower",
        "polarizerVoltages[0]",
        "polarizerVoltages[1]",
        "polarizerVoltages[2]",
        "polarizerVoltages[3]",
        "temp_1",
        "biasVoltage_1",
        "temp_2",
        "biasVoltage_2",
        "synErr",
        "N_EC_rounds",
        "maintenance_flag",
        "estimator_name",
        "f_EC",
        "E_mu_Z_est",
        "R",
        "s",
        "p",
    ]

    cols_to_drop = ["E_mu_phys_est", "f_EC", "R", "s", "p"]
    df_features = df.drop(columns=cols_to_drop)

    targets = df[["block_id", "frame_idx", "R", "s"]]

    df_features = df_features.rename(columns={
        "block_id": "id",
        "E_mu_Z": "value",
        "frame_idx": "date"
    })

    df_features = df_features.sort_values(["id", "date"]).reset_index(drop=True)

    feature_cols = [col for col in df_features.columns if col not in ["id", "date"]]

    df_features[feature_cols] = df_features.groupby("id")[feature_cols].fillna(method="ffill")
    df_features[feature_cols] = df_features.groupby("id")[feature_cols].fillna(method="bfill")

    df_for_ts = df_features.dropna(subset=feature_cols).reset_index(drop=True)

    HORIZON = 8
    HISTORY = 160

    train_df = []
    val_df = []
    test_df = []
    test_targets = []

    for current_id in df_for_ts["id"].unique():
        current_df = df_for_ts[df_for_ts["id"] == current_id]
        train_df.append(current_df.iloc[: -2 * HORIZON])
        val_df.append(current_df.iloc[-2 * HORIZON - HISTORY : -HORIZON])
        test_df.append(current_df.iloc[-HORIZON - HISTORY : -HORIZON])
        test_targets.append(current_df.iloc[-HORIZON:])

    train_df = pd.concat(train_df)
    val_df = pd.concat(val_df)
    test_df = pd.concat(test_df)
    test_targets = pd.concat(test_targets)


    print(f"Форма обучающего набора: {train_df.shape}")
    print(f"Форма валидационного набора: {val_df.shape}")
    print(f"Форма тестового набора: {test_df.shape}")
    print(f"Форма целевых значений теста: {test_targets.shape}")

    print(f"Количество рядов в обучающем наборе: {train_df['id'].nunique()}")
    print(f"Количество рядов в валидационном наборе: {val_df['id'].nunique()}")
    print(f"Количество рядов в тестовом наборе: {test_df['id'].nunique()}")
    print(f"Количество рядов в целевых значениях теста: {test_targets['id'].nunique()}")

    base_date = pd.to_datetime("2000-01-01")

    def convert_dates(series):
        return base_date + pd.to_timedelta(series.astype(int) - 1, unit="D")

    train_df["date"] = convert_dates(train_df["date"])
    val_df["date"] = convert_dates(val_df["date"])

    train_df = train_df.drop(columns=["estimator_name"], inplace=False)
    val_df = val_df.drop(columns=["estimator_name"], inplace=False)

    return train_df, val_df
