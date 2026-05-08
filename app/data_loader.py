import os
import pandas as pd

def preprocess_dataframe(df):
    df["kwota"] = df["kwota"].astype(float)
    df["dlugosc_opisu"] = df["opis"].apply(len)
    df["czy_sklep_spozywczy"] = df["opis"].apply(
        lambda x: any(
            word in x.lower()
            for word in ["biedronka", "lidl", "żabka", "carrefour", "kaufland"]
        )
    ).astype(int)
    df["czy_transport"] = df["opis"].apply(
        lambda x: any(
            word in x.lower() for word in ["mpk", "paliwo", "benzyna", "tramwaj", "bus"]
        )
    ).astype(int)
    df["czy_rachunek"] = df["opis"].apply(
        lambda x: any(
            word in x.lower()
            for word in ["prąd", "gaz", "woda", "tvn", "netflix", "spotify"]
        )
    ).astype(int)
    df["czy_rozrywka"] = df["opis"].apply(
        lambda x: any(
            word in x.lower()
            for word in ["kino", "pub", "restauracja", "piwo", "koncert"]
        )
    ).astype(int)
    df["kwota_kategoria"] = (
        pd.cut(df["kwota"], bins=[0, 20, 100, float("inf")], labels=[0, 1, 2])
        .astype(int)
    )
    return df

def load_and_preprocess(filepath):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    full_path = os.path.join(project_dir, filepath)

    encodings = ["utf-8", "latin-1", "cp1250", "iso-8859-2"]
    df = None

    for encoding in encodings:
        try:
            df = pd.read_csv(full_path, encoding=encoding)
            print(f"Wczytano plik z kodowaniem: {encoding}")
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        msg = f"Nie można odczytać pliku {full_path}"
        raise Exception(msg)

    return preprocess_dataframe(df)

def prepare_features_labels(df):
    feature_columns = [
        "kwota",
        "dlugosc_opisu",
        "czy_sklep_spozywczy",
        "czy_transport",
        "czy_rachunek",
        "czy_rozrywka",
        "kwota_kategoria",
    ]

    x = df[feature_columns].values
    y = df["kategoria"].values

    return x, y, feature_columns
