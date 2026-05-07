import pandas as pd

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)

    df['kwota'] = df['kwota'].astype(float)
    df['dlugosc_opisu'] = df['opis'].apply(len)
    df['czy_sklep_spozywczy'] = df['opis'].apply(
        lambda x: any(
            word in x.lower() for word in ['biedronka', 'lidl', 'żabka', 'carrefour', 'kaufland'])
    ).astype(int)
    df['czy_transport'] = df['opis'].apply(
        lambda x: any(word in x.lower() for word in ['mpk', 'paliwo', 'benzyna', 'tramwaj', 'bus'])
    ).astype(int)
    df['czy_rachunek'] = df['opis'].apply(
        lambda x: any(
            word in x.lower() for word in ['prąd', 'gaz', 'woda', 'tvn', 'netflix', 'spotify'])
    ).astype(int)
    df['czy_rozrywka'] = df['opis'].apply(
        lambda x: any(
            word in x.lower() for word in ['kino', 'pub', 'restauracja', 'piwo', 'koncert'])
    ).astype(int)

    df['kwota_kategoria'] = pd.cut(df['kwota'], bins=[0, 20, 100, float('inf')],
                                   labels=[0, 1, 2]).astype(int)
    return df


def prepare_features_labels(df):
    feature_columns = ['kwota', 'dlugosc_opisu', 'czy_sklep_spozywczy',
                       'czy_transport', 'czy_rachunek', 'czy_rozrywka', 'kwota_kategoria']
    X = df[feature_columns].values
    y = df['kategoria'].values

    return X, y, feature_columns
