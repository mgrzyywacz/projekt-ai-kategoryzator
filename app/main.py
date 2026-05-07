import pandas as pd
import os
from app.decision_tree import DecisionTree
from app.data_loader import load_and_preprocess, prepare_features_labels, preprocess_dataframe

def main():
    print("=" * 60)
    print("Kategoryzator Wydatków – Drzewo Decyzyjne (własna implementacja)")
    print("=" * 60)

    print("\nWczytywanie danych treningowych")
    df_train = load_and_preprocess('data/transactions.csv')
    X_train, y_train, features = prepare_features_labels(df_train)

    print("Trenowanie drzewa decyzyjnego")
    tree = DecisionTree(max_depth=5, min_samples_split=5)
    tree.fit(X_train, y_train)

    accuracy = tree.accuracy(X_train, y_train)
    print(f"Dokładność na zbiorze treningowym: {accuracy:.2%}")

    try:
        print("\nKlasyfikacja nowych transakcji")
        df_new = load_and_preprocess('data/new_transactions.csv')
        X_new, _, _ = prepare_features_labels(df_new)
        predictions = tree.predict(X_new)

        print("\nWyniki klasyfikacji:")
        print("-" * 50)
        for i, (_, row) in enumerate(df_new.iterrows()):
            print(f"Opis: {row['opis']:<35} → {predictions[i]}")

    except FileNotFoundError:
        print("\nBrak pliku new_transactions.csv – pomijam klasyfikację nowych danych.")

    # Interaktywny tryb
    print("\nTryb interaktywny – wpisz opis transakcji, a ja określę kategorię.")
    print("   (wpisz 'exit' aby zakończyć)")

    while True:
        opis = input("\nOpis transakcji: ")
        if opis.lower() == 'exit':
            break

        kwota = float(input("   Kwota (zł): "))

        temp_df = pd.DataFrame([{
            'opis': opis,
            'kwota': kwota,
            'kategoria': 'unknown'
        }])

        temp_processed = preprocess_dataframe(temp_df)
        X_temp, _, _ = prepare_features_labels(temp_processed)
        pred = tree.predict(X_temp)
        print(f" Kategoria: {pred[0]}")


if __name__ == "__main__":
    main()
