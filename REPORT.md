
## Poprawiony `REPORT.md`

```markdown
# Podsumowanie eksperymentu – Kategoryzator Wydatków

## 1. Opis przeprowadzonych testów

Przetestowano własną implementację drzewa decyzyjnego (algorytm ID3) na zbiorze 50 transakcji bankowych podzielonych na 4 kategorie:

- Jedzenie
- Transport
- Rozrywka
- Rachunki

**Parametry eksperymentu:**

| Parametr | Wartość |
| --- | --- |
| Liczba próbek treningowych | 50 |
| Maksymalna głębokość drzewa | 5 |
| Minimalna liczba próbek do podziału | 5 |
| Liczba cech | 7 |

**Przebieg testów:**

1. Wczytanie danych z pliku `data/transactions.csv`
2. Preprocessing danych (ekstrakcja cech: długość opisu, słowa kluczowe, kategoria kwoty)
3. Trenowanie drzewa decyzyjnego na całym zbiorze
4. Ocena dokładności na zbiorze treningowym
5. Testy jednostkowe kluczowych funkcji (entropia, fit, predict, accuracy)

## 2. Uzyskane wyniki

### Wyniki główne

| Metryka | Wartość |
| --- | --- |
| Dokładność (trening) | **87.76%** |
| Liczba poprawnie sklasyfikowanych | 44 / 50 |
| Liczba błędnie sklasyfikowanych | 6 / 50 |

### Wyniki testów jednostkowych
collected 3 items
tests\test_decision_tree.py ... [100%]
3 passed in 0.85s


Testy jednostkowe potwierdzają poprawność:

- `test_entropy` – poprawność obliczania entropii
- `test_fit_and_predict` – działanie trenowania i predykcji
- `test_accuracy` – poprawność metryki dokładności

### Przykładowe klasyfikacje

| Opis transakcji | Kwota | Przewidziana kategoria | Wynik |
| --- | --- | --- | --- |
| Pizza Hut | 32 zł | Rozrywka | poprawnie |
| Biedronka zakupy | 45 zł | Jedzenie | poprawnie |
| MPK bilet miesięczny | 98 zł | Transport | poprawnie |
| Energa prąd | 245 zł | Rachunki | poprawnie |
| Spotify premium | 19.99 zł | Rozrywka | poprawnie |
| Orlen paliwo | 156 zł | Transport | poprawnie |

## 3. Wnioski

### Co się udało?

- **87.76% dokładności** – bardzo dobry wynik jak na prostą implementację i tylko 50 próbek
- **Brak gotowych bibliotek ML** – nie użyto scikit-learn, tensorflow, keras, pytorch
- **Testy jednostkowe** – 3/3 przechodzą, kod jest stabilny
- **Walidacja jakości** – kod spełnia standardy ruff i pre-commit

### Co można poprawić?

- **Brak zbioru testowego** – dokładność mierzona tylko na treningowym (możliwe przeuczenie)
- **Mała liczba danych** – 50 transakcji to za mało dla w pełni wiarygodnej oceny
- **Brak walidacji krzyżowej** – nie sprawdzono stabilności modelu


### Podsumowanie końcowe

Rozwiązanie może być faktycznie używane do:

- Automatycznego budżetowania domowego
- Kategoryzacji wydatków w aplikacjach bankowych
- Edukacyjnego przykładu implementacji drzewa decyzyjnego od podstaw
