# Kategoryzator Wydatków – Drzewo Decyzyjne (własna implementacja)

## Opis projektu

Automatyczna klasyfikacja transakcji bankowych na kategorie:

- **Jedzenie**
- **Transport**
- **Rozrywka**
- **Rachunki**

## Główne funkcje

- Wczytywanie danych z pliku CSV
- Trenowanie drzewa decyzyjnego na podstawie cech (kwota, długość opisu, słowa kluczowe)
- Klasyfikacja nowych transakcji
- Tryb interaktywny (wpisz opis i kwotę – program określi kategorię)
- Testy jednostkowe (pytest)

## Technologie

- Python 3.9+
- numpy, pandas, matplotlib
- pytest (testy)
- ruff (linting)
- pre-commit (jakość kodu)

# Raport z eksperymentu – Kategoryzator Wydatków

## 1. Opis przeprowadzonych testów

Przetestowano własną implementację drzewa decyzyjnego (algorytm ID3) na zbiorze 50 transakcji bankowych podzielonych na 4 kategorie powyżej.

**Parametry eksperymentu:**
- Liczba próbek treningowych: 50
- Podział danych: trening (100%, walidacja wewnętrzna drzewa)
- Maksymalna głębokość drzewa: 5
- Minimalna liczba próbek do podziału: 5

## 2. Uzyskane wyniki

Metryka | Wartość 
--------|---------
Dokładność (trening) | **87.76%** 
Liczba próbek treningowych | 50 
Maksymalna głębokość drzewa | 5 
Liczba testów jednostkowych | 3 (wszystkie przeszły) 

### Przykładowe klasyfikacje

Opis transakcji | Kwota | Przewidziana kategoria 
----------------|-------|-----------------------
Pizza Hut | 32 zł | Rozrywka 
Biedronka zakupy | 45 zł | Jedzenie 
MPK bilet miesięczny | 98 zł | Transport 
Energa prąd | 245 zł | Rachunki 

## 3. Wyniki testów jednostkowych
================================================= test session starts =================================================
collected 3 items

tests\test_decision_tree.py ... [100%]

================================================== 3 passed in 0.85s ==================================================


Wszystkie testy przeszły pomyślnie, co potwierdza poprawność implementacji:
- `test_entropy` – poprawność obliczania entropii
- `test_fit_and_predict` – działanie trenowania i predykcji
- `test_accuracy` – poprawność metryki dokładności

## 4. Wnioski

### Mocne strony:
- Własna implementacja drzewa decyzyjnego działa poprawnie i osiąga ~88% dokładności na zbiorze treningowym.
- Brak użycia gotowych bibliotek ML (scikit-learn, tensorflow, keras, pytorch).
- Algorytm ID3 z entropią i przyrostem informacji został zrozumiany i zaimplementowany samodzielnie.
- Kod przechodzi testy jednostkowe i spełnia standardy jakości (ruff, pre-commit).

### Słabe strony / ograniczenia:
- Dokładność mierzona tylko na zbiorze treningowym – brak niezależnego zbioru testowego.
- Relatywnie mała liczba próbek (50 transakcji) – ryzyko przeuczenia.
- Ekstrakcja cech oparta na prostych słowach kluczowych – w rzeczywistych systemach potrzebna byłaby lematyzacja i obsługa synonimów.

### Propozycje ulepszeń:
1. **Podział na zbiór treningowy i testowy** (np. 80/20) – obiektywna ocena generalizacji.
2. **Przycinanie drzewa (pruning)** – redukcja przeuczenia.
3. **Rozszerzenie słownika słów kluczowych** – więcej synonimów i wariantów.
4. **Zwiększenie zbioru danych** do >200 transakcji – lepsze uczenie.
5. **Dodanie walidacji krzyżowej** – bardziej stabilna ocena modelu.

## 5. Podsumowanie

Projekt pokazuje, że można zaimplementować użyteczny klasyfikator bez gotowych bibliotek ML. Drzewo decyzyjne jest interpretowalne i działa dobrze dla problemów z kilkoma kategoriami i dobrze zdefiniowanymi cechami. 

Rozwiązanie może być faktycznie używane do:
- Automatycznego budżetowania domowego
- Kategoryzacji wydatków w aplikacjach bankowych
- Edukacyjnego przykładu implementacji drzewa decyzyjnego od podstaw

## 6. Załączniki

- Kod źródłowy: `app/decision_tree.py`
- Dane treningowe: `data/transactions.csv`
- Testy jednostkowe: `tests/test_decision_tree.py`
- Instrukcja uruchomienia: `README.md`