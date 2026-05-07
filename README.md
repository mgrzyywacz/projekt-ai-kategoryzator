# Kategoryzator Wydatków – Drzewo Decyzyjne 

## Opis projektu

Automatyczna klasyfikacja transakcji bankowych na kategorie:
- **Jedzenie**
- **Transport**
- **Rozrywka**
- **Rachunki**

Projekt wykorzystuje **własną implementację drzewa decyzyjnego (algorytm ID3)** – bez użycia gotowych bibliotek ML takich jak scikit-learn, tensorflow czy pytorch.

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
- pre-commit (jakość kodu)

## Instalacja

### 1. Sklonuj repozytorium

```bash
git clone https://github.com/mgrzyywacz/projekt-ai-kategoryzator.git
cd projekt-ai-kategoryzator