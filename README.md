# automatic_piano_transcription
This is a repository for a project titled "Automatic Piano Transcription" associated with the curriculum for the "Neural Networks 2" class at the University of Science and Technology in Wrocław

---
- Date begun: March 2024
- Date ended: -
- Author(s): Jakub Wolski, Patryk Gawłowski
---

## Zarys prezentacji projektu z sieci neuronowych
### Wprowadzenie
- sformułowanie problemu (co chcemy osiągnąć)
    - automatyczna polifoniczna transkrypcja pianina
    - krótki opis na czym to polega
- dlaczego chcemy osiągnąć
    - jest to przydatne narzędzie - trudne do wykonania nawet dla ekspertów
    - ...
### Opis problemu
- co trzeba wykonać
---
- wprowadzić plik muzyczny
- na jego podstawie znaleźć jakie dźwięki są grane (polifoniczne)
- móc przerzucić to na piano roll
- wygenerować plik MusicXML (czyli jako partyruta)
#### Oczekiwane wyzwania(?)
- z jakimi problemami możemy się spotkać
- proponowane sposoby radzenia sobie z nimi
---
- rozpoznawanie nut na podstawie obrazu (spektrogramu)
    - zlozone z podstawowego tonu i wyzszych harmonicznych
    - interferencja pomiędzy sąsiadującymi harmonicznymi (analogiczne do problemu nachodzących na siebie bądź transparentnych obiektów przy klasyfikacji obrazów)
---
- overfitting do podawanych danych / brak różnorodności w wariacji sekwencji nut
    - For our train set, we achieve 74.09% F1-score and 99.81% accuracy using the third learning rate schedule (manual step decay). We believe that this indicates that we have overfit to our training set, and we believe that this can be remedied with more regularization and by exposing our model to more varied data. There is also an incredibly large space of possible notes sequences and combinations that could potentially appear in a song. Since our dataset only spans 6 composers and 138 songs, one of the issues that could be hindering our acoustic models performance would be the limited note patterns present in our data. We can remedy this by either acquiring more MIDI files of other songs across many more composers, or by generating synthetic data to train our model with. With synthetically generated data, we would then be able to ensure that our model is continuously exposed to a wide variety of acoustical signals, and thus we would increase the distribution of data that our model is trained on.
### Opis zaproponowanego rozwiązania problemu
#### Ogólne
- system podzielony na dwie części
    - modelowanie akaustyczne zeby zidentyfikować wysokość dźwięku w polifonizcnej muzyce
    - generowanie partytury aby przekonwertować wygenerowane dźwięki pianina (ang. _piano roll_) w "naturalne" nuty
---
- CNN
- Long-Short Term Memory
- DNN
- RNN
---
- agregowanie klatek w celu osiągnięcia informowanej predykcji
---
- generowanie partytury
    - problem: podczas grania jest element ekspresyjny powodujący dosłowne branie wygnenrowanych dźwięków i tworzenie z ich partytury nienaturalne
    - obejście: naturalne generowanie partytury
        - selekcja tempa z grupowaniem rytmicznym
        - wygładzanie
#### Szczegółowe (krok po kroku)
##### 1. Dane
- wejście - pliki muzyczne .wav
- labelki - pliki tekstowe .txt
##### 2. Preprocessing
- downsampling to 16kHz
- tranformata CQT
- ile oktaw/filtrów/hop length
- ile klatek na sekunde
- grupowanie wejść (kontekst)
##### 3. Model akustyczny
- CNN
    - predykcja dla środkowej klatki jest zwiększona podawając informację sąsiadującą
- DNN
- wartstwa wyjściowa
    - 88 neuronów
    - predykcja czy dany klawisz był naciśnięty czy nie
##### 4. Uczenie
- learning plateau - z powodu zbyt dużego learning rate
    - wprowadzenie learning rate schedule
        1. starting at a learning rate of 0.1 and then halving every 5 epochs
        2. starting at a learning rate of 0.05 and then halving every 10 epochs
        3. using a learning rate of 0.05 for the first 5 epochs, dropping to 0.025 for the next 7, dropping to 0.0125 for the next 9, and dropping to 0.00625 for the remaining epochs
##### 5. Metryki sprawdzające jakość predykcji
- accuracy
- F-1 score
    - czułość: $$R = \frac{\text{Prawdziwie Pozytywne}}{\text{Prawdziwie Pozytywne} + \text{Fałszywie Negatywne}}$$
    - precyzja: $$P = \frac{\text{Prawdziwie Pozytywne}}{\text{Prawdziwie Pozytywne} + \text{Fałszywie Pozytywne}}$$
    - F1-Score: $$F1 = \frac{2 \cdot R \cdot P}{R + P}$$
    - pozwala na wyważenie nierównowagi związanej z przeważającą ilością klas zerowych (większość klawiszy nie jest grana podczas grania) podczas sprawdzania jakości naszego alogrytmu
    - samo accuracy jest niewystarczające
        - jeśli na wyjściu byłyby same 0 to acc=96%
##### 6. Generacja partytury
- wzięcie wyjścia modelu akustycznego i utworzenie 'naturalnej' partytury
###### 6.1 Grupowanie rytmiczne
- wybiera tempa najlepiej dopasowane do nut
###### 6.2 Wygładzanie
- natural language model
    - Hidden Markov Model
### Oczekiwane rezultaty
- jak wyjdzie, co ma robić
- jeśli jest to zaczerpnięte z papieru to może jakieś porównania (przy jakiejś zmianie)
---
- wyniki
    - acc ~90+%
    - F1 ~50%
#### Polepszenia
- zmiany parametrów


