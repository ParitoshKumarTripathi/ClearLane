# Methodology

## Unit of prediction

One model row represents one 500 m zone at one two-hour forecast origin. The targets are the violation count and obstruction units in the following two-hour window.

## Cleaning

- De-duplicate by record ID.
- Retain coordinates inside the configured Bengaluru bounding box.
- Remove rejected and duplicate validation records.
- Convert timestamps from UTC to Asia/Kolkata.
- Prefer corrected vehicle fields where available.
- Retain only records containing a recognized parking offence.

## Impact construction

Each record receives:

- an offence severity;
- a vehicle-size factor;
- a junction, main-road or sensitive-place factor;
- a peak-hour factor.

For records with multiple parking offences, the maximum offence weight is retained and 25% of additional weights is added. This avoids treating one stopped vehicle as several independent obstructions while retaining the signal from compound offences.

## Features

- Target-hour and target-weekday cyclical encodings
- Previous-window, previous-day and previous-week lags
- One-day, seven-day and twenty-eight-day rolling means
- Short-term versus long-term trend ratios
- Junction, main-road, high-severity and heavy-vehicle shares
- Repeat-vehicle rate
- Active-device and active-officer intensity
- Record-confidence history
- Zone historical mean and accumulated observations

## Leakage control

- Zones are selected using records before the holdout start.
- Target values are shifted one window forward.
- Lag 1 is the completed window immediately before the prediction target.
- Rolling features end at the forecast origin.
- April target windows are excluded from model fitting.

## Models

The hotspot model uses Poisson loss because the output is a non-negative count. It now uses a slightly larger, more regularized histogram gradient boosting model and a wider deterministic sample of empty windows. The impact model learns `log1p(obstruction_units)` with squared-error loss and transforms predictions back with `expm1`; this prevents rare high-impact windows from producing unstable extreme predictions. Impact prediction is exported as a weighted ensemble of a compact ranker and a regularized ranker, which improves holdout impact error and operational capture while keeping deployment lightweight. Every positive training window is retained, empty windows are sampled deterministically, and inverse sampling weights preserve their population contribution.

## Evaluation

ClearLane is evaluated on:

- MAE, RMSE and Poisson deviance;
- rank correlation;
- Precision@K for predicted versus actual top zones;
- weighted Capture@K, the share of actual obstruction occurring in recommended zones.

Operational ranking metrics are more important than aggregate error because patrol capacity is limited.
