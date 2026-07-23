# Data directory

This directory contains the **reference input data** for the example. These files define the public transportation network together with the demand information used by the example.

The files in this directory are considered **read-only**. They constitute the reference dataset distributed with the example and are not modified by any of the scripts.

---

# Network description

## `metadata.json`

General information describing the scenario.

Typical contents include:

- scenario name,
- temporal horizon,
- global configuration parameters.

---

## `stops.csv`

Definition of all public transport stops.

Each row corresponds to one stop and provides its identifier, name, and geographic coordinates.

This file is used to construct the time-expanded network.

---

## `lines.csv`

Definition of the public transport lines operating in the network.

Each record identifies one transit line and its characteristics.

---

## `trips.csv`

Definition of the individual vehicle trips.

Each trip is associated with one transit line and one operating schedule.

---

## `stop_times.csv`

Timetable of every trip.

Each record specifies the arrival and departure times of a vehicle at a stop.

This file is the main input used to build the time-expanded graph.

---

## `time_bins.csv`

Definition of the departure-time bins used for demand modelling.

Each OD movement belongs to exactly one departure-time bin.

---

# Demand data

## `true_demand.csv`

This file contains the ground-truth origin–destination demand associated with the scenario.

It serves as the reference demand for the example and can be used to assess the quality of the estimated demand.

---

## `prior_demand.csv`

This file contains the a priori origin–destination demand.

It represents the initial estimate of the demand before any estimation procedure is applied.

Typically, `prior_demand.csv` differs from `true_demand.csv`, providing a meaningful estimation problem in which the algorithms attempt to recover the unknown true demand.
