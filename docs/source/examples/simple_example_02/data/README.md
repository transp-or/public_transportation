# Simple Example 02 data

This directory contains the reference input data for a larger synthetic public-transportation example. The files follow the same conventions and CSV schemas as `simple_example_01`.

The example is designed to be larger and structurally richer than `simple_example_01`. In particular, several origin–destination pairs have more than one plausible public-transport route. This makes the route-choice component of the assignment meaningful and gives the dispersion parameter a clearer role.

The files in this directory are reference inputs. They should be treated as read-only. Generated files, such as synthetic measurements and estimation results, should be written by the workflow scripts to their respective `results/` directories.

## Network design

The network contains ten stops arranged as a small stylized urban bus network:

- `A`: west terminal
- `B`: southwest terminal
- `C`: west interchange
- `D`: south interchange
- `E`: east interchange
- `F`: southeast interchange
- `G`: east terminal
- `H`: far east terminal
- `N`: north terminal
- `U`: university terminal

The network has five bus lines, each operated in both directions:

- `L1`: east–west line between `A` and `G`, via `C` and `E`.
- `L2`: south line between `B` and `H`, via `D` and `F`.
- `L3`: diagonal line between `A` and `H`, via `B`, `D`, and `F`.
- `L4`: connector line between `C` and `F`, via `D` and `E`.
- `L5`: north–university line between `N` and `U`, via `C` and `D`.

The service horizon is the morning period from 07:00 to 10:00. The lines have different headways, so route alternatives differ not only by in-vehicle travel time but also by waiting time and transfer opportunities.

## Route-choice structure

Several OD pairs are deliberately chosen to have competing routes. For example:

- `A -> H` can use the direct diagonal line `L3`, or a combination of `L1`, `L4`, and `L2` through the interchange stops.
- `A -> F` can use `L3` directly, or `L1` plus `L4` through `C` and `E`.
- `N -> H` can use `L5` and `L2` through `D`, or alternative transfer patterns involving `C`, `E`, and `F`.
- `B -> G` can use a path through `D`, `E`, and `G`, or a path involving the diagonal and east–west services depending on departure time.

This structure is intended to make the assignment sensitive to the route-choice dispersion parameter and to make boarding and alighting measurements informative about both OD demand and route choice.

## File descriptions

### `metadata.json`

General metadata for the scenario. It includes the title, description, timezone, cost unit, and additional notes about the design of the example.

### `stops.csv`

List of stops in the network.

Columns:

- `stop_id`: unique stop identifier.
- `name`: human-readable stop name.
- `lat`: latitude.
- `lon`: longitude.

### `lines.csv`

List of public-transport lines.

Columns:

- `line_id`: unique line identifier.
- `short_name`: human-readable line name.

### `trips.csv`

List of scheduled vehicle trips.

Columns:

- `trip_id`: unique trip identifier.
- `line_id`: identifier of the line operating the trip.
- `capacity`: vehicle capacity.

### `stop_times.csv`

Timetable for all trips.

Columns:

- `trip_id`: trip identifier.
- `stop_id`: stop served by the trip.
- `sequence`: order of the stop within the trip.
- `arrival_s`: arrival time in `HH:MM:SS` format.
- `departure_s`: departure time in `HH:MM:SS` format.

### `time_bins.csv`

Departure-time bins for OD demand.

Columns:

- `bin_id`: unique time-bin identifier.
- `start_s`: start time in `HH:MM:SS` format.
- `end_s`: end time in `HH:MM:SS` format.

The example uses six half-hour bins from 07:00 to 10:00.

### `true_demand.csv`

Ground-truth origin–destination demand.

Columns:

- `origin_stop_id`: origin stop.
- `dest_stop_id`: destination stop.
- `time_bin_id`: departure-time bin.
- `flow`: true passenger demand for the OD/time-bin combination.

This file represents the demand used to generate synthetic observations.

### `prior_demand.csv`

A priori origin–destination demand.

Columns:

- `origin_stop_id`: origin stop.
- `dest_stop_id`: destination stop.
- `time_bin_id`: departure-time bin.
- `flow`: prior passenger demand for the OD/time-bin combination.

The prior demand is deliberately biased relative to the true demand. It is smoother across time and underestimates some peak-period OD flows. This creates a non-trivial OD estimation problem.

## Scale of the example

The clean data contain:

- 10 stops,
- 5 lines,
- 126 scheduled trips,
- 504 stop-time records,
- 6 departure-time bins,
- 72 OD/time-bin demand records.

This is still small enough to inspect manually, but large enough to exercise transfer behavior, route choice, OD estimation, and post-processing reports.
