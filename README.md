# GPS Navigation System for Madrid Streets

## Overview

This project implements a GPS navigation system for Madrid’s street network, modeling the city as a weighted graph where intersections are vertices and street segments are edges. It supports two optimized routing modes shortest distance (Euclidean segment length) and fastest route (travel time computed from length and road-type speed limits), both powered by Dijkstra’s algorithm.

To build a clean, navigable network from official datasets, the system includes a data-processing pipeline that normalizes complex address strings (via regular expressions) and unifies near duplicate coordinates efficiently using KD-Tree spatial indexing. In addition to real-time routing, the library provides minimum spanning tree analysis through Prim’s and Kruskal’s algorithms for network optimization.

Key features include:
- Two routing formulations: shortest distance vs fastest route
- Street data processing and graph representation from official datasets
- Interactive command-line interface
- Visual route mapping with matplotlib
- MST tooling (Prim/Kruskal) for connectivity and optimization


![Mapa de Madrid](Madrid.png)

## Key Features

* **Dual routing options**:
  - Shortest path by distance (Euclidean)
  - Fastest route considering street types and speed limits
* **Address geocoding** - Convert street addresses to coordinates
* **Turn-by-turn navigation** with distance estimates
* **Interactive visualization** of routes on Madrid's street map
* **Efficient data processing** of large city datasets

## Repository Structure

```
GPSApp/
├── dgt_main.py # Data preprocessing and cleaning
├── callejero.py # Street network representation and utilities
├── grafo.py # Graph implementation and algorithms
├── gps.py # Main application and user interface
├── CRUCES.py # Processed Datasets
├── DIRECCIONES.py # Processed Datasets
├── datasets/ # Sample data files
│ ├── CALLEJERO_VIGENTE_CRUCES_202310.csv
│ └── CALLEJERO_VIGENTE_NUMERACIONES_202310.csv
├── requirements.txt # Python dependencies
├── Informe_GPSApp.pdf # Project documentation
├── Madrid.png # Madrid representation graph
└── README.md # This file
```

## Setup and Installation

1. **Clone the repository**

```bash
git clone https://github.com/Ulisesdz/GPS_App.git
cd GPS_App
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Prepare datasets**

```bash
python dgt_main.py
```

4. **Run the application** 

```bash
python gps.py
```

## Usage
1. When prompted, enter:
- Origin street name and number
- Destination street name and number

2. Select routing preference:
- 1 for shortest distance
- 2 for fastest route

3. The system will:
- Display turn-by-turn instructions
- Show total distance and estimated time
- Generate a visual map of the route

## Data Processing Pipeline
1. Preprocessing (dgt_main.py):
- Cleans street names and addresses
- Normalizes coordinate formats
- Handles special characters and accents

2. Graph Construction (callejero.py):
- Creates vertexes for street intersections
- Adds edges with distance and speed attributes
- Optimizes spatial queries with KD-Tree

3. Routing (grafo.py):
- Implements Dijkstra's algorithm
- Calculates both distance and time-optimized paths


---




