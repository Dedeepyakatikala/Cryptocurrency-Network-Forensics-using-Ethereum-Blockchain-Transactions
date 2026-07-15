# Cryptocurrency Network Forensics using Ethereum Blockchain Transactions

A scalable Big Data framework for detecting anomalous and fraudulent behavior in Ethereum blockchain transactions — combining distributed data processing, rule-based heuristics, unsupervised machine learning, graph analysis, and time-series analytics.

---

## Overview

Ethereum is one of the most widely adopted blockchain platforms, supporting smart contracts, DApps, DeFi, and NFTs. Its transparency and scale, however, come with a cost: the network generates massive volumes of transaction data that are both a monitoring challenge and an attack surface for bot-driven transactions, wash trading, smurfing, zero-value transaction spam, and pump-and-dump market manipulation.

This project builds an end-to-end, **Big Data–driven forensics pipeline** that ingests, cleans, and analyzes real Ethereum blockchain data (blocks + transactions, ~94M+ rows) using Apache Spark, then layers rule-based detection, unsupervised ML, deep learning, graph analytics, and time-series methods on top to surface suspicious wallets, blocks, and transaction patterns — without relying on labeled fraud data.

## Problem Statement & Objectives

**Problem:** Ethereum's high-volume, high-velocity, and evolving transaction data makes manual monitoring and traditional (single-dimensional, rule-only, small-dataset) fraud detection approaches ineffective at real-world scale.

**Objectives:**
- Analyze large-scale Ethereum blockchain data using Big Data technologies (Apache Spark / PySpark)
- Store and preprocess blockchain data efficiently using distributed, Hadoop-style Parquet storage
- Engineer transaction-level, wallet-level, and block-level behavioral features
- Detect known fraud patterns via rule-based heuristics: bots, smurfing, wash trading, zero-value spam
- Detect unknown/evolving anomalies via unsupervised ML (Isolation Forest, Autoencoders)
- Perform graph-based network analysis to uncover circular flows and coordinated wallet behavior
- Perform time-series analysis to detect pump-and-dump market manipulation

## Research Gap & Novelty

Existing literature on Ethereum fraud detection tends to suffer from:
- **Limited scalability** — most studies evaluate on small, curated, or labeled datasets
- **Dependence on labeled data** — supervised models are impractical given how scarce and outdated labeled blockchain fraud data is
- **Single-dimensional detection** — most work isolates one lens (graph-only, classification-only, smart-contract-only)
- **Narrow fraud-type focus** — e.g. scam detection *or* de-anonymization, rarely both plus bots, smurfing, wash trading, and market manipulation together
- **No integrated temporal analysis** — transaction graphs analyzed without regard to time-dependent dynamics

This project's novelty is combining **all of the above into one unified, unsupervised, scalable pipeline**: Spark-based distributed processing on real (not sampled) data, Isolation Forest + Autoencoder anomaly detection with no labels required, multi-level (transaction/wallet/block) feature engineering, and simultaneous rule-based + ML + graph + time-series detection across a wide range of anomaly types.

## Dataset

| Property | Value |
|---|---|
| Name | Ethereum Blockchain Parquet Dataset |
| Source | [Hugging Face — `vnegi10/Ethereum_blockchain_parquet`](https://huggingface.co/datasets/vnegi10/Ethereum_blockchain_parquet) |
| Format | Apache Parquet (columnar) |
| Size | ~20.9 GB |
| License | GPL-3.0 |
| Time Coverage | 17 Nov 2016 – 25 Mar 2025 |
| Scale | 94M+ rows across blocks and transactions |
| Join Key | `block_number` |

**Blocks schema:** `block_hash`, `author`, `block_number`, `gas_used`, `extra_data`, `timestamp`, `base_fee_per_gas`, `chain_id`

**Transactions schema:** `block_number`, `transaction_index`, `transaction_hash`, `nonce`, `from_address`, `to_address`, `value_binary` / `value_string` / `value_f64`, `input`, `gas_limit`, `gas_used`, `gas_price`, `transaction_type`, `max_priority_fee_per_gas`, `max_fee_per_gas`, `success`, `n_input_bytes`, `n_input_zero_bytes`, `n_input_nonzero_bytes`, `chain_id`

> The raw dataset is **not included** in this repository due to size. The notebook downloads it directly from Hugging Face at runtime.

## System Architecture

```
Hugging Face Hub (Parquet dataset)
        │  download
        ▼
Apache Spark Session (PySpark, local[*])
        │
        ▼
Distributed Ingestion  →  Blocks DF + Transactions DF
        │
        ▼
Hadoop-style Parquet Storage (data lake emulation)
        │
        ▼
Schema Alignment & Join (on block_number) → Unified Analytical Table
        │
        ▼
Data Cleaning & Preprocessing (null handling, corrupt-record removal)
        │
        ▼
Feature Engineering (transaction / wallet / block level)
        │
        ▼
Exploratory Data Analysis
        │
        ├──▶ Rule-Based Anomaly Detection (bots, smurfing, wash trading, zero-value spam)
        ├──▶ Machine Learning Detection (Isolation Forest — wallet & block level)
        ├──▶ Deep Learning Detection (Autoencoder reconstruction error)
        ├──▶ Graph-Based Network Analysis (NetworkX — circular flows, wash trading)
        └──▶ Time-Series / Temporal Analysis (rolling-window pump-and-dump detection)
        │
        ▼
Detected Anomalies & Results (flagged wallets, suspicious blocks, pump-and-dump events)
```

## Methodology

1. **Data Collection & Ingestion** — Public Ethereum block/transaction data (Parquet) is ingested with PySpark for distributed, scalable processing.
2. **Data Storage & Preprocessing** — Data is stored in Hadoop-style Parquet storage; incomplete/corrupted records are removed; critical fields (value, gas usage/limit, addresses) are cleaned.
3. **Feature Engineering** —
   - *Transaction-level:* ETH value, gas used/limit, transaction fee, success/failure
   - *Wallet-level:* transaction count, average value, failure rate, gas usage variance
   - *Block-level:* average gas used, gas utilization, base fee per gas, transactions per block
4. **Exploratory Data Analysis** — Statistical summaries and visualizations (value distributions, gas limit vs. gas used, success/failure ratios) to surface skew, spikes, and abnormal trends.
5. **Rule-Based Anomaly Detection** — Threshold heuristics for bots (high frequency + low value), smurfing (many small transfers), zero-value transaction spam, and wash trading (repeated sender–receiver pairs).
6. **Machine Learning-Based Anomaly Detection** — Isolation Forest and Autoencoder models flag anomalies without labeled data (see below).
7. **Graph-Based Network Analysis** — Transactions modeled as a directed graph (wallets = nodes, transfers = edges) to detect circular fund flows and coordinated interaction clusters.
8. **Time-Series & Temporal Analysis** — Transactions aggregated into fixed windows; rolling averages reveal sudden spikes/drops indicative of pump-and-dump activity.
9. **Visualization & Interpretation** — Histograms, bar charts, scatter plots, time-series plots, and transaction graphs used throughout for explainable, interpretable results.

## Anomaly Detection Techniques

### Isolation Forest
An unsupervised algorithm built on the principle that anomalies are *easier to isolate* than normal points. It randomly selects a feature and split value to build multiple isolation trees (iTrees); a data point's average path length across trees is its anomaly signal — short paths indicate anomalies, long paths indicate normal behavior.
- **Advantages:** highly scalable, low computational cost, integrates naturally with Spark workflows
- **Limitations:** less effective on complex temporal patterns alone — best combined with autoencoders, graph analysis, and time-series methods
- Applied at both **wallet level** (transaction count, avg value, failure rate, gas variance) and **block level** (avg gas used, base fee, tx count, gas utilization)

### Autoencoder
An unsupervised deep learning model (Encoder → Latent Bottleneck → Decoder) trained to reconstruct normal transaction/wallet behavior.
- Trained only on normal/majority data (input = output, self-supervised) using MSE loss and gradient descent
- **Reconstruction error** is the anomaly score: normal data reconstructs with low error, anomalous data with high error
- Data points above a chosen error threshold (top ~2% in this project) are classified as anomalous

### Graph-Based Network Analysis
Ethereum transactions are modeled as a directed graph (NetworkX) with wallets as nodes and ETH transfers as edges, enabling detection of:
- Circular fund transfers (A → B → A cycles) — indicative of self-trading / wash trading
- Repeated sender–receiver pairs — wash trading indicators
- Smurfing wallets — many low-value transfers from a single source

### Pump-and-Dump Detection
1. Transactions are bucketed into **10-minute windows**; per-window transaction count and total/average ETH volume are computed.
2. A **1-hour rolling baseline** (previous 6 windows) is calculated for both metrics.
3. A **pump** is flagged when transaction spike > 1.5× and ETH volume spike > 1.5× the baseline.
4. A **dump** is confirmed when the *following* window shows a sharp decline (future transaction change < −0.5).
5. Windows satisfying both conditions are flagged as pump-and-dump events and validated visually via time-series plots.

## Big Data Tools & Technologies

| Category | Tool / Technology |
|---|---|
| Big Data Engine | Apache Spark |
| Storage Layer | Hadoop-style distributed storage (HDFS-compatible) |
| File Format | Apache Parquet |
| Data Ingestion | Hugging Face Datasets |
| Programming Interface | PySpark |
| Query Engine | Spark SQL (Catalyst optimizer) |
| Graph Analytics | NetworkX |
| ML / Anomaly Detection | scikit-learn (Isolation Forest), PyTorch / MLP-based autoencoder |
| Environment | Google Colab |

### The 5 V's of Big Data (as addressed in this project)
- **Volume** — millions of transactions/blocks, several GB of data, handled via distributed storage and in-memory computation
- **Velocity** — continuous block generation handled via parallel batch processing
- **Variety** — structured (transaction/block fields) and semi-structured (input/calldata) data unified via schema-aware Parquet storage
- **Veracity** — null handling, corrupt-record filtering, and schema enforcement ensure reliable inputs to downstream models
- **Value** — statistical, ML, deep learning, and graph techniques converted into actionable fraud/anomaly intelligence

## Spark Environment Setup

```bash
# Environment variables
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export HADOOP_HOME=$HOME/hadoop
export SPARK_HOME=$HOME/spark
export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$JAVA_HOME/bin:$SPARK_HOME/bin:$SPARK_HOME/sbin
source ~/.bashrc

# Start Spark (standalone cluster)
cd $HOME/spark
./sbin/start-master.sh
./sbin/start-worker.sh spark://localhost:7077

# Launch a shell
spark-shell      # Scala
pyspark          # Python

# Submit a job
spark-submit --master local[*] ethereum_forensics.py

# Monitor jobs
# Spark UI → http://localhost:4040
```

The notebook itself creates a local Spark session:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Ethereum Big Data Anomaly Detection") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()
```

## Results

The pipeline successfully surfaced the following, all visualized in the notebook and detailed in the report:

- **Transaction Value Distribution** — strongly right-skewed with a long tail of outlier "whale" transactions (>1000 ETH)
- **Transaction Success vs. Failure** — the large majority of transactions succeed; wallets with disproportionately high failure counts were flagged as wallet-level anomalies
- **Gas Limit vs. Gas Used** — most transactions use well below their gas limit; transactions/blocks consistently near their limit indicate abnormal or complex execution
- **Whale Transactions** — high-value transfers (>1000 ETH) are rare and tightly concentrated, confirming their anomalous nature
- **Bot Wallet Detection** — wallets combining very high transaction frequency with unusually low average value were flagged as likely automated agents
- **Wash Trading Indicators** — repeated sender–receiver pairs and circular transaction cycles (A → B → A) detected via graph analysis
- **Isolation Forest** — flagged a small subset of wallets and blocks with highly anomalous aggregated behavior
- **Autoencoder** — flagged wallets with high reconstruction error, capturing non-linear behavioral deviations traditional methods missed
- **Money Laundering Patterns** — smurfing wallets (many low-value transfers) and circular fund flows identified via the transaction graph
- **Pump-and-Dump Detection** — multiple time windows across 2018–2024 flagged with sharp transaction/ETH volume spikes followed by rapid declines

*(See `D05_report.pdf` §4 and `D05_final.pptx` slides 20–24 for the full set of figures.)*


## Repository Contents

| File | Description |
|---|---|
| `Ethereum_Anomaly_Detection_BigData.ipynb` | Full Spark/PySpark pipeline: ingestion, cleaning, feature engineering, EDA, rule-based detection, Isolation Forest, Autoencoder, graph analysis, and pump-and-dump detection |
| `D05_report.pdf` | Full project report (introduction, literature survey, methodology, results, references) |
| `D05_final.pptx` | Final presentation slides |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Excludes large data/cache directories from version control |

## Setup & Usage

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
```

The notebook is designed to run in **Google Colab** (or any environment with Java 11 available) and will:
1. Install `openjdk-11-jdk-headless`, PySpark, and Hugging Face Hub libraries
2. Create a local Spark session
3. Stream/download the required Parquet files directly from Hugging Face (`vnegi10/Ethereum_blockchain_parquet`)
4. Run the full ingestion → cleaning → feature engineering → detection pipeline

> Given the dataset's 20.9 GB size, the notebook supports partial loading to avoid memory issues; adjust the file selection in the ingestion cell if running with limited resources.

## Future Work

- **Temporal Graph Neural Networks (GNNs)** to better capture evolving transaction patterns and dynamic wallet interactions
- **Real-time streaming analytics** via Spark Structured Streaming for early fraud detection instead of batch-only analysis
- **Multi-chain / DeFi support**, including cross-chain bridges, to detect laundering activity spanning multiple blockchains
- **Adaptive thresholding & reinforcement learning** to reduce false positives and dynamically tune detection sensitivity
- **Collaboration with on-chain labeling/regulatory datasets** to validate detected anomalies and improve evaluation


## License

This project was completed for academic purposes as part of a B.Tech capstone project at Amrita Vishwa Vidyapeetham. The underlying dataset is used under its GPL-3.0 license.
