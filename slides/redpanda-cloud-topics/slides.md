---
theme: default
title: "Under the Hood: Cloud Topics Architecture"
info: A high level view of how Cloud Topics works
routerMode: hash
---

# Cloud Topics

Streaming Data Meets Object Storage

<br>

**Tyler Rockwood** — Redpanda Data

---

# What is Redpanda?

A **Kafka-compatible** streaming data platform built from the ground up in C++

- Drop-in replacement for Apache Kafka — same API, no JVM
- **Thread-per-core** architecture (Seastar) — no context switching, no GC pauses
- **Raft consensus** for every partition — strong consistency & fault tolerance
- **Tiered Storage** — offload historical data to object storage

<br>

> A fault-tolerant transaction log for storing event streams

---

# Redpanda Architecture

```
                  ┌───────────────────────────────────────┐
Producers ──────► │           Redpanda Cluster            │ ◄── Consumers
(Kafka API)       │                                       │     (Kafka API)
                  │  ┌─────────┐ ┌─────────┐ ┌─────────┐  │
                  │  │ Node 1  │ │ Node 2  │ │ Node 3  │  │
                  │  │         │ │         │ │         │  │
                  │  │ P0(L)   │ │ P0(F)   │ │ P0(F)   │  │
                  │  │ P1(F)   │ │ P1(L)   │ │ P1(F)   │  │
                  │  │ P2(F)   │ │ P2(F)   │ │ P2(L)   │  │
                  │  └─────────┘ └─────────┘ └─────────┘  │
                  └───────────────────────────────────────┘
                                    │
                             Tiered Storage
                                    ▼
                          ┌─────────────────┐
                          │ Object Storage  │
                          │  (S3 / GCS)     │
                          └─────────────────┘
```

Each partition is a **Raft group** — one leader (L), multiple followers (F)

---

# The Raft Replication Tax

Every produce request replicates data across availability zones via Raft

```
                     AZ-1           AZ-2           AZ-3
                  ┌────────┐    ┌────────┐    ┌────────┐
  Producer ──►    │ Node 1 │───►│ Node 2 │    │        │
                  │  (L)   │─────────────────►│ Node 3 │
                  └────────┘    └────────┘    └────────┘
                       Cross-AZ Network Traffic 💸
```

<br>

- Cloud providers charge for **cross-AZ data transfer**
- At high write throughput, networking is **70–90%** of total cost
- Every byte of data crosses the network **twice** (to each follower)

---

# Why Cloud Topics?

**What if we could skip the cross-AZ replication tax?**

<br>

| | Standard Topics | Cloud Topics |
|---|---|---|
| **Data replication** | Raft (cross-AZ network) | Object storage (S3/GCS) |
| **Latency** | Single-digit ms | Sub-second |
| **Cost driver** | Network transfer | Storage PUTs/GETs |
| **Ideal for** | Low-latency trading, real-time | Logs, analytics, CDC, high-throughput |

<br>

One customer achieved **>50% infrastructure savings** and **3x less CPU** usage

---

# Cloud Topics: The Key Idea

Separate **where metadata is stored** from **where data is stored**

```
  Standard Topic:    Data + Metadata ──► Raft Log (local disk, replicated)

  Cloud Topic:       Data ────────────► Object Storage (S3/GCS)
                     Metadata ─────────► Raft Log (local disk, replicated)
```

<br>

- Data goes directly to object storage — **no cross-AZ replication**
- Only lightweight metadata (pointers) go through Raft
- Same **transactions & idempotency** guarantees — Raft still handles correctness

---

# The 30,000-Foot View

```
                     ┌───────────────────────────────────────┐
  Producer ────►     │           Redpanda Broker             │
                     │                                       │
                     │  1. Batch in memory                   │
                     │  2. Upload to Object Storage (L0)     │──► Object Storage
                     │  3. Write placeholder to Raft         │       (L0 Files)
                     │  4. ACK to producer                   │          │
                     │                                       │     Reconciler
                     │  Raft Log                             │          │
                     │  ┌───────────────────────┐            │          ▼
                     │  │ ptr │ ptr │ ptr │ ... │            │     (L1 Files)
                     │  └───────────────────────┘            │
                     │                                       │
  Consumer ◄────     │  Read from cache / L0 / L1            │◄── Object Storage
                     └───────────────────────────────────────┘
```

---

# Step 1: The Write Path

Optimized for fast, cheap ingest

```
  Producer
     │
     ▼
  Kafka API Layer
     │
     ▼
  Cloud Topics Subsystem
     │
     ├──► Batch in memory (time window ~0.25s OR size ~4MB)
     │    Batches across ALL partitions and topics
     │
     ▼
  Upload batch to S3 ──────────────────────► L0 File
     │                                    (multi-partition)
     ▼
  Write placeholder to Raft Log
  (filename + offset per partition)
     │
     ▼
  ACK to Producer ✓
```

Batching across partitions **minimizes PUT requests** to object storage

---

# Write Path: Strong Consistency

**How do transactions and idempotency work?**

The placeholder batch reuses the **normal produce path** through Raft

```
  ┌─────────────────────────────────────────────────┐
  │                    Raft Log                     │
  │                                                 │
  │  [batch 0] [batch 1] [placeholder] [batch 3] ...│
  │                          │                      │
  │                     points to L0                │
  │                     file in S3                  │
  └─────────────────────────────────────────────────┘
```

<br>

- Transactions: same Raft-based commit protocol
- Idempotency: same sequence number tracking
- **Data payload lives in the cloud, guarantees live in Redpanda**

---

# Step 2: The Reconciler

L0 files are optimized for **writes** — they contain data from many partitions

```
  L0 File (multi-partition, small)
  ┌───────────────────────────────┐
  │ TopicA-P0 │ TopicB-P1 │ ...   │   ◄── Scattered reads
  └───────────────────────────────┘
```

The **Reconciler** reorganizes L0 → L1 in the background

```
  L1 File (single-partition, large, sorted)
  ┌────────────────────────────────────────────┐
  │ TopicA-P0: offset 0 │ 1 │ 2 │ 3 │ ...      │   ◄── Sequential reads
  └────────────────────────────────────────────┘
```

L1 files are: **larger**, **co-located** by partition, **sorted** by offset

---

# L0 vs L1 Files

Think of it like an **LSM tree** for streaming data

```
                    Write-optimized           Read-optimized
                    ┌───────────┐             ┌───────────┐
                    │    L0     │             │    L1     │
                    │           │  Reconcile  │           │
  Producers ──►     │ Multi-    │ ──────────► │ Single-   │ ──► Historical
                    │ partition │             │ partition │     Consumers
                    │ Small     │             │ Large     │
                    │ batches   │             │ Sorted    │
                    └───────────┘             └───────────┘
                                                   │
                                             Metadata stored
                                             in KV store
                                             (internal topic)
```

---

# Step 3: The Read Path

Reads are routed based on the **Last Reconciled Offset**

```
  Partition offset space:

  ◄──────── L1 (Reconciled) ────────┼──── L0 / Cache ────►
  0                          Last Reconciled            HEAD
                              Offset
```

<br>

- **Tailing consumers** (most workloads): read from **memory cache** — low latency
- **Recent offsets** (> Last Reconciled): follow Raft log pointers → **L0 files**
- **Historical offsets** (< Last Reconciled): read from **L1 files** — large sequential reads

---

# Putting It All Together

```
                          Object Storage
                    ┌──────────────────────────┐
                    │  L0 Files  │  L1 Files   │
                    │ (batched)  │ (optimized) │
                    └──────▲──────────▲────────┘
                           │          │
               Upload ─────┘          │ Reconcile
                    │                 │
  ┌─────────────────┴─────────────────┴───────────────┐
  │                   Redpanda Broker                 │
  │                                                   │
  │  Write Path          Reconciler        Read Path  │
  │  • Batch             • L0 → L1         • Cache    │
  │  • Upload L0         • Compact          • L0 read │
  │  • Raft placeholder  • GC old L0        • L1 read │
  └───────────▲───────────────────────────────┬───────┘
              │                               │
         Producers                       Consumers
```

---

# Multi-Modal Streaming Engine

A single Redpanda cluster supports **both** topic types simultaneously

```
  ┌────────────────────────────────────────────────────┐
  │                  Redpanda Cluster                  │
  │                                                    │
  │  Standard Topics          Cloud Topics             │
  │  ┌────────────────┐      ┌────────────────┐        │
  │  │ Raft-replicated│      │ Object storage │        │
  │  │ Low latency    │      │ Low cost       │        │
  │  │ Trading, RT    │      │ Logs, CDC      │        │
  │  └────────────────┘      └────────────────┘        │
  │                                                    │
  │       Same Kafka API • Same binary • Same cluster  │
  └────────────────────────────────────────────────────┘
```

No separate infrastructure — choose the right mode **per topic**

---

# My Contributions

---

# Contribution: LSM Tree KV Store

**Built the metadata storage engine for Cloud Topics**

<!-- TODO: Fill in details -->

- Problem: needed a scalable, consistent metadata store for L1 file tracking
- Solution: LSM tree-based key-value store
- Integrated with Raft consensus for replication
- Backed by object storage

---

# Contribution: Producer Optimizations

**Optimized the Cloud Topics write path for throughput**

<!-- TODO: Fill in details -->

- Problem: initial write path performance was insufficient
- Solution: targeted producer optimizations
- Result: significant throughput improvements

---

# Contribution: Cluster Epoch System

**Designed the cluster epoch mechanism**

<!-- TODO: Fill in details -->

- Problem: TBD
- Solution: cluster epoch system
- Impact: TBD

---

# Questions?
