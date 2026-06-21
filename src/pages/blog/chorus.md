---
layout: ../../layouts/BlogPost.astro
title: "Chorus: A fast WAL for object storage"
date: "2026-06-22"
readTime: 10
coverImage: "/blog/images/chorus.jpg"
---

Recently, I’ve been fascinated by the trend of building data storage systems on object storage. Systems like [TurboPuffer][tpuf] are all the rage in the search space right now. New storage primitives like [SlateDB][slatedb] make creating bespoke storage systems feasible for a single developer or small team.

A former colleague of mine, Nicolae, actually wrote this really interesting protocol and formal model for a write-ahead log on object storage called [OSWALD][oswald]. It's very cool work, and there are obviously a lot of benefits to object-storage-first systems:

* You get bottomless storage out of the box.

* You can move the compute around really easily, which drastically simplifies Day 2 operations.

* Object storage is very robust and auto-scales incredibly well. It’s just a really good primitive to work with.

However, building on object storage always comes at the cost of write latency.

You can normally hide read latency behind a cache. As an example, TurboPuffer will have a cold start, download the objects to local NVMe, and then do all its searches from there, so its warm latency is incredibly fast. Before a user even hits Enter in the search bar, you can pre-warm the cache and provide instant search while still storing all the data as objects.

But you can’t hide write latency unless you sacrifice durability or move off object storage entirely. Then you lose all those nice benefits of object storage.

## Rapid Storage Buckets

That’s why, earlier this year, I was interested to see that [Google Cloud's Rapid Storage Buckets had become generally available][rapid-ga]. As I understand it, they are basically a very thin veneer over the [Colossus][colossus] storage system at Google.

The really interesting thing is that the interface they expose differs from traditional S3 APIs. The write path is actually based on objects being *appendable*. In my testing, the appends have incredibly low latency. With fsync turned on, I see commit latency of 2 milliseconds at P99.

It is incredibly fast, and I’ve kept wondering: Can we build a robust, object-storage-first storage system without sacrificing write latency by putting the write-ahead log in a Rapid bucket and the rest of the data in a cheap regional bucket? The downside of these fast, appendable objects is that they are scoped to a single zone. If you want higher durability guarantees, you’re going to need to write to a quorum of zones via some kind of replication protocol.

Your use case may not require high availability. Especially for a write-ahead log, if you're archiving every few seconds to regional storage, a couple of seconds of data loss may be acceptable. It will be awkward to swap to another bucket and reconfigure applications to write to another zone, but it's a trade-off that I think you can make a good case for.

However, some use cases absolutely require higher availability and need to tolerate zone failures. In this case, we need a replication protocol, and there are really two ways to go about it. One is to use something like Raft, place each log in the group in a different zone, and replicate across zones. But managing consensus and performing reconfigurations complicates operations. Since storage and compute are decoupled, you need to enforce a single writer for each log anyway. The alternative is an [OSWALD][oswald]-like system in which the client manages replication to each zone.

## Enter Chorus

That’s where this idea, which has been eating at me for a while, comes from. It takes the architectural inspiration of [OSWALD][oswald] and asks: Can we make that work for Rapid Storage Buckets?

Can we get a single-writer write-ahead log that is incredibly fast but replicated across a region? That way, it has higher reliability than a single-zone log.

And that’s where Chorus comes in.

Chorus is a single-writer write-ahead log built on top of Rapid Storage Buckets. For its control plane, it uses a single regional linearizable register to perform Compare-and-Swap (CAS) operations on the log's metadata. This register can be a single object in a regional storage bucket but can easily be swapped out for a database row in Firestore or Spanner. The data plane then consists almost entirely of appends to at least a quorum of zonal objects in different buckets. It comes batteries included with segment rotation, prefix truncation, and orphan cleanup.

Together, this gives you incredibly fast writes, a system built completely on object storage, and zone-fault tolerance. In fact, I benchmarked my implementation against a simple log on [Hyperdisk High-Availability Block Storage][hyperdisk-ha], and the **P99** for Chorus was *faster* than the **P50** for Hyperdisk HA.

In the course of developing this, I wanted to do a few things:

* Use a formal model: Like OSWALD, I wanted to use a formal model to verify that this really works and that there are no weird edge cases around recovery, rotation, or a crash at just the wrong time. Having a single regional register for metadata simplifies a lot of the edge cases. While I know the hardcore people use TLA+, I find [P][p] much more approachable.

* Write it in Rust: Rust is a great systems language, and it’s very easy to build a high-performance Chorus client with it. Also, I might get someone to read this post just because Chorus is written in Rust 😂 But seriously, Rust is great, and my benchmarks show median latency of 1.71ms and P99 latency of 2.74ms for Chorus append operations using Rust, gRPC, and Tokio.

* Deterministic simulation testing: I’ve been really interested in deterministic simulation testing (DST). [Antithesis][antithesis] has done a great job of promoting this mindset, so I have built it into Chorus as well.

So the protocol is formally modeled, we have a high-performance Rust client, and we perform deterministic simulation testing on our production client by injecting faults, stressing the protocol in different ways, and simulating zone outages. The **really** cool thing about P is that we can then use [PObserve][pobserve] to replay the traces from our DST against our formal model which provides strong evidence that our implementation conforms to it.

All of this code is open source on my [GitHub account](http://github.com/rockwotj/chorus).

## Under the Hood: The Protocol

The algorithm itself is fairly straightforward as far as replication protocols go. We have a single metadata register on which we perform Compare-and-Swap operations to linearize all our control plane operations.

Inside our metadata, we store a few things:

* An Epoch: This is where we track the current version, or latest term in Raft-speak. The epoch is bumped every time another process takes over.

* Owner: A random ID of the process that owns the current epoch.

* Segments: A log is broken up into segments so that we can perform prefix truncation and clean up old data. There are three types of segments, and we store all their metadata in the regional register. We don't use LIST APIs (outside of orphan cleanup) and clever naming schemes because those always seem to end with a nasty race condition:

  * The Active Segment: The segment that's currently getting written to.

  * The Sealed Segments: The historical archive of closed segments that are no longer being written to. We also store the CRC and base record index for these segments.

  * The Pending Segment: The preallocated segment that the next rotation event will promote to being the active segment.

We put all of that into our metadata and perform CAS operations to ensure consistency and safe crash recovery at every step.

The reason we have both pending and active objects is that they allow us to change which object we append to without waiting for a Compare-and-Swap operation while keeping recovery safe. We register the pending object, and it is still read during recovery, but it is often empty unless there is a crash in the brief interval when rotating the pending object to active. This lets us start appending to the pending object without waiting for the CAS operation, which takes on the order of 30ms for a regional GCP object, avoiding P99 latency spikes. We can start writing to the pending object and perform the seal and rotation in the background.

Recovery works as you would expect. You read from the sealed segments, the active segment, and potentially the pending segment to understand the state and perform any necessary rotation or initialization. We record the CRCs of the sealed objects in our metadata register and use them during recovery to determine whether something has gone wrong, something has changed, or one segment is different.

If a process crashes or exits while writes are in flight to the active or pending segment, recovery picks a prefix of the log that will include all committed records. This means that an in-flight write that was never acknowledged may become promoted to committed. Since Rapid object cannot be truncated when we find these inconsistencies, we do something much simpler: copy one of the consistent objects to the recovered zone.

## The Big Picture

I recently listened to a [talk][cmu-talk] from the CMU database series in which the CEO of TurboPuffer said that he expects to see many more bespoke data storage systems because the available primitives are now so good. I completely agree, and that idea was one of the things that motivated me to build Chorus.

I hope Chorus can become, alongside SlateDB, one of the major components people use to build robust, bespoke data storage systems on GCP. I have plans to build a Network File System (an NFS server) using SlateDB and Chorus, and I'm excited to see what other people build as well.

Please feel free to reach out if you build with Chorus or are interested in this work. Thanks for reading!

[oswald]: https://nvartolomei.com/
[tpuf]: https://turbopuffer.com/docs/architecture
[rapid-ga]: https://cloud.google.com/blog/products/storage-data-transfer/cloud-storage-rapid-turbocharges-object-storage-for-ai-analytics
[slatedb]: https://slatedb.io/
[colossus]: https://cloud.google.com/blog/products/storage-data-transfer/a-peek-behind-colossus-googles-file-system
[hyperdisk-ha]: https://docs.cloud.google.com/compute/docs/disks/hd-types/hyperdisk-balanced-ha
[antithesis]: https://antithesis.com/
[pobserve]: https://p-org.github.io/P/advanced/pobserve/pobserve/
[p]: https://p-org.github.io/P/whatisP/
[cmu-talk]: https://youtu.be/pqoRNwNaxfs?si=i8nOQpF-ATWFoHrd&t=3466
