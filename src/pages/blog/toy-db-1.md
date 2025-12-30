---
layout: ../../layouts/BlogPost.astro
title: "Let's build a document database!"
date: "2023-01-08"
readTime: 3
---

I'm going to build out a small document database and write out my journey here. This is sort of a kickoff post in a series. This is inspired by Matt Brubeck's excellent [Let's build a browser engine](https://limpet.net/mbrubeck/2014/08/08/toy-layout-engine-1.html) series.

## You're building what?

Let's talk terminology. I originally was going to call this a NoSQL database, but I think that's missing the point, a lot of the concepts I'm going to work through also would apply to a relational database. In my opinion, a toy document database is smaller than a toy relational database because the feature set is smaller. There is also Connor Stack's detailed [DB tutorial](https://cstack.github.io/db_tutorial/) on how to build a [SQLite](https://www.sqlite.org/index.html) clone. Specifically, I'm hoping to build out a [schemaless](https://redis.com/blog/schemaless-databases/) [JSON](https://www.mongodb.com/databases/json-database) database that allows for [transactions](https://en.wikipedia.org/wiki/Database_transaction), [indexes](https://en.wikipedia.org/wiki/Database_index), and a [query planner](https://en.wikipedia.org/wiki/Query_optimization).

## Why a toy document database?

Because I believe hands-on learning is the best kind of learning. You never fully grasp all the devilish little details until you do it yourself. Building a full database is a huge task, especially nowadays as more and more databases are distributed, or support a ton of features. While many databases are open source, due to the complexity of the system, they often aren't practical for a beginner to learn from.

I'm writing this targeting a university-level student. I may miss and overshoot, but I hope to make these topics approachable. As such my toy document database is called SyllabusDB and I encourage you to follow along. I build software in the following steps:

1. Make is simple

2. Make it correct

3. Make it fast


For these blog posts and any associated code, I'm only going to be on step one. Concretely, this means robust (or maybe any) error handling and performance will not be pursued. We'll talk about performance as it's an important aspect of database systems, but mostly at a macro level.

The code for these posts is written in Java - I'll also be staying away from anything outside the standard library for any code relevant to the blog. There are some libraries I chose to use for things like tests, serialization, networking etc. but none of those aspects of databases will be a topic here. I chose Java because it's a common language to learn in university, has multithreading so I can talk about locking (more on this later) and frankly I'll be more productive not thinking about memory or fighting a borrow checker. Feel free to follow along in your language of choice.

## Prerequisites

There are a few concepts I'm going to assume you already know reading these posts. If you don't I recommend doing some reading on the topic before diving in.

### Key-Value Storage (KV Store)

A popular abstraction for a database's storage layer is a [Key-Value Store](https://en.wikipedia.org/wiki/Key%E2%80%93value_database). Conceptionally they are a disk-persisted version of `NavigatableMap<byte[], byte[]>` in Java, `std::map<std::string, std::string>` in C++, or `BTreeMap<Vec<u8>, Vec<u8>>` in Rust. There are a variety of other features they can support, but for this series, we're going to assume that we have an already implemented KV Store with the same feature set as [LevelDB](https://github.com/google/leveldb). Namely, this is:

* Sorted by key

* [Atomic](https://en.wikipedia.org/wiki/Atomicity_(database_systems)) batches of writes

* Iterators over the data provide a [consistent](https://en.wikipedia.org/wiki/Consistency_(database_systems)) snapshot of the data


### Multithreading

When we get to talking about transactions and locking, we'll be diving into the weeds of locks and atomics.
