---
layout: ../../layouts/BlogPost.astro
title: "How Firestore Matches Realtime Queries"
date: "2026-09-05T00:00:00-05:00"
readTime: 7
coverImage: "/blog/images/firestore-query-matching.jpg"
---

# The Story Behind Firestore's Query Matcher

A paper just came out that I'm really excited about: [**The Live Database: Firestore's Scalable and Consistent Realtime Queries**](https://www.vldb.org/pvldb/vol19/p4130-jacobsson.pdf) at VLDB 2026. 

The Firestore team did a great job presenting what is known internally as the **Firestore Watch** system. It was one of the most interesting parts of a system that I got to work on in all my time at Google. I have fond memories from working on that system.

The part I worked on the most is described in **Section 4.3: QueryMatcher**, which is the query matching engine. That was probably my biggest contribution to Firestore Watch, and I worked on it alongside [Jonny Dimond](https://x.com/jonnydimond).

Here’s the backstory of how that system came to be.

---

## Initial Architecture

To set the stage: when Firestore launched, it wasn't using this query matching engine. Instead, it was reusing an existing Google query matching engine that had a similar goal, but was built for Google Search to do full-text search query matching.

It's a very good library for what it does, but it wasn't necessarily a great fit for Firestore.

What we were doing initially was taking document changes coming from Firestore and converting them. Because the search library spoke search documents, we did the infamous Google thing: translating our protos into their protos so we could reuse their library. 

At launch, that was a big win. We got query matching up and running quickly. But in practice, a few serious problems came up:

1. **It became a scaling bottleneck.** Proto conversion accounted for a huge amount of our CPU time. You could have massive, deeply nested documents, but an active query might only care about a single numeric field. You don't actually need to process the entire document to do query matching, but we had to because of how the library interfaces were structured.
2. **Post-filtering overhead.** We had to do a bunch of post-filtering afterwards to accurately overlay Firestore's indexing and query semantics. The search library was great, but it simply wasn't designed for Firestore.

---

## The Intuition

Firestore's core premise at launch was that queries always scale with the size of the result set. Every query is backed by an index, meaning all the queries that Watch supports map very clearly to a single index scan (or more for disjunctive queries).

So the intuition was: 

In a traditional query engine, you have an index range and you scan it to find all the matching points.

What if we flipped that on its head? 

What if we had a bunch of **index ranges** (the queries registered by clients), a **point** comes in (a document change), and we need to efficiently find all the index ranges that match?

It turns out there is already has a great data structure for this: the **[interval tree](https://en.wikipedia.org/wiki/Interval_tree)**. (You should go study it - it's easily one of my favorite data structures).

With an interval tree, you sort the tree by the lower bound of the interval. Then, at every node, you augment it with the maximum value in that subtree. Because of that subtree maximum, you can immediately skip over entire subtrees where your value lies outside the maximum, allowing you to efficiently find all matching points in logarithmic time and easily prune the search space.

---

## Building It

So like many bets, I went ahead and implemented an initial version to verify the gains.

First I wrote an [AVL](https://en.wikipedia.org/wiki/AVL_tree)-tree-based interval tree, written in C++ as a template. I did the thing that everyone tells you you'll never do in your career: write a textbook data structure from scratch like you do in university!

The idea was:
* Take the interval bounds and make them **index entries** which are tuples of indexed values (all using Firestore's value space, index types and semantics).
* Supply a comparator based on the index definition to handle ordering directions (ascending, descending, etc.) so you match real index entries and the scans that Firestore's query executor would perform.

With this in place, for a given document change, you can very efficiently grab just the index entries you need rather than processing the entire protobuf. You only touch the bits where there are actually queries listening.

---

## The core matching algorithm

To keep track of the different indexed values and match incoming changes, we built a **trie structure** of the fields in the index definitions for all active queries. Firestore's semantics are such that all watchable queries have a perfect index that can be deterministically computed based on only the query itself.

The matching process is performed on both the new and previous document by performing a set intersection between the document's field map and the keys in this trie. The intersection was computed by walking both trees in conjunction and descending when there are matches. Every time you get a hit in the trie, there is an interval tree for that field path that you search to find if there are affected queries for that document change.

Now you have this really cool engine: you throw document changes at it, lazily compute only the index entries that have active queries, and throw them into the interval tree to quickly grab all the affected queries. 

From there, the rest of the processing pipeline takes over. Emit the notifications and compute whether the document is a create, update, or delete in the query result set.

---

<iframe
  src="/blog/visuals/query-matching.html"
  title="Interactive diagram: matching a Firestore document to listening queries"
  loading="lazy"
  width="100%"
  height="1700"
  style="display: block; width: calc(100% + 2rem); max-width: none; margin-inline: -1rem; border: 0; font: inherit; color-scheme: inherit;"
></iframe>

<noscript>
  <p>This interactive figure needs JavaScript. <a href="/blog/visuals/query-matching.html">Open the figure and its static explanation.</a></p>
</noscript>

---

## Servicing Disjunctive Queries

This detail is left out in the paper, but `IN` and `OR` queries are supported naturally with a minor extension.
 
For `IN` and `OR` queries we compute the **[Disjunctive Normal Form (DNF)](https://en.wikipedia.org/wiki/Disjunctive_normal_form)** of the query. That breaks the query down into its individual disjunctive clauses, where each branch corresponds cleanly to an index scan.

We then take each individual disjunction and insert it into the appropriate interval tree for its corresponding index definition. When an incoming document mutation arrives, if it touches *any* of those registered intervals across the trees, we know the query has been matched, and we update the query accordingly.

---

## Perfy Gold

We built this primitive, and rolled it out in what we called "shadow mode", where we performed both query matching pipelines and ensured that the results between the new and old were consistent. At the same time this validated the performance. This technique was inspired at the time by [Github's scientist library](https://github.blog/developer-skills/application-development/scientist/).

It ended up rolling out just in time, we were just starting to see a big ramp-up in scale and higher adoption rates of the realtime queries.

It very rapidly reduced latency and increased throughput across the read side of the Watch system. My recolection is that we saw greater than a 10x improvement for latency and throughput. I don't remember the exact numbers at this point, but I remember looking at the perfy submission charts and thinking they looked too wild to be real.

That work earned me a **Gold Perfy** award. Google used to give out these awards periodically for impressive performance engineering work, so it was a huge honor to have this project recognized.

---

<figure>
    <img src="/blog/images/perfy.jpg" alt="Perfy Shirt">
    <figcaption>Proof I'm not making this stuff up</figcaption>
</figure>

---

## In summary

It's very cool to now see this work published in VLDB for others to learn from and for me to have nostalgia.

If you have any interest in this sort of topic or want to chat more about query matching and database internals, I'm always happy to talk over email. 
