---
layout: ../../layouts/BlogPost.astro
title: "Multi Tenancy in Software Systems"
date: "2026-05-25"
readTime: 4
coverImage: "/blog/images/multiple-tenants.jpg"
---

Over my career so far, I've worked on three storage-level systems. Two databases and a message broker:

- [Firebase Realtime Database][rtdb]
- [Google Cloud Firestore][firestore]
- [Redpanda Streaming Engine][rp]

They have all been a lot of fun to work on, and all have had their challenges and problems.
Recently, I've been stewing on one of the largest architectural differences between them: multitenancy[^why].

As defined by [wikipedia][wiki], multitenancy is: "a software architecture in which a single instance of software runs on a server and services multiple tenants". The important thing I want to call out here is that multitenancy is fundamentally _architectural_.

The Firebase Realtime Database and Google Cloud Firestore are both multitenant systems. The Redpanda Streaming engine is single tenant. I'm not writing this to say that Redpanda is flawed for not being multi-tenant - Redpanda **does** have a multitenant offering, but the multitenancy is provided by a proxy layer on top of the core engine. This has a number of interesting tradeoffs: the core engine stays simple, and you could decouple the team and deliverables. However, fundamentally there are features that cannot be reasonably made multitenant (i.e. the number of pending transactions in flight, or the number of idempotent producers for a tenant). Ultimately,  workarounds were built into Redpanda to service use cases like this (along with functionality such as moving tenants between clusters). I have to think back from my previous experience that ultimately the multitenancy was not really ever an issue with those systems, because it was architected from the beginning to be multitenant. This brings me back to the benefit of building multitenancy into storage systems from the start - and I'm not talking about row level security with Postgres, but true deep namespacing where a transaction from one customer can't affect another (ie. table or database level locks can't be taken). Noisy neighbor issues are fundamentally always present in these systems, but these multitenant architectures are much better suited to mitigate them.

I suppose what I am getting at is that I encourage new systems being architected to **deeply** consider architecting for multitenancy.

## Multitenancy is leveled

While storage systems themselves can be multitenant for the service provider to their customers, it's often the case that the customers themselves *also* need multitenancy. There are a couple cases here worthwhile to call out so let me break it down a bit:

### Internal SaaS

Large companies often have an internal "DB-as-a-service", or "Queue-as-a-service", etc team. In this case the company wants to have each of their internal product teams be their "tenants" and they want to slot them all into a single pane of glass to understand quotas, usage, breaking down cost, etc. Often times for a cloud hosted service, just letting each team use a "tenant" directly from the service provider can work, as long as the tooling allows for the cross cutting views and concerns for the internal team to understand usage as a whole across the company. This often looks like a single pane of glass that aggregates things across the internal team usage to provide a view across the company. This is the easier mode.

### Nested Multitenancy

Often times, a company wants to provide each of _their customers_ what appears to be fully isolated data and environments. But they can't go making a new database instance in their Cloud for each new person or agent that goes through the signup flow! That would be a mess, and often there are scaling limits in terms of number tenants that people can run into trying to do this. But providing this capability is super powerful and can remove a huge swath of security issues, and reliablity issues by the customer not having to re-implement multitenancy on top of an existing multitenant system.

## On Testing

Perhaps one thing people don't think about when designing a system for multitenancy is that it can greatly aid the developer experience during testing of your software. How you may ask? It can make running your integration test much faster to execute! Often you want to write an end-to-end or integration test that does a few operations and asserts some output. For larger systems, while the actual operations generally take less than a second,  the test may take multiple seconds or even close dozens of seconds to run. Driving down this iteration time can be a huge boon to productivity. The reason that these tests take so long is that they often resort to using isolation primatives like spinning up a bunch of docker containers per test so each test is isolated. Aha! But if my system itself is multitenant, then I can run each test as its own tenant and we only pay the setup/teardown cost once for a bunch of tests, instead of one per test[^dedicated].

You may wonder why this note is here and I'll leave that as a teaser for a future blogpost, but let's just say I recently discovered this first hand[^why] 😁

## The Holy Grail of Multitenancy

I am convinced that nested multitenancy enables users to do the simplest thing possible and just give each of the end "customer" their own database. This is one of the big selling points of systems like [Turbopuffer][tpuf], [Turso][turso] or [Cloudflare's D1][d1]. If you're in the process of designing a new system, I highly recommend looking at architecting for multitenancy from day one - I've not worked on a system that provides both layers of isolation here, but as someone working on building a product instead of infrastructure now, I can feel the appeal.

[^why]: As for *why* I've been thinking about multitenancy, that's the subject for my next blog post 😉
[^dedicated]: Of course some tests themselves might need different configuration or they want to test some interaction of tenants (or that there should not be any!) and in this case providing an opt-in mechanism for a dedicated server to run tests in can be very powerful. This is about the developer ergonmics of biasing towards multitenant, cheaper tests as opposed to heavier isolation mechanisms by default.

[wiki]: https://en.wikipedia.org/wiki/Multitenancy
[rtdb]: https://firebase.google.com/products/realtime-database
[firestore]: https://cloud.google.com/products/firestore
[rp]: https://www.redpanda.com/data-streaming
[turso]: https://turso.tech/
[d1]: https://developers.cloudflare.com/d1/
[tpuf]: https://turbopuffer.com/docs/concepts#multi-tenancy
