---
layout: ../../layouts/BlogPost.astro
title: "Google Cloud Firestore Transactions"
date: "2026-08-05"
readTime: 5
coverImage: "/blog/images/firestore-txn.jpg"
---

I've been seeing a lot of misinformation from LLMs about the transaction model that [Google Cloud Firestore][firestore] exposes. I'm not sure if that's because it's tied to the Firebase Realtime Database or Google Cloud Datastore, which historically have had limited transactions in one way or another. If you read the [Firestore whitepaper][whitepaper] you can see that it's built upon [Spanner][spanner]. This means that Firestore transactions under the hood are essentially just Spanner transactions and they inherit [full serializability][serializability] and [external consistency][external-consistency] from Spanner. Serializability is a property of transactions which means that the database executes the transactions in a manner *as if* they were done sequentially. However, in reality it would be super slow to do this in practice so transactions that don't touch the same rows/documents end up executing concurrently.

Database isolation levels can feel like a very academic subject for a full stack developer, but if you aren't using serializable transactions and your app is any more complex than a trivial CRUD app, you probably have some bugs unless you're always *very* careful.

## A motivating example

Let's draw out the power of serializable transactions with an example derived from a question that spawned this blog. Let's say a user in our application can have at most 100 projects. The naive version of this in most databases looks like:

1. `SELECT COUNT(*)` their projects
2. If under the limit, `INSERT` a new one

Under weaker isolation levels (which most databases default to), two concurrent requests can both read a count of 99, both insert, and now the user has 101 projects. Neither transaction touched the same *row*, so nothing conflicts. This anomaly is called [write skew][skew], and it can be a super gnarly bug to track down.

In Firestore and other databases that support serializable isolation levels, you can just write the obvious code:

```typescript
await db.runTransaction(async (tx) => {
  const projects = await tx.get(
    db.collection("projects").where("owner", "==", uid)
  );
  if (projects.size >= 100) {
    throw new Error("too many projects");
  }
  tx.create(db.collection("projects").doc(), { owner: uid, name });
});
```

No advisory locks, no denormalized counter document that you have to keep in sync - the obvious code is also the correct code.

## Watch it happen

We can actually write a small example that interleaves two transactions in a single process to trigger the conflict. We inject a synthetic barrier so that the first transaction stalls after its query, and the second one reads the same range *before* the first commits. Here's a complete script you can run against the [Firestore emulator][emulator]:

```typescript
import { Firestore, Transaction, Query } from "@google-cloud/firestore";

const db = new Firestore({ projectId: "demo" });
const LIMIT = 3; // small limit so the script is quick
const projects = (): Query =>
  db.collection("projects").where("owner", "==", "tyler");

async function addProject(
  name: string,
  holdUntil?: Promise<void>,
  onRead?: () => void
) {
  let attempts = 0;
  await db.runTransaction(async (tx: Transaction) => {
    attempts++;
    const snapshot = await tx.get(projects());
    console.log(`[${name}] attempt ${attempts}: sees ${snapshot.size} projects`);
    onRead?.();
    await holdUntil; // hold the transaction open after reading
    if (snapshot.size >= LIMIT) throw new Error(`[${name}] limit reached`);
    tx.create(db.collection("projects").doc(), { owner: "tyler", name });
  });
  console.log(`[${name}] committed after ${attempts} attempt(s)`);
}

async function main() {
  // Seed LIMIT - 1 projects, so exactly one slot remains.
  for (let i = 0; i < LIMIT - 1; i++) {
    await db.collection("projects").add({ owner: "tyler", name: `seed-${i}` });
  }

  // The barrier: tx1 reads first, then holds its transaction open
  // until tx2 has read the same range. Both see one slot left.
  let tx1HasRead!: () => void;
  let tx2HasRead!: () => void;
  const tx1Read = new Promise<void>((r) => (tx1HasRead = r));
  const tx2Read = new Promise<void>((r) => (tx2HasRead = r));

  const tx1 = addProject("tx1", tx2Read, tx1HasRead);
  await tx1Read;
  await addProject("tx2", undefined, tx2HasRead);
  await tx1;
}

main().catch((e) => console.error(`${e.message}`));
```

Here's the output:

```text
[tx1] attempt 1: sees 2 projects
[tx2] attempt 1: sees 2 projects
[tx1] attempt 2: sees 2 projects
[tx1] committed after 2 attempt(s)
[tx2] attempt 2: sees 3 projects
[tx2] limit reached
```

Both transactions read a count of 2 - one under the limit. If the database let both commit, we'd end up with 4 projects. Instead, each transaction's commit conflicts with the *range* the other one scanned, so the database picks a loser and the SDK automatically re-runs its callback once the winner is out of the way. On its retry the loser sees 3 projects and throws. (Which transaction wins can vary between runs - the invariant holds either way.) This is also why the docs tell you to keep side effects out of the transaction callback - it can be re-executed when there's a conflict.

## But how?

What kind of magic enables the database to detect these conflicts? Range locks (also known as gap locks)!

Every database query is a scan over an index or table. `where("owner", "==", uid)` is a scan of a contiguous range of the `owner` index: all the entries with key prefix `uid`. When a transaction runs that query, the underlying storage takes a lock on that key range, not just on the entries that happened to exist at the time.

Now when another transaction inserts a new project for the same user, that insert has to write a *new entry into the locked range* of the index. That write conflicts with the range lock, so one of the two transactions loses and gets retried. This is the classic solution to the [phantom read][phantom] problem: you can't lock a document that doesn't exist yet, but you *can* lock the place in the index where it would appear.

<figure>
<svg width="100%" viewBox="0 0 680 330" xmlns="http://www.w3.org/2000/svg" style="font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif">
<defs>
  <marker id="arrow-ok" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M2 1L8 5L2 9" fill="none" stroke="#0F6E56" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  </marker>
  <marker id="arrow-no" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M2 1L8 5L2 9" fill="none" stroke="#B3261E" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  </marker>
</defs>
<!-- Index label -->
<text x="40" y="88" font-size="13" fill="#5F5E5A">the <tspan font-family="monospace">owner</tspan> index (sorted by key)</text>
<!-- Range lock band: covers the whole owner == "tyler" range, including empty space -->
<rect x="158" y="100" width="330" height="80" rx="10" fill="#EEEDFE" stroke="#534AB7" stroke-width="1" stroke-dasharray="6 4"/>
<text x="323" y="44" text-anchor="middle" font-size="13" font-weight="500" fill="#3C3489">tx1 holds a lock on the range it scanned:</text>
<text x="323" y="62" text-anchor="middle" font-size="13" font-family="monospace" fill="#534AB7">owner == "tyler"</text>
<line x1="323" y1="70" x2="323" y2="94" stroke="#534AB7" stroke-width="1" stroke-dasharray="4 3"/>
<!-- Index entries, sorted -->
<!-- neighbor before -->
<g>
  <rect x="40" y="118" width="106" height="44" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
  <text x="93" y="140" text-anchor="middle" dominant-baseline="central" font-size="12" font-family="monospace" fill="#444441">(kai, p7)</text>
</g>
<!-- tyler entries inside the locked range -->
<g>
  <rect x="170" y="118" width="130" height="44" rx="8" fill="#FDFCFA" stroke="#534AB7" stroke-width="0.5"/>
  <text x="235" y="140" text-anchor="middle" dominant-baseline="central" font-size="12" font-family="monospace" fill="#3C3489">(tyler, seed-0)</text>
</g>
<g>
  <rect x="312" y="118" width="130" height="44" rx="8" fill="#FDFCFA" stroke="#534AB7" stroke-width="0.5"/>
  <text x="377" y="140" text-anchor="middle" dominant-baseline="central" font-size="12" font-family="monospace" fill="#3C3489">(tyler, seed-1)</text>
</g>
<!-- empty slot where a new entry would land: still inside the lock -->
<rect x="454" y="118" width="22" height="44" rx="6" fill="none" stroke="#7F77DD" stroke-width="1" stroke-dasharray="3 3"/>
<!-- neighbor after -->
<g>
  <rect x="500" y="118" width="106" height="44" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
  <text x="553" y="140" text-anchor="middle" dominant-baseline="central" font-size="12" font-family="monospace" fill="#444441">(zoe, p2)</text>
</g>
<!-- tx1 insert: allowed, it owns the lock -->
<path d="M340 268 L430 268 L458 172" fill="none" stroke="#0F6E56" stroke-width="1.5" marker-end="url(#arrow-ok)"/>
<rect x="150" y="248" width="190" height="40" rx="8" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.5"/>
<text x="245" y="262" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="500" fill="#085041">tx1: insert (tyler, p-new)</text>
<text x="245" y="278" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#0F6E56">owns the lock — commits ✓</text>
<!-- tx2 insert: conflicts with the range lock -->
<path d="M560 248 L500 190" fill="none" stroke="#B3261E" stroke-width="1.5" marker-end="url(#arrow-no)"/>
<circle cx="530" cy="219" r="9" fill="#FBEAE9" stroke="#B3261E" stroke-width="1"/>
<text x="530" y="219" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="600" fill="#B3261E">✕</text>
<rect x="470" y="248" width="190" height="40" rx="8" fill="#FBEAE9" stroke="#B3261E" stroke-width="0.5"/>
<text x="565" y="262" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="500" fill="#8C1D18">tx2: insert (tyler, p-new)</text>
<text x="565" y="278" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#B3261E">conflicts with lock — retry ↻</text>
<!-- note: neighbors untouched -->
<text x="93" y="186" text-anchor="middle" font-size="11" fill="#73726C">outside the range:</text>
<text x="93" y="200" text-anchor="middle" font-size="11" fill="#73726C">unaffected</text>
</svg>
<figcaption><small>Figure 1: tx1's query scanned the <code>owner == "tyler"</code> range of the index, so it holds a lock on the whole range - including index entries that don't exist yet. tx2's insert lands inside that range and conflicts.</small></figcaption>
</figure>

A nice consequence of this is that it works for any query shape - as well as [aggregation queries][count]. If you swap the query in the transaction for `tx.get(projects().count())` you don't transfer any documents over the wire, but you keep the exact same guarantee - the count still executes as a scan over the same index range, and the scan is what takes the lock.

There is a trade off, however: it is more expensive in terms of the memory used to track these ranges, and also the compute to check all the overlapping ranges at commit time. This is why most databases don't default to serializable transactions, or impose other limits like Firestore's transaction timeouts. So in some aspects you're trading off performance for safety and ergonomics.

## Wrapping up

It's actually incredibly freeing as a developer to be able to work in purely serializable transactions. In terms of consistency, you only have to worry about things outside of transactions. Within a transaction, you can always feel confident that your database will protect the invariants of your logic based on the reads and writes you perform. So next time you choose a database, I strongly encourage you to research the isolation it provides. As TigerBeetle says: ["give me strict serializability or give me death"][tb]!

[firestore]: https://cloud.google.com/products/firestore
[tb]: https://tigerbeetle.com/blog/2026-03-19-a-trillion-transactions/#give-me-strict-serializability-or-give-me-death
[emulator]: https://firebase.google.com/docs/emulator-suite
[count]: https://firebase.google.com/docs/firestore/query-data/aggregation-queries
[skew]: https://en.wikipedia.org/wiki/Snapshot_isolation
[phantom]: https://en.wikipedia.org/wiki/Isolation_(database_systems)#Phantom_reads
[whitepaper]: https://research.google/pubs/firestore-the-nosql-serverless-database-for-the-application-developer/
[spanner]: https://en.wikipedia.org/wiki/Spanner_(database)
[external-consistency]: https://docs.cloud.google.com/spanner/docs/true-time-external-consistency
[serializability]: https://docs.cloud.google.com/spanner/docs/isolation-levels#serializable
