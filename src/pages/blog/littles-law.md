---
layout: ../../layouts/BlogPost.astro
title: "Little's Law in Practice"
date: "2026-04-03"
readTime: 5
coverImage: "/blog/images/littles-law.jpg"
---

[Little's law][wiki] is a theorem that tells us the relationship in queuing systems between the throughput of the system versus the number of requests and how long it takes to process each request. When developing [Cloud (as in direct to object storage) Topics][ct-blog] in Redpanda, we came across this very practically: each request in Redpanda normally hits NVMe storage and is processed in microseconds but for the first time with object storage in the produce path each request now takes hundreds of milliseconds. Understandably, this caused us to find bottlenecks now that the latency profile has drastically changed. As we were first testing scaling up from MB/s scale to GB/s we found that some pipelined processing code we had introduced a few years ago was preventing us from getting the throughput we needed. For low latency topics, the produce request processing looks something like this:

<figure>
<svg width="100%" viewBox="0 20 680 280" xmlns="http://www.w3.org/2000/svg">
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  </marker>
<mask id="imagine-text-gaps-ycqhyf" maskUnits="userSpaceOnUse"><rect x="0" y="0" width="680" height="320" fill="white"/><rect x="59.32111740112305" y="69.02759552001953" width="71.3577651977539" height="21.944807052612305" fill="black" rx="2"/><rect x="162.8422393798828" y="60.20817565917969" width="42.31550598144531" height="19.583648681640625" fill="black" rx="2"/><rect x="271.9397888183594" y="57.02759552001953" width="56.120418548583984" height="21.944807052612305" fill="black" rx="2"/><rect x="250.2879638671875" y="78.20817565917969" width="99.424072265625" height="19.583648681640625" fill="black" rx="2"/><rect x="245.82537841796875" y="94.20817565917969" width="108.34925079345703" height="19.583648681640625" fill="black" rx="2"/><rect x="510.53094482421875" y="57.02759552001953" width="58.93806838989258" height="21.944807052612305" fill="black" rx="2"/><rect x="500.0867614746094" y="78.20817565917969" width="79.82645416259766" height="19.583648681640625" fill="black" rx="2"/><rect x="495.5218811035156" y="94.20817565917969" width="88.95626831054688" height="19.583648681640625" fill="black" rx="2"/><rect x="52.42161560058594" y="198.2081756591797" width="205.15676879882812" height="19.583648681640625" fill="black" rx="2"/><rect x="241.22653198242188" y="248.2081756591797" width="177.54693603515625" height="19.583648681640625" fill="black" rx="2"/><rect x="266.3278503417969" y="154.2081756591797" width="137.3442840576172" height="19.583648681640625" fill="black" rx="2"/><rect x="237.74209594726562" y="170.2081756591797" width="194.51580810546875" height="19.583648681640625" fill="black" rx="2"/></mask></defs>

<!-- Row 1 (y=80): forward flow — producer → stage 1 → stage 2 -->
<!-- Producer -->
<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="40" y="58" width="110" height="44" rx="8" stroke-width="0.5" style="fill:rgb(241, 239, 232);stroke:rgb(95, 94, 90);color:rgb(0, 0, 0);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="95" y="80" text-anchor="middle" dominant-baseline="central" style="fill:rgb(68, 68, 65);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Producer</text>
</g>

<!-- batch → stage 1 -->
<line x1="150" y1="80" x2="218" y2="80" marker-end="url(#arrow)" style="fill:none;stroke:rgb(115, 114, 108);color:rgb(0, 0, 0);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="184" y="70" text-anchor="middle" dominant-baseline="central" style="fill:rgb(61, 61, 58);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">batch</text>

<!-- Stage 1 -->
<g onclick="sendPrompt('What is the Raft replication buffer enqueue?')" style="fill:rgb(0, 0, 0);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="220" y="44" width="160" height="72" rx="8" stroke-width="0.5" style="fill:rgb(238, 237, 254);stroke:rgb(83, 74, 183);color:rgb(0, 0, 0);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="300" y="68" text-anchor="middle" dominant-baseline="central" style="fill:rgb(60, 52, 137);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Stage 1</text>
  <text x="300" y="88" text-anchor="middle" dominant-baseline="central" style="fill:rgb(83, 74, 183);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">Enqueue in Raft</text>
  <text x="300" y="104" text-anchor="middle" dominant-baseline="central" style="fill:rgb(83, 74, 183);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">replication buffer</text>
</g>

<!-- stage 1 → stage 2 -->
<line x1="380" y1="80" x2="448" y2="80" marker-end="url(#arrow)" style="fill:none;stroke:rgb(115, 114, 108);color:rgb(0, 0, 0);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

<!-- Stage 2 -->
<g onclick="sendPrompt('How does Raft majority acknowledgement work?')" style="fill:rgb(0, 0, 0);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="450" y="44" width="180" height="72" rx="8" stroke-width="0.5" style="fill:rgb(238, 237, 254);stroke:rgb(83, 74, 183);color:rgb(0, 0, 0);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="540" y="68" text-anchor="middle" dominant-baseline="central" style="fill:rgb(60, 52, 137);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Stage 2</text>
  <text x="540" y="88" text-anchor="middle" dominant-baseline="central" style="fill:rgb(83, 74, 183);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">Majority ack</text>
  <text x="540" y="104" text-anchor="middle" dominant-baseline="central" style="fill:rgb(83, 74, 183);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">+ flush to disk</text>
</g>

<!-- Row 2 (y=200): ack return paths, well below the forward flow -->

<!-- Ack 1: stage 1 → producer, travels below -->
<path d="M220 116 L220 190 L95 190 L95 104" fill="none" stroke="#7F77DD" stroke-width="1" stroke-dasharray="4 3" marker-end="url(#arrow)" style="fill:none;stroke:rgb(127, 119, 221);color:rgb(0, 0, 0);stroke-width:1px;stroke-dasharray:4px, 3px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="155" y="208" text-anchor="middle" dominant-baseline="central" style="fill:rgb(61, 61, 58);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">ack 1 — producer sends next batch</text>

<!-- Ack 2: stage 2 → producer, travels further below -->
<path d="M450 116 L450 240 L95 240 L95 106" fill="none" stroke="#534AB7" stroke-width="1" stroke-dasharray="4 3" marker-end="url(#arrow)" mask="url(#imagine-text-gaps-ycqhyf)" style="fill:none;stroke:rgb(83, 74, 183);color:rgb(0, 0, 0);stroke-width:1px;stroke-dasharray:4px, 3px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="330" y="258" text-anchor="middle" dominant-baseline="central" style="fill:rgb(61, 61, 58);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">ack 2 — write fully committed</text>

<!-- Ordering constraint label -->
<rect x="220" y="152" width="230" height="36" rx="6" stroke-width="0.5" fill="none" stroke="var(--color-border-secondary)" stroke-dasharray="4 3" style="fill:none;stroke:rgba(31, 30, 29, 0.3);color:rgb(0, 0, 0);stroke-width:0.5px;stroke-dasharray:4px, 3px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="335" y="164" text-anchor="middle" dominant-baseline="central" style="fill:rgb(61, 61, 58);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">ordering preserved for</text>
<text x="335" y="180" text-anchor="middle" dominant-baseline="central" style="fill:rgb(61, 61, 58);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">idempotency sequence numbers</text>

</svg>

<figcaption><small>Figure 1: Produce processing for low latency topics</small></figcaption>
</figure>

We process each produce request sequentially per producer/network connection and we take the producer's batch and (after doing various validations and checks for idempotency, etc) we enqueue the batch in a buffer in the raft leader to be sent to all replicas. At that point, the order the batch will be inserted into the log is guaranteed and we can ask the request processing layer to release the next batch into the system. This preserves order of requests and is required for [idempotent producers][kip] in Kafka. However, we do not acknowledge the request to the producer until we properly replicate the batch to a majority of the Raft group. This allows for nice pipelined produce request processing.

You might ask: where is the bottleneck then? With Cloud Topics, we write the data from the producer to object storage, then we replace the batch with metadata and replicate it via raft like a "normal" batch from a low latency topic. However, we've now multiplied our latency from producer to stage one by over 100 times by introducing batching and a write to object storage! Little's law equates to `Throughput = Latency * Concurrency` - in order to improve our throughput with the increased latency we need to add more concurrency. We ended up fixing this by… introducing another layer of queuing.

<figure>
<svg width="101%" viewBox="0 20 680 360" xmlns="http://www.w3.org/2000/svg">
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  </marker>
</defs>

<!-- Forward flow -->

<!-- Producer -->
<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="20" y="138" width="100" height="44" rx="8" stroke-width="0.5" style="fill:rgb(241, 239, 232);stroke:rgb(95, 94, 90);color:rgb(0, 0, 0);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="70" y="160" text-anchor="middle" dominant-baseline="central" style="fill:rgb(68, 68, 65);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Producer</text>
</g>

<!-- producer → queue -->
<line x1="120" y1="160" x2="146" y2="160" marker-end="url(#arrow)" style="fill:none;stroke:rgb(115, 114, 108);color:rgb(0, 0, 0);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

<!-- Queue -->
<g onclick="sendPrompt('What does the queue layer do?')" style="fill:rgb(0, 0, 0);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="148" y="124" width="130" height="72" rx="8" stroke-width="1" style="fill:rgb(225, 245, 238);stroke:rgb(15, 110, 86);color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="213" y="148" text-anchor="middle" dominant-baseline="central" style="fill:rgb(8, 80, 65);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Queue</text>
  <text x="213" y="168" text-anchor="middle" dominant-baseline="central" style="fill:rgb(15, 110, 86);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">Accepts batches,</text>
  <text x="213" y="184" text-anchor="middle" dominant-baseline="central" style="fill:rgb(15, 110, 86);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">sends ack 1</text>
</g>

<!-- queue → upload -->
<line x1="278" y1="160" x2="304" y2="160" marker-end="url(#arrow)" style="fill:none;stroke:rgb(115, 114, 108);color:rgb(0, 0, 0);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

<!-- Upload -->
<g onclick="sendPrompt('What happens during upload?')" style="fill:rgb(0, 0, 0);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="306" y="138" width="96" height="44" rx="8" stroke-width="0.5" style="fill:rgb(225, 245, 238);stroke:rgb(15, 110, 86);color:rgb(0, 0, 0);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="354" y="160" text-anchor="middle" dominant-baseline="central" style="fill:rgb(8, 80, 65);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Upload</text>
</g>

<!-- upload → raft enqueue -->
<line x1="402" y1="160" x2="428" y2="160" marker-end="url(#arrow)" style="fill:none;stroke:rgb(115, 114, 108);color:rgb(0, 0, 0);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

<!-- Raft enqueue (stage 1) -->
<g onclick="sendPrompt('What is the Raft pre-replicate enqueue step?')" style="fill:rgb(0, 0, 0);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="430" y="124" width="130" height="72" rx="8" stroke-width="0.5" style="fill:rgb(238, 237, 254);stroke:rgb(83, 74, 183);color:rgb(0, 0, 0);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="495" y="145" text-anchor="middle" dominant-baseline="central" style="fill:rgb(60, 52, 137);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Raft enqueue</text>
  <text x="495" y="163" text-anchor="middle" dominant-baseline="central" style="fill:rgb(83, 74, 183);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">Stage 1: pre-replicate</text>
  <text x="495" y="179" text-anchor="middle" dominant-baseline="central" style="fill:rgb(83, 74, 183);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">enqueue + ack</text>
</g>

<!-- raft enqueue → raft replicate -->
<line x1="560" y1="160" x2="586" y2="160" marker-end="url(#arrow)" style="fill:none;stroke:rgb(115, 114, 108);color:rgb(0, 0, 0);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

<!-- Raft replicate (stage 2) -->
<g onclick="sendPrompt('How does Raft majority replication work?')" style="fill:rgb(0, 0, 0);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
  <rect x="588" y="124" width="72" height="72" rx="8" stroke-width="0.5" style="fill:rgb(238, 237, 254);stroke:rgb(83, 74, 183);color:rgb(0, 0, 0);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
  <text x="624" y="145" text-anchor="middle" dominant-baseline="central" style="fill:rgb(60, 52, 137);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Raft</text>
  <text x="624" y="163" text-anchor="middle" dominant-baseline="central" style="fill:rgb(83, 74, 183);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">Stage 2:</text>
  <text x="624" y="179" text-anchor="middle" dominant-baseline="central" style="fill:rgb(83, 74, 183);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">replicate</text>
</g>

<!-- ACK 1: Queue → Producer — routed ABOVE the forward flow -->
<path d="M148 124 L148 60 L70 60 L70 138" fill="none" stroke="#1D9E75" stroke-width="1" stroke-dasharray="4 3" marker-end="url(#arrow)" style="fill:none;stroke:rgb(29, 158, 117);color:rgb(0, 0, 0);stroke-width:1px;stroke-dasharray:4px, 3px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="165" y="48" text-anchor="start" dominant-baseline="central" style="fill:rgb(61, 61, 58);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:start;dominant-baseline:central">ack 1 — producer sends next batch</text>

<!-- Stage 1 ack → queue: releases next metadata into raft — below -->
<path d="M430 196 L430 256 L213 256 L213 198" fill="none" stroke="#7F77DD" stroke-width="1" stroke-dasharray="4 3" marker-end="url(#arrow)" style="fill:none;stroke:rgb(127, 119, 221);color:rgb(0, 0, 0);stroke-width:1px;stroke-dasharray:4px, 3px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="320" y="274" text-anchor="middle" dominant-baseline="central" style="fill:rgb(61, 61, 58);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">stage 1 ack — releases next metadata in order</text>

<!-- Ack 2: Raft replicate → Producer — deepest below -->
<path d="M588 198 L588 318 L70 318 L70 184" fill="none" stroke="#534AB7" stroke-width="1" stroke-dasharray="4 3" marker-end="url(#arrow)" style="fill:none;stroke:rgb(83, 74, 183);color:rgb(0, 0, 0);stroke-width:1px;stroke-dasharray:4px, 3px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="340" y="336" text-anchor="middle" dominant-baseline="central" style="fill:rgb(61, 61, 58);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">ack 2 — metadata replicated on majority, data committed</text>

</svg>

<figcaption><small>Figure 2: Produce processing for Cloud Topics</small></figcaption>
</figure>

The additional queue becomes a way to have the producer/networking layer release more batches into the system, however while queued in this new layer we can upload multiple batches from a single producer in parallel. Once the data is uploaded, we preserve the ordering from the producer and release those into the raft layer, waiting for the raft layer to enqueue the request before we release the next one. We still hold the producer acknowledgement for until the metadata is fully replicated in Raft. This means that we preserve the correct ordering and data durability requirements at every stage while allowing more concurrency into the latency bound part of the request processing.

What's the diff? Running [Open Messaging Benchmark][omb] with this change allowed us to easily push 10x more data and hit the GB/s scale without needing to change any producer configurations. If you're interested in learning more about Cloud Topics, check out [this blog][ct-blog] I wrote on the overall architecture. The code for the producer queue is available [here](https://github.com/redpanda-data/redpanda/blob/b167f29ee54bbe31624c194304dff206604f4255/src/v/cloud_topics/level_zero/common/producer_queue.h) and [here](https://github.com/redpanda-data/redpanda/blob/b167f29ee54bbe31624c194304dff206604f4255/src/v/cloud_topics/level_zero/common/producer_queue.cc) with the it's main usage being [here](https://github.com/redpanda-data/redpanda/blob/b167f29ee54bbe31624c194304dff206604f4255/src/v/cloud_topics/frontend/frontend.cc#L941-L983) and [here](https://github.com/redpanda-data/redpanda/blob/b167f29ee54bbe31624c194304dff206604f4255/src/v/cloud_topics/frontend/frontend.cc#L577-L793).

[wiki]: https://en.wikipedia.org/wiki/Little%27s_law
[kip]: https://cwiki.apache.org/confluence/display/KAFKA/KIP-98+-+Exactly+Once+Delivery+and+Transactional+Messaging#KIP98ExactlyOnceDeliveryandTransactionalMessaging-IdempotentProducerGuarantees
[omb]: https://github.com/redpanda-data/openmessaging-benchmark
[ct-blog]: https://www.redpanda.com/blog/cloud-topics-architecture
