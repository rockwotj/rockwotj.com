---
layout: ../../layouts/BlogPost.astro
title: "Musings on KubeCon WasmDay"
date: "2023-11-07"
readTime: 3
---

I had a great time today giving my first conference talk at [Cloud Native Wasm Day](https://events.linuxfoundation.org/kubecon-cloudnativecon-north-america/co-located-events/cloud-native-wasm-day/#thank-you-for-attending). ~~I'll have the link to the recording posted later if you want to give it a listen~~ (EDIT: [the recording is live here](https://youtu.be/t4-Al2FoU0k?si=E7hvzSlTWwduDhql)).

Listening to the other presenters speak, one that stood out to me was one on [WasmCloud](https://wasmcloud.com/). WasmCloud uses the latest and greatest [Component Model from the Bytecode Alliance](https://component-model.bytecodealliance.org/) as the key ingredient. As a recap for those of us who don't live in the cutting edge of the WebAssembly world, my summary of the Component Model is that it is essentially two things on top of existing WebAssembly:

1. A [Canonical ABI](https://component-model.bytecodealliance.org/design/canonical-abi.html) that defines a common in-memory representation for types that are cross language. Today most languages can interop via C bindings, ala `extern "C"` in languages like Rust, JNI for Java and the N-API for NodeJS. However, unlike those bindings they work both ways, IE Rust could theoretically call into Java as they both speak the same ABI. Truly this is one FFI to rule them all.

2. A bunch of tooling for automating a higher level [IDL](https://en.wikipedia.org/wiki/Interface_description_language) (think protocol buffer gRPC service/methods if you're familiar with `.proto` files), and generating code in all the languages so they can talk to each other.


That's a little bit of a simplification and that dream is not yet fully utilized, but it's cool to see WebAssembly be able to interop between languages so well. Two talks at Cloud Native Wasm Day extolled the component model, and I recommend giving them both a watch if you want to learn more here ([1](https://colocatedeventsna2023.sched.com/event/1Rj12/webassembly-component-model-enhancing-security-productivity-and-green-computing-bailey-hayes-cosmonic-kate-goldenring-fermyon-technologies-inc) and [2](https://colocatedeventsna2023.sched.com/event/1Rj1r/a-love-letter-to-isolation-kelly-shortridge-fastly)).

WasmCloud is interesting because it takes this component model and automagically creates RPC boundaries for these components to talk to each other, then allows deploying these anywhere and hooking up the network so they can talk to each other.

While this is cool on its own, I got excited because of a recent paper from Google I had just read: [**Towards Modern Development of Cloud Applications**](https://sigops.org/s/conferences/hotos/2023/papers/ghemawat.pdf)**.** I highly recommend giving the paper a read, but my TL;DR is they poke at the microservice trend and state that we want to write code in a monolithic fashion because it's easy to reason about and microservices end up being a spaghetti of different services mashed together and it becomes very difficult to deploy them, make breaking changes, and have real operational and performance overhead. They then go on to show an example of how you could create "components", then write normal function calls and at runtime have a scheduler decide if they should be in the same process or different ones (meaning they become RPCs). The benefits are you can independently scale these components and split them up (or combine them) as needed. Another cool thing they outline is how to do **atomic** rollouts where only a single version of a component only communicates to the same version of another component, this the scheduler can atomically roll out the new code without breakages. If you've ever built with microservices, I don't know how you can't be yelling yes to understanding this pain.

**WasmCloud** is **so close** to being the model described in this paper with the difference being having an IDL instead of constructs within a language to define a component, and you get the WebAssembly sandbox for security. The pros of the in-language construct is that it's less friction to define components, but the pros of the IDL is that you can communicate between languages, or even use it to draw a line in the sand on where communication with existing services (stateful storage, non-wasm components, etc) happen.

I'm looking forward to the day when I can deploy my monolithic app as WebAssembly and have a k8s like environment shard it automatically into microservices.
