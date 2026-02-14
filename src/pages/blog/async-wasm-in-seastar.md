---
layout: ../../layouts/BlogPost.astro
title: "Running Async WebAssembly on Seastar's Reactor"
date: "2026-02-14"
readTime: 7
coverImage: "/blog/images/seastar.jpg"
---

In late 2023, I landed a series of patches to [Wasmtime](https://wasmtime.dev) and [Redpanda](https://redpanda.com) that moved WebAssembly (Wasm) execution off of dedicated threads and onto [Seastar's](https://seastar.io) reactor threads. The result was a 3x throughput improvement for Redpanda's [Data Transforms](https://docs.redpanda.com/current/develop/data-transforms/). This post is a deep dive into how we bridged the gap between Rust's async model, Wasmtime's fiber-based concurrency, and Seastar's cooperative reactor — and the trick we used to support async host functions without extra polling overhead.

### Background: Two Flavors of Async

Before getting into the details, it's worth briefly contrasting how Rust and C++ (specifically Seastar) approach asynchronous programming, because the impedance mismatch between these two models is at the heart of the problem.

#### Rust Futures

Rust's `Future` trait is simple on the surface:

```rust
trait Future {
    type Output;
    fn poll(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Self::Output>;
}
```

A future is a state machine that you repeatedly `poll` until it returns `Poll::Ready(value)`. The runtime (like [tokio](https://tokio.rs)) is responsible for calling `poll`, and the `Context` carries a `Waker` that the future uses to signal "poll me again, I might be ready now." Rust's `async`/`await` syntax compiles down to these state machines - each `.await` point becomes a variant in a generated enum, and the compiler transforms your sequential-looking code into a series of poll-able states.

The important thing to note is that **Rust futures are lazy and pull-based**. Nothing happens until you poll them, and they only make progress when polled.

#### Seastar Futures

Seastar takes a different approach. It's a C++ framework for high-performance I/O built around a **thread-per-core reactor model**. Each physical CPU core runs a single thread with an event loop, and Seastar's `future<T>` type is different: when a future becomes ready, it immediately invokes the attached continuation (assuming there pending I/O). There's no polling at this level - although the reactor does poll for low latency I/O. You compose these futures by chaining `.then()` continuations:

```cpp
seastar::future<int> read_and_parse() {
    return read_from_network().then([](std::string data) {
        return std::stoi(data);
    });
}
```

With C++20, Seastar also supports coroutines, which is much more ergonomic:

```cpp
seastar::future<int> read_and_parse() {
    auto data = co_await read_from_network();
    co_return std::stoi(data);
}
```

Seastar is a thread-per-core architecture, meaning that we run only a single thread per core. Within that thread we use cooperative multi-tasking to ensure that we can be responsive to I/O (client network traffic, disk reads finishing, etc.). This is a big paradigm shift for many, as what this boils down to is we have a latency budget for each task that runs on the reactor (~500μs) - if a task needs to take longer than that before performing asynchronous I/O it needs to yield control of the CPU itself (this is one of the key mechanisms we use to keep Redpanda's [tail latencies low](https://www.redpanda.com/blog/what-makes-redpanda-fast)). This is different than say, Golang, which uses [preemptive scheduling](https://hidetatz.github.io/goroutine_preemption/) as opposed to cooperative scheduling like Seastar.

### The Problem: Wasm on the Reactor

Previously, Redpanda ran all WebAssembly execution on dedicated [alien](https://docs.seastar.io/master/namespaceseastar_1_1alien.html) threads (non-Seastar threads dedicated for Wasm execution). Host functions — the functions Wasm calls back into for things like reading records or producing output — would bounce work back to a Seastar shard via `seastar::alien::submit_to()`, wait for the result with `.get()`, and then return. This worked, but it meant every record processed incurred the overhead of cross-thread synchronization. It also meant Wasm execution was scheduled at the mercy of the kernel, instead of being able to rely on Seastar's userspace task scheduler.

The goal was to run Wasm **directly on the reactor thread** - no thread bouncing, no synchronization overhead. But this introduced two challenges:

1. **Cooperative yielding**: Wasm code is an opaque blob of computation. If a guest enters a tight loop, the reactor starves. We need a way to periodically yield control back to the reactor from inside the VM.
2. **Async host functions**: Some host functions are inherently asynchronous (e.g., reading from a Seastar I/O queue). When Wasm calls one of these, we need to suspend the VM, return to the reactor, and resume when the I/O completes - all without blocking or unnecessary polling.

Both of these require Wasmtime's async support.

### Wasmtime's Async Model: Fibers Under the Hood

Wasmtime's solution to async is built on [stackful coroutines (fibers)](https://docs.wasmtime.dev/api/wasmtime/struct.Config.html#asynchronous-wasm). Rather than rewriting Wasm into Rust state machines (which would be incredibly complex for arbitrary compiled code), Wasmtime allocates a separate native stack for Wasm execution.

When you call `wasmtime_func_call_async()`, the VM runs on its own fiber stack. From Wasm's perspective, everything is synchronous - it has no idea fibers exist. But the host can **switch away** from the fiber at any point, returning a `wasmtime_call_future_t` that can be polled later. The mechanism looks like this:

```
Host (reactor thread)                    Fiber Stack (Wasm VM)
       |                                         |
       |  poll(call_future)                      |
       | -------- stack switch --------->        |
       |                                    [Wasm runs]
       |                                    [calls host fn]
       |                                    [host fn needs I/O]
       | <------- stack switch ----------        |
       |  returns Poll::Pending                  |
       |                                    [suspended]
       |  ... reactor runs other tasks ...       |
       |  ... I/O completes ...                  |
       |  poll(call_future)                      |
       | -------- stack switch --------->        |
       |                                    [host fn returns]
       |                                    [Wasm continues]
```

This design means Wasmtime's async API presents as a standard pollable future to the host runtime, while Wasm code is none the wiser.

#### Cooperative Yielding via Fuel

Wasmtime injects checks at strategic points in compiled Wasm code (loop headers, function prologues) that decrement a "fuel" counter. When fuel runs out, execution is interrupted. In async mode, you can configure this to **yield** instead of trap - the fiber switches back to the host, which can refuel and resume later. There's also epoch-based interruption, but for our purposes fuel was easier to integrate with.

On the Seastar side, our poll loop for driving the Wasm future looks like this:

```cpp
seastar::future<> wasmtime_engine::call(/* ... */) {
    auto fut = wasmtime_func_call_async(ctx, &func, args, nargs,
                                        results, nresults, &trap, &error);
    while (!wasmtime_call_future_poll(fut.get())) {
        if (_pending_host_function) {
            auto host_future = std::exchange(_pending_host_function, {});
            co_await std::move(host_future).value();
            continue;
        }
        co_await seastar::coroutine::maybe_yield();
    }
    // handle trap/error...
}
```

Each time the Wasm future returns "not ready" (from a fuel yield), we `co_await maybe_yield()` - giving the Seastar reactor a chance to run other tasks.

### The Side-Channel: Async Host Functions Without Extra Polling

The trickier problem is async host functions. When Wasm calls a host function that needs to do I/O (like reading the next record batch), we need to:

1. Start the asynchronous operation (a Seastar future)
2. Suspend the Wasm VM (return from the fiber)
3. Wait for the I/O to complete on the reactor
4. Resume the Wasm VM with the result

The natural approach with Wasmtime's C API would be to return `false` from the continuation callback (meaning "I'm not done yet"), let the poll loop spin, and eventually return `true` when the I/O finishes. But this means the poll loop has no idea what it's waiting for — it would have to busy-poll or sleep-poll, which is wasteful and adds latency.

Instead, we use what I think of as a **side-channel future**. The async host function registers its Seastar future directly on the engine, and the poll loop knows exactly what to await:

```cpp
template<typename Module, auto HostFn, typename ReturnType, typename... Args>
void invoke_async_host_fn(
    wasmtime_caller_t* caller,
    wasmtime_val_t* args, size_t nargs,
    wasmtime_val_t* results, size_t nresults,
    wasm_trap_t** trap_ret,
    wasmtime_async_continuation_t* continuation) {

    auto* engine = static_cast<wasmtime_engine*>(
        wasmtime_context_get_data(wasmtime_caller_context(caller)));

    // Invoke the host function — it returns a seastar::future<ReturnType>
    auto host_future = seastar::futurize_invoke(HostFn, extract_args(args)...);

    if (host_future.available()) {
        // Fast path: if the future completed synchronously (common!),
        // handle it inline without any suspension.
        results[0] = to_wasm_val(host_future.get());
        continuation->callback = [](void*) { return true; }; // already done
        return;
    }

    // Slow path: I/O is pending.
    // Allocate a flag the continuation can check.
    auto* done = new async_call_state();
    host_future.then([done, results](ReturnType val) {
        results[0] = to_wasm_val(val);
        done->finished = true;
    });

    // Register this future with the engine so the poll loop can await it.
    engine->_pending_host_function = std::move(host_future);

    // The continuation just checks the flag.
    continuation->env = done;
    continuation->finalizer = [](void* p) { delete static_cast<async_call_state*>(p); };
    continuation->callback = [](void* env) {
        return static_cast<async_call_state*>(env)->finished;
    };
}
```

The key insight is in the poll loop shown earlier. When `wasmtime_call_future_poll` returns `false`, we check `_pending_host_function`. If there is one, we `co_await` it directly — no busy polling, no wasted reactor cycles. The Seastar reactor runs other tasks while the I/O is in flight, and as soon as it completes, we resume the Wasm fiber.

### Upstreaming the C API

None of this would have been possible without async support in Wasmtime's C API, which didn't exist before. The Rust API had async support for a long time, but C/C++ embedders were left out. I opened [PR #7106](https://github.com/bytecodealliance/wasmtime/pull/7106) to add the full async C API, which exposes:

- `wasmtime_config_async_support_set` / `wasmtime_config_async_stack_size_set` — configure async mode
- `wasmtime_linker_define_async_func` — register async host functions
- `wasmtime_func_call_async` — invoke Wasm and get back a pollable future
- `wasmtime_call_future_poll` / `wasmtime_call_future_delete` — drive the future to completion
- `wasmtime_context_epoch_deadline_async_yield_and_update` — epoch-based cooperative yielding

The core of the Rust implementation bridges C callbacks to Rust futures. The `wasmtime_async_continuation_t` struct implements Rust's `Future` trait — `poll()` calls the C callback and maps the boolean return to `Poll::Ready` / `Poll::Pending`. This means Wasmtime's internal fiber machinery works unchanged; we just swap out the Rust async closure for a C callback wrapper. If you're interested in the design discussions, the [original issue #3111](https://github.com/bytecodealliance/wasmtime/issues/3111) has context going back to 2021.

### Results

After landing [the Redpanda PR](https://github.com/redpanda-data/redpanda/pull/14046), the Wasm execution moved entirely onto reactor threads. The alien thread pool was replaced with a single compilation thread (since module compilation is still CPU-heavy and shouldn't block the reactor). The runtime config shrunk to:

```cpp
wasmtime_config_async_support_set(config, true);
wasmtime_config_async_stack_size_set(config, 128_KiB);
wasmtime_config_max_wasm_stack_set(config, 64_KiB);
```

128 KiB for the fiber stack, 64 KiB for Wasm, leaving 64 KiB for host functions — small stacks that fit in cache and allow many concurrent Wasm guests per core. See my [previous blog](https://rockwotj.com/blog/protecting-against-stack-overflow-in-wasmtime-host-functions/) for more information on the stack sizes.

The measured improvement was **3x throughput** in Data Transforms benchmarks. The wins came from eliminating cross-thread synchronization, better cache locality (everything runs on one core), and the ability to interleave Wasm execution with other reactor work via cooperative yielding.

### Takeaways

If you're embedding Wasmtime into a non-Rust application with its own event loop, the async C API gives you everything you need to integrate Wasm as a cooperative citizen of your runtime. The pattern is:

1. Enable async support and fuel/epoch-based interruption
2. Use the pollable future API to integrate with your event loop
3. Register async host functions that communicate pending I/O through a side channel
4. Exploit the fast path for synchronously-completing operations

The side-channel pattern in particular is useful beyond Wasmtime — any time you need to bridge a poll-based API (like fibers or coroutines) with a push-based runtime (like Seastar), having the caller register its pending work where the driver can see it avoids the overhead of blind polling.

Thanks for reading! If this sort of low-level systems work is interesting to you, all the code referenced in this post is open source — go explore the [Wasmtime C API](https://github.com/bytecodealliance/wasmtime/blob/main/crates/c-api/include/wasmtime/async.h) and see how it all fits together.
