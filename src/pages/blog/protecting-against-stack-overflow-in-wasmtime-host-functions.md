---
layout: ../../layouts/BlogPost.astro
title: "Protecting against stack overflow in Wasmtime Host Functions"
date: "2023-12-30"
readTime: 4
---

Recently I was able to contribute a patch to Wasmtime for supporting custom runtime stacks for it's [asynchronous execution mode](https://docs.wasmtime.dev/api/wasmtime/struct.Config.html#asynchronous-wasm) ([#7209](https://github.com/bytecodealliance/wasmtime/pull/7209)). This is post documents the abilities that unlocks and allows for better testing against stack overflow for hosts that embed Wasmtime.

### Wasmtime Async Support

First, an explanation of how asynchronous functionality works in Wasmtime - if you're familiar with this you can skip to the next paragraph. WebAssembly does not have a notion of asynchronous execution, so this is achieved by using stackful coroutines within the Wasmtime runtime. As a refresher, an operating system thread (i.e. `pthread_create` or `std::thread` in C++), has a stack, with a bunch of information about the current execution frame, and all the caller frames. Below is a nice diagram from [Wikipedia's article on Call stacks](https://en.wikipedia.org/wiki/Call_stack).

[![Image of a CallStack from Wikipedia](/blog/images/call-stack-layout.png)](https://en.wikipedia.org/wiki/Call_stack)

Stackful coroutines, multiplex multiple of these stacks onto a single operating system thread. This is done by saving the current stack pointers, frame pointers and [callee saved registers](https://en.wikipedia.org/wiki/Calling_convention) in memory, then restoring that same information from another stack and jumping to the right location. If you're interested in the details here, I have written a tiny version of this in C++ [here](https://github.com/rockwotj/wasmcc/tree/main/runtime/thread).

The nice thing about this approach to concurrency, is that other code is oblivious to what is happening. It's completely transparent to the calling function that the stack was switched away when it called some function. This property allows the WebAssembly VM to switch to/from the Wasm code executing without the host or guest having any idea what is going on. The stack switching is also used to support host functions that are asynchronous: the Wasm VM stack calls the async function, and then switches back to the host that called into the VM. The host is responsible for calling back into the VM when the asynchronous operation has finished (i.e. the future has completed), which then the VM stack can be switched back to and the guest running in the VM never knew that it was suspended.

### Custom Stacks

Ok, back to [PR #7209](https://github.com/bytecodealliance/wasmtime/pull/7209). Wasmtime allocates these VM stacks for it's async mode using [anonymous `mmap`](https://en.wikipedia.org/wiki/Mmap#File-backed_and_anonymous). The patch supports hosts the ability to plug in custom stack memory, perhaps using preallocated memory (note this is for unix only, as Windows requires usage of it's [own APIs for fibers](https://nullprogram.com/blog/2019/03/28/)). One of non-obvious advantages of this patch is that it allows us to know the bounds of this stack that Wasmtime executes both guest code and host functions. Why would this be useful? Well Wasmtime provides limiting the VM's usage on the stack via the [`max_wasm_stack`](https://docs.rs/wasmtime/13.0.0/wasmtime/struct.Config.html#method.max_wasm_stack) configuration option. To quote the documentation for that function:

> When the `async` feature is enabled, this value cannot exceed the `async_stack_size` option. Be careful not to set this value too close to `async_stack_size` as doing so may limit how much stack space is available for host functions.

Plugging in our own custom stack, and knowing the bounds of it allows us to ensure host functions (which also run on this other stack) don't overflow the stack (which usually results in a SIGABRT on Linux when the [guard page](https://stackoverflow.com/questions/73604324/what-is-stack-guard-page-and-probing-stack) is written too). How can ensure our code doesn't stack overflow? By testing! Generally Wasm code will use very little stack space (most of the programs I test against in my day job only use a few kilobytes of stack space), but a malicious program could be crafted to use just up to the limit of the stack then call a host function. Most of the time the guest code is unlikely use most of the stack limit it's allowed: `512KiB` by default in Wasmtime or 1/4 of the default async stack size of `2MiB`. So there could be conditions in which you have a host function that uses `1.7MiB` of stack memory and it would only cause issues if you had guest code that called a host function while close to the limit.

In order to combat this, it's possible to force that host functions **always** run with the most limited amount of stack space using [`alloca`](https://man7.org/linux/man-pages/man3/alloca.3.html) or [variable length arrays (VLA)](https://en.wikipedia.org/wiki/Variable-length_array). Here's how to do it in pseudo code:

```python
def host_function(caller):
  int a = 0;
  # Use the stack address to find our current stack bounds
  int [bottom, top] = find_allocated_stack_bounds(&a)
  int stack_left = &a - bottom
  alloca(stack_left - 1.5MiB)
  # Now call the normal host call implmentation with a reduced stack
  return do_host_function_impl(caller)
```

I implemented this technique in [Redpanda PR #14161](https://github.com/redpanda-data/redpanda/pull/14161) if you want to see this in a real world embedding. Now in order for this to work, it's important that there is sufficient test coverage over all host functions and all the branches are covered (but you're doing that already 😉).

This same technique could work if you're only using Wasmtime's synchronous APIs via [`pthread_attr_getstack`](https://linux.die.net/man/3/pthread_attr_getstack) instead of using a custom stack to know the stack bounds. Although arguably Wasmtime itself should have a feature to force the maximum guest stack usage when calling host functions!

### Takeaways

Thanks for reading! If you're someone who is embedding Wasmtime into their application, hopefully you've learned a trick to make your embedding more robust. Otherwise, if you are into streaming, you know more about how Redpanda makes Data Transforms robust and secure. If this sort of work is interesting to you, reach out, [my team is hiring](https://redpanda.com/careers#job-listings)!
