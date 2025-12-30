---
layout: ../../layouts/BlogPost.astro
title: "C++ Memory Management Rules"
date: "2021-12-18"
readTime: 3
---

Do you ever wonder how giant company's like Google have millions of lines of C++ and can still have robust systems? Is the answer to rewrite all the things in Rust<sup>[1](#footer1)</sup>?

During my time at Google I learned the rules that are used to keep the owner of memory very clear. First usage of `new` and `delete` should be considered a code smell and avoided. A better choice is smart pointers via`std::make_unique` and `std::make_shared`. If you cannot use those because your objects have private constructors like so:

```c++
class Bar {
  public:
    std:: unique_ptr<Bar> Create();
  private:
    Bar();
}
```

Then in your `Bar::Create` method to have the creation of Bar look something like:

```c++
return absl::WrapUnique(new Bar());
```

So it's impossible to leak the memory.

Another important aspect of memory management is calling other functions. Let's take an example signature:


```c++
void Foo(const Bar& bar, std::unique_ptr<Baz> baz, Qux* qux);
```

A Bar class is passed by constant reference, which signifies that Foo will not keep any references to it after it returns. A smart pointer to the Baz class is passed, which signifies that Foo is taking ownership of that memory. Lastly is Qux, which is a raw pointer. A raw pointer signifies that Foo is allowed to mutate Qux, but shouldn't keep a reference past the function call. Sometimes you need to have a single class own multiple objects, and those objects need references to one another. In that case I was used to seeing comments like this in the declaration:

```
std::unique_ptr<Bar> MakeBar(/*unowned*/Qux* qux);
```

Which signifies that Qux needs to outlive Bar. These sorts of relationships should ideally be set at the start of the program and be classes that stick around for the whole program's life time, as these are cases where it's easy to mess up (and where Rust shines over these conventions).

The advantage of these rules is that it's clear at the callsite what's going on.

```cpp
Foo(bar, std::move(baz), &qux);
```

We can look and tell baz is now owned by Foo and Foo may mutate qux. That last point is why non-const r-values should not be used as function parameters, it's hard to tell from the callsite if an class can be mutated.

Take an example from the standard library, and you are preforming a compare and swap loop:

```
template<typename T>
class stack
{
    std::atomic<node<T>*> head;
 public:
    void push(const T& data)
    {
      node<T>* new_node = new node<T>(data);

      // put the current value of head into new_node->next
      new_node->next = head.load(std::memory_order_relaxed);

      // now make new_node the new head, but if the head
      // is no longer what's stored in new_node->next
      // (some other thread must have inserted a node just now)
      // then put that new head into new_node->next and try again
      while(!head.compare_exchange_weak(new_node->next, new_node))
          ; // the body of the loop is empty
    }
};
```

The first time I read that code I thought the loop would run forever because nothing changes, however, if you look at the function declaration:

```cpp
bool compare_exchange_weak( T& expected, T desired);
```

`new_node->next` is being updated each iteration of the loop! Now if that had followed the rules above it's be much more obvious that was happening.

If you're interested in more information here I suggest checking out the [Google C++ style guide](https://google.github.io/styleguide/cppguide.html#Inputs_and_Outputs) or emailing me.

Thanks for reading!


<div id="footer1"><sup>1</sup> FWIW I'm a huge fan of Rust, but C++ isn't going anywhere.</div>
