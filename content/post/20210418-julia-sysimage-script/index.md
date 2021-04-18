---
slug: julia-sysimage-script
date: 2021-04-18T00:00:00.000Z
title: "How to Reduce the Loading Time of Julia Scripts"
description: "Creating and optimizing custom sysimages"
tags:
  - julia
  - tip
keywords:
  - julia
  - tip
url: /post/julia-sysimage-script/
---

{{< figure src="featuredImage.jpg" caption="[Photo Credit](https://pixabay.com/photos/dog-attention-mixed-breed-dog-6082017/)" >}}

## Motivation

Julia is a promising new language for scientific computing and data science. I've demonstrated that doing whole work masking in Julia can be a lot faster (up to 100x) than in Python [in this post](/post/julia-whole-word-masking/). The secret of Julia's speed is from its use of JIT compilers (rather than interpreters used by R and Python). However, this design also impedes Julia's ambition as a general-purpose language since ten seconds of precompiling time for a simple script is unacceptable for most use cases.

Let's use this use case of mine as an example: I have a few CSV files containing price histories of some products. I want to write a script that reads in a file and prints out the history of price changes for a product. Analysts will run this script whenever they wish to analyze the pricing history of a product. Without any optimization, this script would take more than fifteen seconds to run, which is obviously not ideal for such a simple task and creates an abysmal user experience.

### PackageCompiler

Although still not as convenient as compiling executable from Golang or C++, Julia provides a tool called [PackageCompiler](https://julialang.github.io/PackageCompiler.jl/dev/) to address this issue. It enables you to create [sysimages](https://julialang.github.io/PackageCompiler.jl/dev/sysimages/) (serialized Julia sessions) to speed up loading time and even create [executables](https://julialang.github.io/PackageCompiler.jl/dev/apps/) that can run independently.

Two of the major drawbacks of PackageCompiler are:

1. The size of sysimages is rather large (more than 100 MB).
2. The sysimage needs proper optimization to achieve a satisfying speedup.

Because the large-volume SSD disk is already really cheap right now, the first drawback might be acceptable in local environments. I'm not familiar with the executable creation of PackageCompiler, but I'd guess the sizes of executables will also be quite large. This could create some problems if we want to distribute executables via the Internet.

The second drawback is really the restriction of the JIT compilers. People with experience working with TorchScript/PyTorch JIT should be able to empathize with dealing with this kind of intricacy. The good news is that PackageCompiler provides some helpers for you to reduce the pain, but first, you need to learn how to use them.

The rest of the post will walk you through a simple working example that is hopefully easier to read than the official documentation. By the end of this post, we'd reduce the loading time of our simple script from about 13 seconds to 350 milliseconds (a 35x+ speedup).

| Custom Sysimage      | Loading Time | Image Size |
| -------------------- | ------------ | ---------- |
| Built-in sysimage    | 13 s         | 180 MB     |
| Packages             | 7.5 s        | 183 MB     |
| Packages + functions | 350 ms       | 190 MB     |

## The Target Script

Our target script comes from the price history example (saved as `example.jl`):

```julia
using CSV, DataFrames, Dates

df = DataFrame(CSV.File("data/some_prices.csv", types = Dict(:id => String)))
df.time = Dates.epochms2datetime.(df.timestamp .* 1000) .+ Dates.Year(1970) .+ Dates.Hour(8)
```

It reads the price CSV file into a data frame and converts epoch times in the timestamp column into DateTime objects. There is, of course, some code that extracts the merchandise and price change points. I omit those parts since these first three lines already take up the majority of the loading time.

## Show which function is being Compiled

We can let Julia print out the exact function currently being compiled via the `--trace-compile=stderr` argument. Example: `julia --trace-compile=stderr some_script.jl`. The output looks like this:

```julia
precompile(Tuple{typeof(Base.reinterpret), Type{Int64}, Array{UInt8, 1}})
precompile(Tuple{typeof(Base.getindex), Array{UInt32, 1}, Int64})
precompile(Tuple{typeof(Base.getindex), Array{String, 1}, UInt32})
```

You’ll see many functions being compiled, which all contribute to the long loading time.

We can store this list into a file via `--trace-compile=some_file.jl` and use this file to create a sysimage. More on this later.

## A sysimage with Packages Loaded

The most straightforward way to create a sysimage is to pass a list of packages we want to precompile:

```julia
using PackageCompiler

create_sysimage([:CSV, :DataFrames, :Dates], sysimage_path="sys_simple.so")
```

(You need to have PackageCompiler, CSV, DataFramse, and Dates already installed.)

You can now use the `-J` argument to specify which sysimage to use:

```bash
julia -J sys_simple.so example.jl
```

If you enter REPL via `julia -J sys_simple.so`, you should see the CSV package already loaded. No `import` or `using` statement is needed.

This sysimage reduces the loading time from 13 seconds to 7.5 seconds, which is still not ideal. If you check the output from `--trace-compile`, you can see there is still a lot of compiling going on. This is because Julia uses dynamic dispatching; that is, it determines the types of the input arguments in real-time and compiles only the function with the corresponding signature.

### A sysimage with Packages Loaded and Functions Compiled

Besides loading the packages, we can also ask PackageCompiler to run a script and compile functions inside that script:

```julia
using PackageCompiler

create_sysimage(
    [:CSV, :DataFrames, :Dates],
    sysimage_path="sys_complete.so",
    precompile_execution_file="example.jl" # the new line
)
```

I use the target script (`example.jl`) above, but you might want to craft a separate script if your use case is more complicated (a CLI with many modes). Because of the dynamic dispatching mechanism, you'd want to make sure you cover as many compile-time-consuming function calls in this file as possible. Since properly done unit tests cover all possible inputs, one simple way to do this is to run the unit test inside this file.

You can also use the precompile list from `--trace-compile` here. It can be helpful for interactive applications where you aren't willing to write unit tests (not a recommended practice, though).

This sysimage successfully reduce the loading time to under one second. Hooray! Admittedly, it probably isn't worth it if you have to jump through all these hoops to get a loading time as fast as Python (Python takes about 300 ms to do the same thing as the example script). But it's definitely gonna worth it if you're running some time-consuming operations inside the script (e.g., string manipulation). I'll write more about such situations shortly. Stay tuned.
