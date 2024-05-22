# Welcome to part 3 of the workshop.
# Here, we will explore ways to make Julia run even faster
# as well as more advanced packages.

########## PARALLEL COMPUTING ##########

##### Threads.jl ##### (https://docs.julialang.org/en/v1/manual/multi-threading/)
# Threads is built in so no need to import it

# You can find out how many threads you have active by:
# If this number is 1, restart Julia by typing "julia -t auto"

Threads.nthreads()

# Threads gives us a very useful and powerful macro called 'Threads.@threads'
# that lets us run for loops in parallel to speed them up

using BenchmarkTools

function non_parallel_loop(N)
    array_to_be_filled = zeros(Float64, 8);
    for i in 1:8
        array_to_be_filled[i] = factorial(big(N))/factorial(big(N-1))
    end
    return array_to_be_filled
end

function parallel_loop(N)
    array_to_be_filled = zeros(Float64, 8);
    Threads.@threads for i in 1:8
        array_to_be_filled[i] = factorial(big(N))/factorial(big(N-1))
    end
    return array_to_be_filled
end

# Expensive processes get biggest benefits of multithreading
@btime non_parallel_loop(100000);
@btime parallel_loop(100000);

# As you can see below, cheaper processes have smaller gains
@btime non_parallel_loop(1000);
@btime parallel_loop(1000);

# In some cases, multithreading actually slows things down!
@btime non_parallel_loop(1);
@btime parallel_loop(1);

# You must be careful that your code is free of a data race.
# A data race occurs when two or more threads in a single process access 
# the same memory location concurrently, and at least one of the accesses is for writing, 
# and the threads are not using any exclusive locks to control their accesses to that memory
# Below is an example of where there is a data race.

function sum_multi_bad()
    s = 0
    Threads.@threads for i in 1:1000
        s += i
    end
    return s
end

sum_multi_bad()
sum_multi_bad()

### EXERCISE 1

##### Distributed.jl ##### (https://docs.julialang.org/en/v1/manual/distributed-computing/)

using Distributed, SharedArrays

# Multithreading uses a shared memory and is best run on multiple cores of the same CPU
# Distributed.jl allows you to create processes that work with separate memory (or on separate computers)
# @distributed provides the same functionality as Threads.@threads, but a shared data structure must be used.
# You can customize this by using Channels, but I won't get into that.
# The standard library package SharedArrays.jl provides this functionality for arrays.
# THe line below creates workers:

addprocs(9 - nprocs())

@everywhere using SharedArrays

function distributed_loop(N)
    array_to_be_filled = SharedVector{Float64}(8);
    @distributed for i in 1:8
        array_to_be_filled[i] = factorial(big(N))/factorial(big(N-1))
    end
    return array_to_be_filled
end

@btime non_parallel_loop(100000);
@btime parallel_loop(100000);
@btime distributed_loop(100000);

# Remove workers
for i in 2:nprocs()+1
    rmprocs(i)
end
nprocs()

########## ADVANCED DATA STRUCTURES ##########

##### DataStructures.jl ##### (https://juliacollections.github.io/DataStructures.jl/latest/)

using Pkg; Pkg.add("DataStructures");
using DataStructures 

# This package allows us to use more interesting data structuers than the ones vanilla Julia
# provides us.
# In my opinion, the most useful data structure from this package is the DefaultDict
# This works just like a dictionary, but has a default value for each key.
# This can be useful if you want to increment/add onto arrays in dictionaries
# The default value goes into the parentheses after the definition

dict = DefaultDict{String, Vector{Float64}}([]);
subjs = ["A1", "B2", "C3"];
for subj in subjs
    for i in 1:10
        push!(dict[subj], rand())
    end
end
display(dict) # Display is an alternative to println. It sometimes displays things better

# The other useful data structure is a PriorityQueue. In this structure, the elements
# in your dict will be ordered based on their value.
# 'dequeue!' will take out the top key from the queue

pq = PriorityQueue{String, Int}()
for subj in subjs
    pq[subj] = rand(1:10)
end
display(pq)
dequeue!(pq)

PriorityQueue{String, Int}(Base.Order.Reverse)
for subj in subjs
    pq[subj] = rand(1:10)
end
display(pq)
dequeue!(pq)

# In a Deque (double ended queue), you can efficiently push/pop items from the front or back.
# In Julia, Arrays have the same capability, so there's little reason to use 'Deque'

d = Deque{Int}();
push!(d, 1)
pushfirst!(d, 2)
pop!(d)
popfirst!(d)

# A 'Stack' provides last-in-first-out (LIFO) access
# 'push!' and 'pop!' are used to do this.
# A 'Queue' provides first-in-first-out (FIFO) access
# 'enqueue!' and 'dequeue!' are used to do this, but again I don't see any real benefit over the standard Arrays

s = Stack{Int}();
push!(s, 1);
push!(s, 2);
push!(s, 3);
pop!(s)

q = Queue{Int}();
enqueue!(q, 1);
enqueue!(q, 2);
enqueue!(q, 3);
dequeue!(q)

##### DataFrames.jl ##### (https://dataframes.juliadata.org/stable/)

Pkg.add(["DataFrames", "CSV"]);
using DataFrames, CSV

# This package is most similar to pandas in Python. Here is a comparison of the functions: https://dataframes.juliadata.org/stable/man/comparisons/
# As we've seen in the previous section, it's easy to load CSVs into DataFrames

df = CSV.read("laptops.csv", DataFrame)

# We can get a quick glance at the dataset by using 'describe'

describe(df)

# To get the list of all columns, use 'names' or 'propertynames'

names(df)
propertynames(df)

# 'size' gives the dimensions of the dataframe

size(df)

# We can select columns in multiple ways. These will not produce a copy,
# and altering them alters the original dataframe

df.Price
df[!, "Price"]

# To make a copy, use either of the following:

df[:, "Price"]
df[:, :Price]

# You can select multiple rows/columns simply like this:

df[1:10, [:Price, :Screen_Size]]
df[1:10, [3, 10]]

# In Julia, you can use @view on indexed arrays/dataframes to force
# it not to make a copy. This can be very memory efficient but dangerous

@view df[1:10, [3, 10]] # creates a "SubDataFrame'

# If you want to select only some rows, use

small_screens = df[df.Screen_Size .== 14.0, [:Price, :Screen_Size]]

# If you want to split up the dataframe based on a column, use

scren_size_group = groupby(df, :Screen_Size);
scren_size_group[1].Screen_Size
scren_size_group[2].Screen_Size
scren_size_group[3].Screen_Size

# The functions dropmissing and dropmissing! can be used to remove the rows containing missing values 
# from a data frame and either create a new DataFrame or mutate the original in-place respectively.

size(df)
size(dropmissing(df))

########## ADVANCED STATS ##########

##### StatsBase.jl ##### (https://juliastats.org/StatsBase.jl/stable/)

Pkg.add("StatsBase");
using StatsBase

# Think of StatsBase as an extension to Statistics, that adds a bit more detail
# We get can more fancy properties of data like the standard error of the mean, 
# inter-quantile range, mode, higher order moments like skew and kurtosis

x = randn(100);
sem(x)
iqr(x)
mode(x)
skewness(x)
kurtosis(x)

# You can also calculate the fancy means:

geomean(x .+ 3) # doesnt work with negative numbers
harmmean(x)

# To get some base info on your vector, you can use 'describe'

describe(x)

# You can easily z-score you data using 'zscore'

zscore(x)

# Given a collection of probabilities, you can calculate the entropy, cross-entropy and KL divergence with:

p = [0.1, 0.4, 0.2, 0.3];
q = [0.2, 0.3, 0.2, 0.3];

entropy(q)
crossentropy(p, q)
kldivergence(p, q)

# We can also get the root mean squared deviation between two vectors

x = randn(100);
y = 3 .+ randn(100);
rmsd(x, y)

# 'countmap' (and 'proportionmap') return a dictionary of how often each element of a collection is present

x = rand(1:5, 1000);
countmap(x)
proportionmap(x)

# 'ordinalrank' (no ties) 'competerank' and 'denserank' allow ranking of elements in an array

x = [3, 1, 1, 2];
ordinalrank(x)
competerank(x)
denserank(x)

# 'wsample' lets you sample from an array given some probabilities (replace can be used to set sampling with/without replacement)

wsample(["A", "B", "C"], [0.9, 0.05, 0.05], 10)
wsample(["A", "B", "C"], [0.9, 0.05, 0.05], 2, replace=false)

##### Distributions.jl ##### (https://juliastats.org/Distributions.jl/stable/)

Pkg.add(["Distributions", "Plots"]);
using Distributions, Plots

# This package lets us sample from, get the pdf/cdf and other properties of many statistical distribitions.
# The list of available distributions in in their documentation. I will not list it since there are >100
# The syntax is as follows: first, define a distribution object, and then call functions around it.

d = Normal()
mean(d)
var(d)

# To sample from the distribution, use 'rand'

rand(d, 10)

# 'fieldnames' tells you about the parameters of the distribution

fieldnames(Normal)
d = Normal(1, 2)

# If we want to get the pdf/cdf (e.g. for plotting), we need to give the function
# a range of x values to evaluate the pdf/cdf on

x = -4:0.01:4;
plot(x, pdf.(d, x))
plot!(x, cdf.(d, x))

# To create truncated distributions, use:

lower_limit = -2
upper_limit = 2
d = truncated(Normal(), lower_limit, upper_limit);
plot(x, pdf.(d, x))

# We can also easily fit the parameters of a distribution to some data:

data = rand(Poisson(5), 1000);
d = fit(Poisson, data)

x = 0:15;
histogram(data, normalize=:probability, bins=x .- 0.5, label="data")
plot!(x, pdf.(d, x), linewidth=3, label="fit")

##### HypothesisTests.jl ##### (https://juliastats.org/HypothesisTests.jl/stable/)

Pkg.add("HypothesisTests");
using HypothesisTests

# This package lets us do all sorts of hypothesis tests like t-test, ANOVA, Mann-Whitney, etc
# The documentation is amazing and can help you decide which test is best for your use case.
# The example below is for a two sample t-test. You can obtain the p-value using the 'pvalue' function

x = randn(50);
y = randn(50) .+ 2;
test = EqualVarianceTTest(x, y)
pvalue(test)
dump(test)

### EXERCISE 2

########## INTERACTING WITH OTHER LANGUAGES ##########

##### PythonCall.jl ##### (https://juliapy.github.io/PythonCall.jl/stable/)

Pkg.add(["PythonCall", "CondaPkg"]);
using PythonCall, CondaPkg

# You can use 'pyimport' to import python packages/your own Python functions
# CondaPkg.jl is used to install python packages (https://github.com/JuliaPy/CondaPkg.jl)

CondaPkg.add("matplotlib")

plt = pyimport("matplotlib.pyplot")
x = range(0;stop=2*pi,length=1000); y = sin.(3*x + 4*cos.(2*x));
plt.plot(x, y, color="red", linewidth=2.0, linestyle="--")
plt.show()

# You can run python code directly in Julia using 'pyexec'

fib(x) = pyexec("""
def fib(n):
    a, b = 0, 1
    while a < n:
        print(a, end=' ')
        a, b = b, a+b
    print()
fib(y)
""", Main, (y=x,))

fib(10)

##### RCall.jl ##### (https://juliainterop.github.io/RCall.jl/stable/)

Pkg.add("RCall");
Pkg.build("RCall");
using RCall

# Run 'using RCall' in the terminal and press '$' to enter R command line interface
# Otherwise, place R before the string of R code

b = [1, 2, 3];
R"a <- $b"

##### MATLAB.jl ##### (https://github.com/JuliaInterop/MATLAB.jl)
# # DOES NOT WORK ON APPLE SILICON (M1-3 chips)
# Pkg.add("MATLAB");
# using MATLAB
# # You can use 'mxcall' to call existing/custom MATLAB functions
# # Otherwise, you can use mat in front of a string of MATLAB code
# mat"a=[0 1; 1 0]"