# Welcome to part 2 of the workshop.
# Here, we will delve into essential packages you need to get started
# with using Julia for your own projects.

########## STANDARD LIBRARY PACKAGES ##########

##### Random.jl ##### (https://docs.julialang.org/en/v1/stdlib/Random/)

using Random

# This package includes basic functions to deal with random numbers
# First, we can set the seed for the random number generator with

Random.seed!(0)

# 'rand' generates random floats between 0 and 1

rand(5)
rand(3, 3)

# It can also be used to generate random numbers from a specified collection

rand(1:3, 5) # 5 random numbers from [1, 2, 3]

# 'bitrand' is used to generate an array of 1s and 0s 

a = bitrand(5)

# We can use this array to index another.
# This essentially selects elements of the array with 50% chance

b = [1, 2, 3, 4, 5]
b[a]

# 'randn' works like 'rand' but generates normally-distributed samples

randn(5)

# 'randexp' works like 'rand' but generates exponentially-distributed samples

randexp(5)

# 'randstring' generates a random string

randstring(10)
randstring('a':'z', 10)

# 'ransubseq' generates a random subsequence of a given array with a probability p
# This essentially subsamples your array

p = 0.5 #(include each element with 50% probability)
randsubseq(1:5, p)

# 'randperm' gives a random permutation of a given length.
# This permutation can be used to index an array to shuffle it.

a = ["first", "second", "third", "fourth", "fifth"]
b = randperm(5)
a[b] #(shuffled array)

# 'shuffle' randomly shuffles your array
# use 'shuffle!' if you want to do this in place

a = ["first", "second", "third", "fourth", "fifth"]
shuffle(a)
shuffle!(a)

##### Dates.jl ##### (https://docs.julialang.org/en/v1/stdlib/Dates/)

using Dates

# This package allows us to format dates/times. I find it most useful to measure time

first_time = now()

# Wait a few seconds before executing this line:

now() - first_time

##### Logging.jl ##### (https://docs.julialang.org/en/v1/stdlib/Logging/)

using Logging

# This package is a better way to debug your code than just using 'println' statements.
# @info, @warn, @error are all ways to display custom messages.

A = rand(3, 3);
@info "The sum of elements in the matrix is" A sum(A)

A = rand(100, 100);
@warn "The matrix is quite big" size(A)

A[10, 10] = 1000000;
@error "The matrix has large values" maximum(A)

##### Statistics.jl ##### (https://docs.julialang.org/en/v1/stdlib/Statistics/)

using Statistics

# This package provides basic stats tools
# The functions below compute the mean, media, standard deviation, and variance of a collection of numbers.

a = randn(100);
mean(a)
median(a)
std(a)
var(a)

# 'cov' returns the covariance between two vectors

b = a + randn(100)*10;
cov(a, b)

# 'cor' returns the Pearson correlation between two vectors

b = a + 0.1*rand(100);
cor(a, b)

# 'quantile' calculates the quantiles of a set of numbers at given percentiles.
# The code below calculates the 2.5% and 97.5% quantiles.

quantile(a, [0.025, 0.975])

##### LinearAlgebra.jl ##### (https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/)

using LinearAlgebra

# This package provides useful linear algebra operators
# 'tr', 'det' and 'inv' implement the trace, determinant, and inverse of a matrix

A = rand(1:3, 3, 3)
tr(A)
det(A)
inv(A)

# To get the transpose, use an apostrophe (A') or 'transpose':

Aᵀ = A'
transpose(A)

# To use basic operators element-wise, use the broadcasting operator '.'

sin.(A)
cos.(A)
log.(A)
A .^ 2
A ./ 2

# The matrix product is obtained simply with '*'

A * A != A .* A

# The eigenvalues and eigenvectors can be obtained with 'eigvals' and 'eigvecs'

eigvals(A)
eigvecs(A)

# The svd decomposition can be obtained using 'svd'

U, Λ, Vᵀ = svd(A)

# 'pinv' computes the pseudo-inverse

B = rand(1:3, 2, 5)
pinv(B)

# 'dot' computes the dot product between two vectors (can also use \cdot):

a = [1, 2, 3];
b = [-2, -4, -6];
dot(a, b)
a ⋅ b

# 'cross' computes the cross product between two vectors:

cross(a, b)

# 'diagm' turns a vector into a matrix with that vector on its diagonal:

diagm(a)

# 'diag' returns the diagonal of a matrix as a vector:

diag(A)


# The variable 'I' is usually reserved for the identity matrix, which in Julia is called 'UniformScaling'

I
I*A == A*I == A

# To take the square root of a matrix, we can use 'sqrt', or 'cholesky'.

A = a * a'
sqrt(A)
cholesky(A)

# You can only take the cholesky decomposition if your matrix is positive definite.
# To check for this, use 'isposdef'

isposdef(A)
A = (a * a') + 2I
isposdef(A)
cholesky(A)

# Similarly, 'issymmetric' tests whether the matrix is symmetric

issymmetric(A)

# The rank of a matrix is unsurprisingly computed using 'rank':

rank(A)

# The p-norm for a vector/matrix is computed using 'norm':

norm(A)
norm(a)
norm(a, 1) #(this calculates the 1-norm)

# To normalize a vector, either use 'normalize':

normalize(a)

# Get the nullspace of a matrix with 'nullspace':

nullspace(B)

########## CODING TOOLS ##########

##### ProgressBars.jl ##### (https://github.com/cloud-oak/ProgressBars.jl)
# If this gives you an error, make sure to install the package first!
# Go into package mode by typing ']' into the terminal, and then run 'add ProgressBars'
# Backspace to go out of package mode, alternatively run the line below

using Pkg; Pkg.add("ProgressBars");
using ProgressBars 

# This works like tqdm in Python, and shows the progress of loops

a = 0
for i in ProgressBar(1:10_000_000)
    a += 1
end

a = 0
for i in tqdm(1:10_000_000) # also works using tqdm if you're tied to habits
    a += 1
end

# If you wish to set a custom description:

iter = ProgressBar(1:100);
for i in iter
    # ... Neural Network Training Code that takes forever
    sleep(0.05)
    loss = exp(-i/100)
    accuracy = sum(110*rand(100) .< i)
    set_description(iter, "Loss: $(round(loss, sigdigits=2)), Accuracy: $(round(accuracy))%")
end

# Sometimes you just want to display a progress bar without wrapping an iterable.

pbar = ProgressBar(total=100);
update(pbar);
update(pbar, 49);

# In Jupyter notebooks, use ProgressMeter.jl instead (https://github.com/timholy/ProgressMeter.jl)

Pkg.add("ProgressMeter");
using ProgressMeter
a = 0
@showprogress for i in 1:10_000_000
    a += 1
end

##### BenchMarkTools.jl ##### (https://juliaci.github.io/BenchmarkTools.jl/stable/)

Pkg.add("BenchmarkTools");
using BenchmarkTools

# This package lets you find out how fast your functions run.
# use the '@btime' macro before your function, and the time taken, as well as memory allocations will be displayed
# @btime will run your function many times to get a good estimate, so it will take a while depending on your function

"""
    fib_memo(n)

Compute n'th Fibonacci number using memoization
"""
function fib_memo(n)
    known = zeros(BigInt, n)
    function memoize(k)
        if known[k] != 0
            # do nothing
        elseif k == 1 || k == 2
            known[k] = 1
        else
            known[k] = memoize(k-1) + memoize(k-2)
        end
        return known[k]
    end
    return memoize(n)
end

fib_memo(1000) # These numbers get big
@btime fib_memo(1000); # It takes Julia mere microseconds to compute even huge numbers like these!

# You can use @benchmark to get some more info.

@benchmark fib_memo(1000)

# If you have more expensive functions, you can simply use @time or @elapsed if you want to store the value
# These estimates are not as trust-worthy as the ones from @btime and @benchmark

@time fib_memo(1000);
time_taken = @elapsed fib_memo(1000)

########## FILE SAVING/LOADING ##########

##### FileIO.jl ##### (https://juliaio.github.io/FileIO.jl/stable/)

Pkg.add("FileIO")
using FileIO

# This package interfaces with a number of file formats (https://juliaio.github.io/FileIO.jl/stable/registry/)
# Use 'save' and 'load' to read and write data

# To save Julia variables, 'JLD2' is the gold standard, and works similar to Python's pickle (https://juliaio.github.io/JLD2.jl/dev/)

Pkg.add("JLD2")
using JLD2

data = Dict{String, Vector{Int}}("1A3CD" => [1, 4, 2, 4, 5], "2HD7E" => [6, 4, 3, 8, 2], "F4JK3" => [0, 4, 4, 5, 5])
save("data.jld2", data)
data2 = load("data.jld2")

# If you want to load arrays saved from numpy in .npy format, use 'NPZ.jl'

Pkg.add("NPZ")
using NPZ

a = rand(100);
save("data.npy", a)
b = load("data.npy")
a == b

# For .csv files, it makes most sense to load them directly into DataFrames (we will cover these later)

Pkg.add(["CSV", "DataFrames"]);
using CSV, DataFrames

CSV.write("data.csv", data)
data2 = CSV.read("data.csv", DataFrame)
data2 == data

# To read/write .rda/.rdata files, use 'RData.jl'
    # CANT WRITE TO .rda FILES. ONLY USE THIS IF YOU HAVE SOME .rda FILE YOU WANT TO READ

    # Pkg.add("RData")
    # using RData

    # data2 = load("data.rda")

# .txt files can be read directly using 'readlines'
# If it contains data in numericals, use 'parse' followed by either Int64 or Float64

a = readlines("data.txt")

##### MAT.jl ##### (https://juliaio.github.io/MAT.jl/stable/methods/)
# To read/write .mat files, use 'MAT.jl' (doesn't interface with FileIO)

Pkg.add("MAT")
using MAT

matwrite("data.mat", Dict("A" => rand(10, 10), "a" => randn(20)); version="v7.3") #(version specifies MATLAB version)
vars = matread("data.mat")
size(vars["A"])

##### JSON.jl ##### (https://github.com/JuliaIO/JSON.jl)
# To read/write .json files, use 'JSON.jl' (doesn't interface with FileIO)

Pkg.add("JSON")
using JSON

# You first have to convert your dict to a json string using 'JSON.json'

json_string = JSON.json(data);

# Now we can use the general file opening functionality that we have been avoiding
# by using FileIO

open("data.json","w") do f 
    write(f, json_string) 
end

# 'JSON.parsefile' reads a .json file

data2 = JSON.parsefile("data.json")
data2 == data

### EXERCISE 1

########## PLOTTING ##########

# There are a number of plotting packages
# - Plots.jl - The default, most documented package, what we'll be using - (https://docs.juliaplots.org/stable/)
# - Makie.jl - A newer alternative that has a lot to offer - (https://docs.makie.org/stable/)
# - Plotly.jl - Originally for JavaScript plotting - (https://plotly.com/julia/getting-started/)
# - Gladfly.jl - Most like seaborn and ggplot - (https://gadflyjl.org/stable/)

Pkg.add("Plots")
using Plots

##### BASIC PLOTTING #####

# Plots.jl does almost all of its work through keyword arguments on the base 'plot' function
# To plot a single series of data, simply put the data into 'plot'
# The first time you run this, it might take a while...

x = collect(0:0.1:10);
y = x + randn(length(x));
plot(y)

# You can also specify which x points to plot at
# I personally prefer the ticks on the outside

plot(x, y, tick_direction=:out)

# Specify the size of your plot with 'size' (by default in pt measure, width first then height)
# I generally don't like the grid and the label for a single series. You can get rid of these using the 'label' and 'grid' keyword arguments
# You can specify axis labels and titles using 'xlabel', 'ylabel' and 'title'
# x and y ticks can be specified using the 'xticks' and 'yticks' arguments

plot(x, y, size=(200, 200), label="", grid=false, xlabel="X axis", ylabel="Y axis", title="Some plot", xticks=[0, 5, 10], yticks=[0, 5, 10])

# As you can tell, this plot line is getting rather long.
# This is nice since it is all contained in one command, but can get arduous to look at.
# If you prefer, you can split up the function like so:

plot(x, y, 
    xticks=[0, 5, 10], yticks=[0, 5, 10], 
    size=(200, 200), 
    xlabel="X axis", ylabel="Y axis", title="Some plot", 
    label="", grid=false)


# If you want to change what is displayed on the xticks, but not where they're shown, use:
# The limits for the axes can be specified with xlim and ylim:
# 'linewidth' sets the width of the line
# You can choose to flip the x or y axes with 'xflip' and 'yflip':
# You can also change the scale from linear to logarithmic

plot(x .+ 1, y .+ 2, # make non negative for log 
    xscale=:identity, yscale=:log10,
    xflip=true, yflip=false,
    linewidth=3,
    xlim=(-2, 15), ylim=(-5, 20),
    xticks=[0, 5, 10], yticks=[0, 5, 10], 
    size=(200, 200), 
    xlabel="X axis", ylabel="Y axis", title="Some plot", 
    label="", grid=false)


# Alternatively, you can choose whether to display the axes on the other side of the figure with 'xmirror' and 'ymirror'
# You can also rotate the ticks (in degrees) with 'xrotation' and 'yrotation'

plot(x, y, 
    xrotation=-20, yrotation=50,
    xmirror=true, ymirror=true,
    xflip=false, yflip=false,
    linewidth=3,
    xlim=(-2, 15), ylim=(-5, 20),
    xticks=[0, 5, 10], yticks=[0, 5, 10], 
    size=(200, 200), 
    xlabel="X axis", ylabel="Y axis", title="Some plot", 
    label="", grid=false)


# Sometimes, there will be unwanted whitespace below/above/left/right of your plots.
# You can manage margins using:
# (The example below is purposefully bad)

plot(x, y, 
    xflip=true, yflip=false,
    linewidth=3,
    xlim=(-2, 15), ylim=(-5, 20),
    xticks=[0, 5, 10], yticks=[0, 5, 10], 
    size=(200, 200), 
    xlabel="X axis", ylabel="Y axis", title="Some plot", 
    label="", grid=false,
    bottom_margin=-10Plots.pt, top_margin=20Plots.pt, 
    left_margin=-15Plots.pt, right_margin=20Plots.pt)

# 'color' uses Colors.jl to specify the color of the series.
# For a list of colors, check: https://juliagraphics.github.io/Colors.jl/stable/namedcolors/

plot(x, y, c=:red)

# If you want to use your own rgb color, use 'rgb' in a string 
# or with alpha value 'rgba' or with keyword 'alpha'

plot(x, y, c="rgb(100, 200, 20)")
plot(x, y, c="rgb(100, 200, 20, 0.3)")
plot(x, y, c="rgb(100, 200, 20)", alpha=0.3)

# You can change the linestyle to:

plot(x, y, linestyle=:dash, linewidth=3)
plot(x, y, linestyle=:dot, linewidth=3)
plot(x, y, linestyle=:dashdot, linewidth=3)
plot(x, y, linestyle=:dashdotdot, linewidth=3)

# You can also rotate the ticks (in degrees)

plot(x, y, xrotation=-20, yrotation=50)

# If you want to plot multiple series, you can either form a matrix
# Or you can make two plot statements
# For the latter, we use the '!' notation, which adds on top of the existing figure.

y1 = 1 .+ 2*x;
y2 = 5 .- 3*x;
plot(x, [y1 y2])

plot(x, y1)
plot!(x, y2)

# You can make the legend box transparent with:

plot(x, y1, label="first", background_color_legend=nothing, foreground_color_legend=nothing)
plot!(x, y2, label="second")

# To position the legend elsewhere, use 'legend_position'

y3 = 5 .- 0*x;
plot!(x, y3, label="third", legend_position=:bottomright)

# You can also change the layout of the legend elements by using 'legendcolumns'

x2 = 0*x;
plot!(x2, y1, label="fourth", legendcolumns=3)

# If you want to use subplots instead of having everything on the same plot,
# you can use layout to specify how many subplots there should be.
# You then specify with 'sp' on which subplot things should go.
# I usually write one 'plot' line to lay everything out, and then use a bunch of
# 'plot!' to do everything on the subplots.

plot(layout=(2, 2), xticks=[], legend=false, grid=false)
plot!(x, y1, sp=1)
plot!(x, y2, sp=2)
plot!(x, y3, sp=3)
plot!(x2, y1, sp=4)

# You can link the axes so that they are matched in scale
# (either match x axes by column, or y axes by row, or both)

plot(layout=(2, 2), link=:both, legend=false, grid=false)
plot!(x, y1, sp=1)
plot!(x, y2, sp=2)
plot!(x, y3, sp=3)
plot!(x2, y1, sp=4)

# You can also specify how big each subplot is

plot(layout=grid(2, 1, heights=[0.2, 0.8]), link=:both, legend=false, grid=false)
plot!(x, y1, sp=1)
plot!(x, y2, sp=2)

# More advanced layouts are also possible

l = @layout [a{0.3w} [grid(3,3)
             b{0.2h}  ]]
plot(rand(10, 11), layout=l, legend=false, seriestype=[:bar :scatter :path])

### EXERCISE 2

##### ALTERNATE PLOTS #####

# To make a scatter plot, you can use either:
plot(x, y, seriestype=:scatter)
scatter(x, y)

# You can change the marker shape
# Choose from [:none, :auto, :circle, :rect, :star5, :diamond, :hexagon, :cross, :xcross, :utriangle, :dtriangle, :rtriangle, :ltriangle, :pentagon, :heptagon, :octagon, :star4, :star6, :star7, :star8, :vline, :hline, :+, :x]

scatter(x, y, markershape=:rect)
scatter(x, y, markershape=:diamond)
scatter(x, y, markershape=:star5)

# Change the size, color, and border:

scatter(x, y, markersize=10, markercolor=:purple, markerstrokecolor=:red)

# To make a histogram, simply use 'histogram'

x = [0.3*randn(1000) .- 2..., 0.3*randn(1000) .+ 2...];
histogram(x)

# This histogram is too coarse, use bins to get finer

histogram(x, bins=100)

# If you want to normalize the histogram

histogram(x, bins=100, normalize=:probability)

# Histograms can also be made in two dimensions

x = randn(10000);
y = randn(10000);
histogram2d(x, y, nbins=20, normalize=:probability)

# Similarly, we can define a heatmap

data = randn(100, 100);
heatmap(1:100, 1:100, cov(data), colorbar_title="\nColorbar title", right_margin=15Plots.pt)

# The colormap or theme can be set by choosing from https://docs.juliaplots.org/latest/generated/colorschemes/
# I also often prefer to have (0, 0) at the top left, so I flip the y axis.
# The title of the colorbar can be set with 'colorbar_title'

heatmap(1:100, 1:100, cov(data), cmap=:grays, yflip=true, xticks=[], yticks=[], colorbar_title="\nColorbar title", right_margin=15Plots.pt)

# Making a bar plot

data = Dict{String, Vector{Int}}("1A3CD" => [1, 4, 2, 4, 5], "2HD7E" => [6, 4, 3, 8, 2], "F4JK3" => [0, 4, 4, 5, 5]);
plot(collect(keys(data)), mean.(collect(values(data))), seriestype=:bar)
bar(collect(keys(data)), mean.(collect(values(data))))

# You can change the width of the bars

bar(collect(keys(data)), mean.(collect(values(data))), bar_width=0.4)

# You can make the bars horizontal too (by switching the x and y axis)

bar(collect(keys(data)), mean.(collect(values(data))), permute=(:x, :y), xlim=(0, 3))

# To add horizontal/vertical lines to your plot, 'vline' and 'hline' make it easy:

vline!([1, 2, 3], linewidth=3)
hline!([1, 2])

# Pie charts

x = ["Nerds", "Hackers", "Scientists"];
y = [0.4, 0.35, 0.25];
pie(x, y, title = "The Julia Community")

##### TEXT ON PLOTS #####

# You can specify font attributes like:
# family: "serif" or "sans-serif" or "monospace"
# pointsize: Size of font in points
# halign: Horizontal alignment (:hcenter, :left, or :right)
# valign: Vertical alignment (:vcenter, :top, or :bottom)
# rotation: Angle of rotation for text in degrees (use a non-integer type)
# color

x = 1:10;
y = x;
plot(x, y, xlabel="X label", ylabel="Y label", xguidefont=font("Helvetica", 22, :red), yguidefont=font("Arial", 12, :blue))

# You can place text anywhere on the plot using 'annotate!'

plot(x, y)
annotate!(4, 6, "a note")

# If you want to use LaTeX symbols as part of your plots, import LaTeXStrings (https://github.com/JuliaStrings/LaTeXStrings.jl)
# Wrap your latex code into a string, and place it inside the 'latexstring' function
# If you are using any symbols, make sure to use DOUBLE backslack \\

Pkg.add("LaTeXStrings")
using LaTeXStrings

plot(x, y, xlabel="\n"*latexstring("\\sqrt{\\frac{\\alpha}{\\beta}}"), ylabel=latexstring("\\mathcal{N}"), size=(300, 300), guidefontsize=16)

##### SAVING PLOTS #####

# You can save your plot as .png, .svg or .pdf
# Use the 'savefig' function to do so. 
# Make sure you set the dpi high (300) to get high quality plots. 
# If you want to save your text as text rather than shapes for .svg
# so you can edit in illustrator, make sure you set the fontfamily of all your
# fonts to 'Helvetica'

plot(1:10, exp.(1:10), dpi=300, label="", grid=false, yscale=:log)
savefig("example_plot.png")

##### StatsPlots.jl #### (https://docs.juliaplots.org/latest/generated/statsplots/)

Pkg.add("StatsPlots")
using StatsPlots

# StatsPlots adds a few extra plots that are nice to have
# For example, we can get the marginals of our 2D histogram with:
# (I like to pick colormaps that start at white so that they blend in with the background)

x = randn(1000);
y = randn(1000);
marginalhist(x, y, nbins=20, cmap=:Blues)

# Similarly, we can also plot the contours using kernel density estimates
# (levels sets how many contour levels to plot, code can take a second to run)

marginalkde(x, y, levels=10, cmap=:Blues)

# If you have multiple variables and want to see their correlation, 
# you can use cornerplot.

data = randn(1000, 4); #(imagine we have 4 measure, 1000 samples each)
data[:, 4] .= -data[:, 2] .+ 2*randn(1000); #(let's assume the second and fourth measure are weakly and negatively correlated)
data[:, 3] .= data[:, 1] .+ 0.5*randn(1000); #(let's assume the first and third measure are strongly correlated)
cornerplot(data, size=(500, 500), normalize=:probability)

# Andrews plot is a way to look for structure in high dimensional data.
# We make our data such that there are three patters in it.
# You can see these in the Andrews plot, as well as their variability

data = randn(500, 4);
data[1:100, :] .= hcat([[1, 2, 3, 4] .+ 0.1*randn(4) for _ in 1:100]...)'
data[101:400, :] .= hcat([[7, 6, 5, 4] .+ randn(4) for _ in 1:300]...)'
data[401:end, :] .= hcat([[10, 10, 0, 10] .+ 0.5*randn(4) for _ in 1:100]...)'
andrewsplot(data, alpha=0.1, label="", grid=false)

# qqplot is a good way to check if data is Normally distributed
# If you want to compare a distribution to the normal, use qqnorm

x = randn(100);
y = randexp(100);
qqplot(x, y)
qqnorm(x)

# covellipse lets you plot the covariance as an ellipse
# Multiply covariance by 3^2 to capture 3sd worth of data

data1 = randn(1000, 2);
data2 = [5 .+ randn(1000) 3 .+ randn(1000)];
data3 = [5 .+ randn(1000) randn(1000)];
data3[:, 2] = -data3[:, 1] + randn(1000);
scatter(data1[:, 1], data1[:, 2], alpha=0.1, c=:blue)
scatter!(data2[:, 1], data2[:, 2], alpha=0.1, c=:red)
scatter!(data3[:, 1], data3[:, 2], alpha=0.1, c=:green)
covellipse!(mean(data1, dims=1)[:], cov(data1).*9, c=:blue)
covellipse!(mean(data2, dims=1)[:], cov(data2).*9, c=:red)
covellipse!(mean(data3, dims=1)[:], cov(data3).*9, c=:green)

# To plot raw data across subjects, we can create
# violin, box and dot plots as follows:

subjects = ["1A3CD" "2HD7E" "F4JK3"];
data = [4*randn(100) -1 .+ randn(100) 2 .+ 0.5*randn(100)];
violin(subjects, data, alpha=0.5)
boxplot!(subjects, data, alpha=0.5)
dotplot!(subjects, data, alpha=0.5)

##### ANIMATIONS #####

# We can make animations very easily by using the '@animate' macro around a for loop that makes a plot every frame:
# 'gif' then turns the animation into a '.gif' and sets the frames-per-second ('fps')

n = 100;

anim = @animate for i ∈ 1:n
    x = sin.(i/n*2π)
    y = cos.(i/n*2π)
    scatter([x], [y], xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), size=(300, 300), label=nothing, grid=false)
end;
gif(anim, fps = 30)

# Below is a more advanced animation:

function julia_icon_x(t; dir=:clockwise)
    x = 0
    if dir == :clockwise
        if t < 0.333
            x = sin.(t*2π*3*5/6 - 5π/6)
        elseif t < 0.666
            x = 1 .+ sin.((-(t-0.333)*2π*3*7/6) - π/6)
        else
            x = -1 .+ sin.((t-0.666)*2π*3*5/6 + π/2)
        end
    else
        if t < 0.333
            x = sin.(-t*2π*3*7/6 - 5π/6)
        elseif t < 0.666
            x = 1 .+ sin.(((t-0.333)*2π*3*5/6) - π/6)
        else
            x = -1 .+ sin.(-(t-0.666)*2π*3*7/6 + π/2)
        end
    end
    return x
end

function julia_icon_y(t; dir=:clockwise)
    y = 0
    if dir == :clockwise
        if t < 0.333
            y = 1 .+ cos.(t*2π*3*5/6 - 5π/6)
        elseif t < 0.666
            y = -(√3-1) .+ cos.((-(t-0.333)*2π*3*7/6) - π/6)
        else
            y = -(√3-1) .+ cos.((t-0.666)*2π*3*5/6 + π/2)
        end
    else
        if t < 0.333
            y = 1 .+ cos.(-t*2π*3*7/6 - 5π/6)
        elseif t < 0.666
            y = -(√3-1) .+ cos.(((t-0.333)*2π*3*5/6) - π/6)
        else
            y = -(√3-1) .+ cos.(-(t-0.666)*2π*3*7/6 + π/2)
        end
    end
    return y
end

n = 400;
T = 15;
colors = [[Colors.JULIA_LOGO_COLORS.green  for _ in 1:33]..., 
    [Colors.JULIA_LOGO_COLORS.red for _ in 1:33]..., 
    [Colors.JULIA_LOGO_COLORS.purple for _ in 1:34]...];


anim = @animate  for i ∈ 1:n
    plot()
    for k in 1:3
        r = collect(maximum([0, i-T]):i) .+ 33*k
        x, y, c = zeros(length(r)), zeros(length(r)), zeros(RGB, length(r))
        for j in eachindex(r)
            dir = :clockwise
            ii = r[j]
            if 100 <= ii < 200 || 300 <= ii < 400
                dir = :anticlockwise
            end
            ii = (ii % 100) + 1
            x[j] = 0.5*julia_icon_x.(ii / (n/4), dir=dir)
            y[j] = 0.5*julia_icon_y.(ii / (n/4), dir=dir)
            c[j] = colors[ii]
        end
        linewidths = 4*reverse(1 .- collect(1:length(r)) ./ (T))
        plot!(x, y, xlim=(-2.2, 2.2), ylim=(-2.2, 2.2), label="", size=(300, 300), c=c, grid=false, axis=([], false), axes=false, linewidth=linewidths)
    end
end
gif(anim, fps=50)

### EXERCISE 3


