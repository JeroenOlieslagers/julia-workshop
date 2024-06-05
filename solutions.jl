### EXERCISE 1

# Write a loop that counts from 1 to 100, printing each number on a separate line. But, there's a twist:
# - For numbers that are multiples of 3, you should print "Fizz" instead.
# - For multiples of 5, print "Buzz".
# - For numbers that are multiples of both 3 and 5, don't print anything.

##### TODO: CODE BELOW

for i in 1:100
    if i % 3 == 0
        println("Fizz")
    elseif i % 5 == 0
        println("Buzz")
    elseif (i % 3 == 0) && (i % 5 == 0)
        continue
    else
        println(i)
    end
end

##### TODO: CODE ABOVE

### EXERCISE 2

# Write a piece of code that for the given dictionary, creates a separate dictionary
# where the values in the arrays repeated. For example:
# data["FG34A"] -> [1, 2, 3, 4] would turn into
# new_data["FG34A"] -> [1, 2, 3, 4, 1, 2, 3, 4]

# Use the dictionary below as original
data = Dict{String, Vector{Int}}("1A3CD" => [1, 4, 2, 4], "2HD7E" => [6, 4, 3, 8, 2], "F4JK3" => [0, 4, 4, 5, 5])

##### TODO: CODE BELOW

dict = Dict{String, Vector{Int}}()
for (k, v) in data
    dict[k] = vcat(v, v)
end

##### TODO: CODE ABOVE

### EXERCISE 3

# Write a TYPED function that takes in an Vector of Int64, and returns Nothing
# The in-place function should square each element of the vector
# (bonus points is you use the broadcasting operator)

##### TODO: CODE BELOW

function square!(x::Vector{Int64})::Nothing
    x .^= 2
    return nothing
end

##### TODO: CODE ABOVE

### EXERCISE 1

# Write a piece of code that generates a random 100x100 matrix (using 'randn'), takes the pseudo-inverse of this matrix, 
# and concatenates 500 of these together into a single 50,000 x 100 matrix.
# Incorporate ProgressBar/ProgressMeter to indicate your progress across the 100 matrices.

##### TODO: CODE BELOW

using LinearAlgebra, ProgressBars

a = randn(100, 100);
b = pinv(a);
A = b;
for i in ProgressBar(1:499)
    a = randn(100, 100)
    b = pinv(a)
    A = [A; b]
end

##### TODO: CODE ABOVE

### EXERCISE 2

# Create any line plot of your liking (single plot, no subplots)
# It must meet these requirements:
# - 4 lines
#   - colors: green, blue, gray, red
#   - line width: thick, thick, thin, thin
#   - labeled (however you want)
# - no legend box
# - at most 3 x ticks, and at least 5 y ticks
# - no grid lines
# - must have axis labels and a title
# - plot should be square in shape

##### TODO: CODE BELOW

using Plots

x = -1:0.01:1;
y1 = x .^ 1;
y2 = x .^ 2;
y3 = x .^ 3;
y4 = x .^ 4;
plot(background_color_legend=nothing, foreground_color_legend=nothing, xticks=[-1, 0, 1], yticks=[-1, -0.5, 0, 0.5, 1], grid=false, title="Polynomials", xlabel="x", ylabel="y", size=(400, 400))
plot!(x, y1, c=:green, linewidth=4, label="linear")
plot!(x, y2, c=:blue, linewidth=4, label="square")
plot!(x, y3, c=:gray, linewidth=1, label="cube")
plot!(x, y4, c=:red, linewidth=1, label="quartic")

##### TODO: CODE ABOVE

### EXERCISE 3

# Create an animation with following requirements
# - fixed axis scales (both x and y do not resize during the animation)
# - uses the given data and plots a line plot of these
# - the line should extend from the left (x=0) to the right (x=120)
# - each frame should add a single point to the line
# - fps should be 30
# BONUS: make the rightmost point a large red star on each frame

# Data for plotting is below:
data = cumsum(randn(120));

##### TODO: CODE BELOW

anim = @animate for i ∈ 1:120
    plot(1:i, data[1:i], xlim=(-1.2, 130), ylim=(-12, 12), size=(300, 300), label="", grid=false)
    scatter!([i], [data[i]], markershape=:star5, c=:red, label="")
end;
gif(anim, fps = 30)

##### TODO: CODE ABOVE

### EXERCISE 1

# Provided is an array of 20 matrices of 500x500 size. Your job is to compute the trace ('tr') of
# the pseudoinverse ('pinv') of each matrix, and store it in a sepate array. Do all of this using
# multithreading

matrices = [randn(500, 500) for _ in 1:20];

##### TODO: CODE BELOW

result = zeros(20);
Threads.@threads for i in ProgressBar(eachindex(matrices))
    result[i] = tr(pinv(matrices[i]))
end

##### TODO: CODE ABOVE

# CHECK IF YOUR ANSWER IS CORRECT:

CORRECT_ANSWER = tr.(pinv.(matrices)) # might take a few seconds to run

CORRECT_ANSWER == result

### EXERCISE 2

# Identify for which subjects the mean of the data from experiment1 and experiment2
# is statistically significantly different (assume unequal variance)

using DataFrames, HypothesisTests, CSV
df = CSV.read("data_.csv", DataFrame)

##### TODO: CODE BELOW

for subj in unique(df.subject)
    df_subj = df[df.subject .== subj, :]
    x = df_subj[df_subj.experiment .== 1, :].data
    y = df_subj[df_subj.experiment .== 2, :].data
    res = UnequalVarianceTTest(x, y)
    if pvalue(res) > 0.05
        println("Subject $(subj) IS NOT significant")
    else
        println("Subject $(subj) IS significant")
    end
end

##### TODO: CODE ABOVE