# Welcome to the Julia basics tutorial!

# Write comments just as in python with the '#' symbol

########## GETTING STARTED IN VS CODE ##########

# Press [SHIFT] + [ENTER] to execute a line in VS code
# (the first time you do this, Julia will start up which takes a few seconds)
# try on the line below!

print("Hello world")

# You should see a tick mark next to the function, which means it has executed without error
# You should also see 'Hello world' printed in the terminal, at the bottom of your window
# In this window, you can write any piece of Julia code 
# This usually just one-liners, we leave more complex code for the script
# Try running the same line as above in the terminal

########## PACKAGES AND MODES ##########

# Packages are central to any piece of Julia code (just as any language)
# They allow us to use other people's work, specializing what we want our code to do
# Packages can either be from the standard library which comes with your Julia installation,
# or they are installed from the internet.
# To load a package, run the following line:

using LinearAlgebra

# This is one of the standard library packages, so you don't need to do any installations
# For the next package, we will need to download it from the internet
# There are two ways to download a package. 
# The first is by using the 'Pkg' package:

using Pkg
Pkg.add("BenchmarkTools")

# The second (and in my opinion easier) way to install a package is by using the terminal
# Press the right bracket ']' button while you are in the terminal, and it should turn from green to blue.
# You are now in "package mode". From here, it is very easy to install, update, or remove packages.
# Simply type 'add Benchmarktools' while in package mode to install the package.
# To exit the package mode, simply press the backspace button on your keyboard.

# If you press the question mark '?' button, you will enter "help mode"
# Similarly to MATLAB, this model gives details about any function you might wish to use.
# Try type 'svd' and press enter. You should get some information about how the svd function works.

# Next, try enter package mode, and type '?' followed by enter. 
# This will give you an overview of all the functions you can run.
# For example, it shows that you can use 'status' to check all your installed packages.

# The final mode is shell mode. Enter this mode by pressing the semi-colon ';' button
# From here, you can write anything you would in the terminal such as 'ls', 'cd' etc

########## VARIABLES ##########

# As we saw earlier, Julia is dynamically typed, 
# which means we don't have to specify a type for the variables we use.
a = 1
b = "two"
c = [3]

# You will see that after executing each of these lines, the value of the variable will be displayed on the right.
# To stop this from happening, put a semi-colon ';' after the line
a = 2;
b = "three";
c = [1];

# If you don't want to run each line individually, you can use the 'begin' 'end' functionality
begin
    a = 3
    b = "one"
    c = [2]
end;

# This is purely aesthetic, but you can use any unicode character from https://docs.julialang.org/en/v1/manual/unicode-input/
# To use these, type the common name such as 'alpha' after a backslash, and press [TAB] ('\alpha' + [TAB])
α = "\alpha";
😎 = "\:sunglasses:";
ℵ = "\aleph";

# You can also use this technique to do sub/super scripts.
# To do this, type '\_1' or '\^2'. Note: you can't sub/super script anything
σ² = "\sigma [TAB] \^2 [TAB]";
Xₜ = "X [TAB] \_t [TAB]";
αᵅ = "\alpha [TAB] \^alpha [TAB]";

########## TYPES ##########

# While you don't have to worry/care about types if you wish, 
# it still help in understanding, potentially speeding up, and safe-guarding your code
# You can use the function 'typeof' to check the type of any variable
typeof(a)
typeof(b)
typeof(c)

# To check whether a variable is of a given type, you can use 'isa'
a isa Int
a isa Int64

# Wait, what just happened? 'a' is both an Int and an Int64? What does that mean?
# In Julia, there are two categories of types. Abstract types and Concrete types.
# Concrete types are what you get from 'typeof', and they are the most specific description of your variable.
# Abstract types are more general, and can be used to check broader categories.
# For example, whether your variable is a Float64 or an Int64, they are both numbers.
# The type 'Number' is an Abstract type that encompasses all numbers, for example
1 isa Number
-2.5 isa Number
√π isa Number

# The most common primitive (most basic) types are:
Bool #(boolean, either true or false)
Int64 #(64 bit integer, most numbers without decimal places are Int64)
Float64 #(64 bit floating point number, most numbers with decimal places are Float64)
Char #(character, strings are made of these)
Symbol #(usually used for keyword arguments, will expand on later)

########## BASIC OPERATORS ########## (https://www.geeksforgeeks.org/operators-in-julia/)

# Here are the most common operators in julia
2 + 2
4 - 3
3 * 3
8 / 2 #(returns float)
10 ÷ 3 #('\div' is integer division, it divides and ignores the decimal points)
10 \ 3 #(inverse division, only really useful when doing linear algebra)
2 ^ 3
10 % 3 #(modulo operator, gives remainder after integer division)

# You can also change the value of a variable in place
a = 2;
a += 1; #(same as a = a + 1)
print(a)
b = 6;
b *= 5;#(same as b = b * 5)
print(b)

# Here are the most common logical operators
3 > 2
2 <= 4
4 == 4 #(checks if two values are the same)
4 != 4 #(checks if two values are different)
(3 > 2) && (2 < 4) #(logical and)
(3 > 2) || (4 != 4) #(logical or)
!(2 <= 4) #(logical not)

########## STRINGS ########## 

# Strings are useful if we want to print/display some text.

text = "This is a string"

# If you want to incorporate a variable into your string, do the following:

text = "The value of a is: $(a)"

# Adding two strings together can be done using the '*' operator

text = "A bit of text." * " And some more text"

# Use \t to insert a tab, and \n to insert a line break

text = "Once upon a\ttime...\nOops";
println(text)

# To go directly from numbers to strings:

text = string(2)
text isa String

# You can also go from text to numbers

numbers = parse(Int64, "10")
numbers isa Number

# You can split strings into arrays

split("This is a sentence.", " ")

# Or join an array into a string

join(["This", "is", "also", "a", "sentence."], "-")

########## IF-ELSE ##########

# One of the most common tools in programming are the if-else statements
# In Julia, and 'if' statement is always followed by an 'end' (like in MATLAB)
if 10 > 4
    print("not surprising")
end 

a = 5;
if a % 2 == 0
    print("a is even")
elseif a % 2 == 1
    print("a is odd")
else
    print("something went wrong...")
end

# A very useful tool to compact things is the ternary operator which uses the question mark '?'
# The way this operator works is as follows: you first provide a statement which could be true or false
# Then you write the questoin mark, and then you write what follows if the statement is true.
# After this, you write a colon ':', and then comes what has to happen is the statement is false.

a = 5;
a % 2 == 0 ? print("a is even") : print("a is odd")

# You can also chain these together

a = 5;
a % 2 == 0 ? print("a is even") : a % 2 == 1 ? print("a is odd") : print("something went wrong...")

########## LOOPS ##########

# Another very common sight in programming are loops. These let you execute bits of code multiple times
# I will cover for loops, but not while loops, since I don't find them very useful (but they exist in Julia)
# If we want to iterate over 10 numbers, we first create the range of numbers as in MATLAB by 1:10

for i in 1:10
    print(i)
end

# Why did it print everything on one line? That's because most of the time we actually want to use 'println' and not 'print'

for i in 1:10
    println(i)
end

# If we want to loop over every other number, we can use the following notation

for i in 1:2:10
    println(i)
end

# We can also achieve this by using the 'continue' functionn

for i in 1:10
    if i % 2 == 0 # if i is even, skip and continue to next iteration
        continue
    end
    println(i)
end

# If you want to break the loop early, because you achieve what you wanted, you can use the break function

for i in 1:100000
    if i ^ 2 > 100 # stop the loop after the square of i is bigger than 100
        break
    end
    println(i)
end

# Similarly to the ternary operator '?', we can do loops in a single line (same as Python's list comprehension)
# Note that here, the results will go into an array (which we will cover next)
a = [i for i in 1:10]

### EXERCISE 1

########## BASIC DATA STRUCTURES ##########

# We have previously discussed primitive types, but we can do more interesting stuff with them.
# For example, once you have a number, you may be interested in forming a collection of numbers
# In Julia, there is a number of ways to store these numbers, the simplest of them being the array
# Like in MATLAB (and unlike Python), 1D arrays are vectors and 2D arrays are matrices
# Unline MATLAB (and like Python), it is very easy to add onto these arrays

a = [5, 4, 3, 2, 1];
typeof(a)

# We can find the length of the array with 'length'

length(a)

# The sum and product of elements in an array can be obtained with 'sum' and 'prod'

sum(a)
prod(a)

# To check whether the array is empty, use 'isempty'

isempty(a)

# Like with the range 1:10, we can loop over the contents of an array using 'in' (or ∈ = '\in')

for element in a
    println(element)
end

# You can use 'in' (or '∉' '\notin') to check if an array contains (or does not contain) an item

2 ∈ a
3 ∉ a

# You can use the 'enumerate' function to keep track of the index as well as the value

for (index, element) in enumerate(a)
    println("$(index) - $(element)")
end

# To add onto an array, use the 'push!' function

push!(a, 0)

# This function works in place, and so no assignment using '=' is needed
# It adds the number to the end of the array. To remove the element at the end of the array, use 'pop!'

popped_number = pop!(a)
print(a)

# If you want to add/remove numbers to the front of the array, use 'pushfirst!' and 'popfirst!'

pushfirst!(a, 6)
popped_number = popfirst!(a)

# Like MATLAB (and unline Python), Julia uses 1 indexing. (Mathematical consistency, Readability and clarity, Reduced potential for off-by-one errors)
# This means that to get the first element of an array you do:

a[1]

# And to get the last index, you do:

a[end]
a[end-1] #(second to last index)

# To get the array in reverse order, use 'reverse':

reverse(a)

# EXTRA DETAIL
# This is not used very often, but to 'unpack' your array, you can use '...'

println(a...)

# If you want to initialize a Vector, you can use the 'zeros' or 'ones' functions

zeros(5)
ones(5)

# For higher order arrays, add more arguments

zeros(2, 2)
ones(5, 2)

# For higher dimensional arrays, use 'size' instead of 'length':

a = zeros(10, 3)
size(a)

# If you want to initialize a Vector as being empty, I recommend setting the type of the Vector as follows:

a = Vector{Int64}() #(this specifies what goes into your vector)
a = Int64[] #(this is an alternative way of doing it)
a = Vector{Vector{Int64}}() #(this is a vector of vectors, not a matrix!)
a = [] #(this is an empty vector with type 'Any' that goes in it, which means it can contain anything)

# Use MATLAB notation to create matrices

A = [1 2; 3 4]

# You can use this to create block matrices too

B = [A A; A A]

# You can also do this with vectors

a = [1, 2, 3]
A = [a a a]

# The 'hcat' and 'vcat' operators do the same thing:

hcat(a, a)
vcat(a, a)

# Arrays are mutable, which means that the data inside of them can be changed.
# This is often useful, but leads to memory and speed inefficiencies.
# The other way to store a series of number are tuples.
# Tuples are immutable meaning their content cannot be changed.

a = (1, 2, 3)
typeof(a)
a[end]

# The third basic data structure is the dictionary. This is implemented as a hash map
# Dictionaries are useful if you want to index your data structure arbitrarily.
# For example, to store the data for different subjects, you may want to use a dictionary
# since you can use the subject IDs to access the data

data = Dict{String, Vector{Int}}("1A3CD" => [1, 4, 2, 4], "2HD7E" => [6, 4, 3, 8, 2]) # This is one way to add elements to the dictionary
data["F4JK3"] = [0, 4, 4, 5, 5] # This is another way
data["1A3CD"]

# You can use the functions 'keys' and 'values' to access the set of keys and values of the dictionary.

keys(data)
values(data)

# This means we can loop over the data like so:

for subj in keys(data)
    println(length(data[subj])) # print length of data from each subject
end

# To convert these into arrays, we need to use the 'collect' function.

k = collect(keys(data))
v = collect(values(data))

### EXERCISE 2

########## FUNCTIONS ##########

# Julia is a funcional programming language. What that means is that everything is done through functions.
# As we will see later, there are objects called 'structs' that have some similarities to classes in python,
# but Julia is not an object oriented programming language.
# To define your own function, use the 'function' keyword. Use 'return' to indicate what the function should return when called

function sum_across_subjects(data)
    result = 0
    for subj in keys(data)
        result += sum(data[subj])
    end
    return result
end
sum_across_subjects(data)

# You can add optional arguments like so

function sum_across_subjects(data; exponent=1)
    result = 0
    for subj in keys(data)
        result += sum(data[subj])^exponent
    end
    return result
end
sum_across_subjects(data; exponent=2)

# If you want to bullet-proof your code, assigning a type to the input and outputs of your function
# makes sure that it takes in and spits out things in the correct format. 
# We can assign a type to the inputs using the double colon notation '::' followed by the type
# To assign a type to the output, use the double colon after the name of the function

function sum_across_subjects(data::Dict{String, Vector{Int}}; exponent::Int=1)::Int
    result = 0
    for subj in keys(data)
        result += sum(data[subj])^exponent
    end
    return result
end
sum_across_subjects(data; exponent=2) # correct inputs
sum_across_subjects(data; exponent="s") # incorrect inputs

# Commenting your code is always a good idea. 
# But for functions, there's an extra bit you can do to improve your code.
# docstrings: explanations about how your function works, can be implemented using triple quotations """
# The format in Julia is the have the docstring right above your function, and as first line, have the name of
# the function with some inputs with four space before, followed by a one-line explanation of what the function does in present tense.

"""
    sum_across_subjects(data; exponent=1)

Sum data for each subject and raise to exponent.
"""
function sum_across_subjects(data::Dict{String, Vector{Int}}; exponent::Int=1)::Int
    result = 0
    for subj in keys(data)
        result += sum(data[subj])^exponent
    end
    return result
end

# Now try and go into 'help mode' by pressing '?' in the terminal and then typing the function's name sum_across_subjects
# You should see that the docstring is now displayed.

# One neat feature of julia is that you can broadcast your function over arrays.
# What that means is that we can have a function (e.g. sum) and apply it to ever element of an array.
# To do this, write a dot/period after the function and before the parentheses 'sum.(array)'
# Lets collect all the values of our data dict, and broadcast the sum operator over it.

v = collect(values(data))
a = sum.(v)
sum(a) # this now returns the same as our sum_across_subjects function!

# The broadcast operator is also used to operate vectors element wise

a = [3, 2, 1]
b = [4, 5, 6]
a * b

# This code returns an error because it does not know how to multiply two row vectors together
# The line below broadcasts the multiply operator over each element of the vector.

a .* b

# Some functions in Julia have an exclamation mark in them (like push! and pop! that we used earlier).
# The exclamation mark is a convention in Julia to indicate that these functions work in place.
# What this means is that these functions don't return anything: they modify their arguments.
# Here is an example below (note that the type of return is 'Nothing')

"""
    square_subject_data!(data)

Square data from each subject.
"""
function square_subject_data!(data::Dict{String, Vector{Int}})::Nothing
    for subj in keys(data)
        data[subj] = data[subj] .^ 2
    end
    return nothing
end
square_subject_data!(data)
println(data)

# Similar to Python's lambda functions, Julia also has 'anynonymous functions' (aka one line functions)
# These functions use arrow notation. A simple example is shown below:

square = x -> x^2
square(2)
power_power = (x, y) -> 2^x^y
power_power(3, 3)

# These anonymous functions can be useful in a number of cases. The most obvious is to
# find the position of an element in an array. This uses the 'findfirst' or 'findall' functions
# These functions take a 'check' function as first argument to tell them what to look for.
# This check is a statement that is applied to all elements, and those that satisfy it are returned

a = [0, 0, 1, 4, 3, 2, 4, 2]
check = x -> x == 2 # this checks whether 'x' is equal to 2 or not
findfirst(check, a) # returns the index of the first element where check is satisfied
findall(check, a) # returns all indices where check is satisfied

# If we only want the values, but not the indices, we can use the broadcasting operator.
# For example, in the following array, imagine we only want the lists with length greater than 2.

a = [[1, 2, 3], [3, 3, 5, 6, 2], [2], [4, 2], [5, 2], [2], [2, 4, 6, 7]]

# We fist generate an array with a 1 if the element satisfied our demand, and 0 otherwise

b = length.(a) .> 2

# Then we can index our original array with this 'BitVector' to get the same result
a[b]

### EXERCISE 3

########## STRUCTS ##########

# As I mentioned earlier, Julia is a functional programming language. 
# If creating your own type (which you can do but we won't get into) is not enough, or you want a very custom data structure
# then structs may be useful. By default, structs are immutable, but they can be made mutable.

struct Car
    license_plate::String
    brand::String
    year_built::Int
    owners::Vector{String}
end
my_car = Car("1BC6TF", "Audi", 2017, ["Judy", "Bob"])
my_car.year_built
my_car.license_plate = "1BC6TG"
pop!(my_car.owners)

mutable struct MCar
    license_plate::String
    brand::String
    year_built::Int
    owners::Vector{String}
end
my_car = MCar("1BC6TF", "Audi", 2017, ["Judy", "Bob"])
my_car.year_built
my_car.license_plate = "1BC6TG"
pop!(my_car.owners)

# You can view everything about a struct using 'dump'

dump(Car)
dump(my_car)





