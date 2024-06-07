# Welcome to part 4 of the workshop.
# Here, we will explore more specific packages used for:
# - mixed linear models
# - optimization
# - multivariate statistics
# - deep learning
# - differential equations

##### GLM.jl ##### (https://juliastats.org/GLM.jl/stable/)

using Pkg; Pkg.add(["GLM", "DataFrames", "MixedModels"]);
using GLM, DataFrames, MixedModels

# Data: reaction times for subjects after days of sleep deprivation

data = DataFrame(MixedModels.dataset(:sleepstudy));
describe(data)

# We will first fit a simple linear model on this data.
# We will regress the reaction time against the number of days of sleep deprivation (with an intercept)

fm = @formula(reaction ~ 1 + days)
model = fit(LinearModel, fm, data)

# The coefficients are obtained with 'coef' and their standard error with 'stderror'

coef(model)
stderror(model)

# To obtain everything else like p-values, use 'coeftable(model).cols'

coeftable(model)
coeftable(model).cols

# We can obtain various metrics of fitness:

r2(model)
adjr2(model)
aic(model)
bic(model)
loglikelihood(model)

# To obtain the predictions of the model to some independents, use 'predict'

predict(model, DataFrame(:days => [1, 2, 3]))

##### MixedModels.jl ##### (https://juliastats.org/MixedModels.jl/dev/)

# If we wanted to incorporate the random effect of subjects, we need to use mixed models.
# This is where MixedModels.jl comes in. The syntax is the same as before.
# The only difference is that random effects have a '|' in front of them.

fm = @formula(reaction ~ 1 + days + (1|subj))
model = fit(MixedModel, fm, data)

# To get the fixed and random effects, use either 'coef' or 'fixef' and 'ranef'

fixef(model)
ranef(model)[1][:]

# To obtain the variance of the fixed effects, an unfortunately long notation is used
# (subj and Intercept here are dependent on your formula and data)

VarCorr(model).σρ.subj.σ.var"(Intercept)"

##### Optim.jl ##### (https://julianlsolvers.github.io/Optim.jl/stable/)

Pkg.add(["Optim"]);
using Optim

# Optim.jl let's us minimize a broad range of functions.
# The main syntax is: define an objective function that takes in a vector of parameters x
# and spits out a number. Use the 'optimize' function with an initial point and potential bounds,
# and if possible, gradients, and optimize away!

f(x) = -20.0 * exp(-0.2*sqrt(0.5 * (x[1]^2 + x[2]^2))) - exp(0.5*(cos(2π*x[1]) + cos(2π*x[2]))) + ℯ + 20;
x = collect(-3:0.01:3); y = collect(-3:0.01:3); mat = reduce(hcat, [[f([x[i], y[j]]) for i in eachindex(x)] for j in eachindex(y)]);
plot(x, y, mat, st=:surface, camera=(20,30), xflip=true, zflip=true, cmap=cgrad(:inferno, rev=true), colorbar=false)
heatmap(x, y, mat)

x0 = [1.0, -1.0];
res = optimize(f, x0)

# Check out https://julianlsolvers.github.io/Optim.jl/stable/user/config/
# for all possible configurable options such as tolerance, function calls and time limit

# 'Optim.minimizer' and 'Optim.minimum' give the results

xstar = Optim.minimizer(res)
Optim.minimum(res)

scatter!([x0[1]], [x0[2]], c=:blue, markershape=:square, markersize=5, label="x0")
scatter!([xstar[1]], [xstar[2]], c=:red, markershape=:star5, markersize=10, label="x*")

# As you can see, this initial point leads us to converge to a local minimum.
# Unfortunately, there is no way to guarantee global convergence if our function is not convex.
# One good remedy is to do multiple starting points, and select the minimum from all starts.

xstars = zeros(1000, 2);
for i in 1:1000
    x0 = [randn(), randn()]
    res = optimize(f, x0)
    xstars[i, :] = Optim.minimizer(res)
end

plot(x, y, mat, levels=4)
scatter!(xstars[:, 1], xstars[:, 2], label=nothing, alpha=0.01, c=:red)

# If your function is univariate (over one variable), you can converge using bounds without needing a starting point

f_univariate(x) = 2x^2+3x+1
res = optimize(f_univariate, -2.0, 1.0)
x = -2:0.01:1; y = f_univariate.(x);
plot(x, y)
vline!([Optim.minimizer(res)], c=:red, label="minimum")

# If you have more than one variable, and you wish to impose bounds, you must specify a gradient.
# If you can't compute gradients explicitely, you can use automatic differentiating to do it for you.
# Below is an example

lb = [-0.5, -0.5];
ub = [0.5, 0.5];
x0 = [rand()-0.5, rand()-0.5];
res = optimize(f, lb, ub, x0, Fminbox(); autodiff=:forward)
xstar = Optim.minimizer(res)

x = collect(-3:0.01:3); y = collect(-3:0.01:3); mat = reduce(hcat, [[f([x[i], y[j]]) for i in eachindex(x)] for j in eachindex(y)]);
plot(x, y, mat, levels=4)
vline!([lb[1], ub[1]], label=nothing, c=:black)
hline!([lb[2], ub[2]], label="bounds", c=:black)
scatter!([x0[1]], [x0[2]], markershape=:square, label="start", c=:blue)
scatter!([xstar[1]], [xstar[2]], markershape=:star5, label="minimum", c=:red)

##### MultiVariateStats.jl ##### (https://juliastats.org/MultivariateStats.jl/dev/)

Pkg.add(["MultivariateStats", "RDatasets"]);
using MultivariateStats, RDatasets

# We will use a dataset that measures flowers in 4 ways (4 variables) and labels them into three categories.
# We will use multiple component analysis tools to reduce the dimensionality from 4 to 2

data = RDatasets.dataset("datasets", "iris")
X = Matrix(data[:, 1:4])'
y = Vector(data[:, 5])

function iris_plot(Xp, y, sp, title)
    for s in unique(y)
        points = Xp[:, y .== s]
        scatter!(points[1, :], points[2, :], label=nothing, sp=sp, title=title)
    end
end

# A whole set of analyses is implemented, here shown are:
# - PCA, you can set the output dimension with 'maxoutdim' (PPCA is similar)
# - LDA (Subspace LDA is similar)
# - ICA
# - Factor Analysis
# - MDS

pca = fit(PCA, X; maxoutdim=2)
lda = fit(MulticlassLDA, X, y, outdim=2)
ica = fit(ICA, X, 2; tol=1e-1)
fa = fit(FactorAnalysis, X; maxoutdim=2)
mds = fit(MDS, X; maxoutdim=2, distances=false)

# The results of each are shown below (they are all very similar)

plot(layout=(2, 3), size=(600, 400), showaxis=false, grid=false)

Xp = predict(pca, X);
iris_plot(Xp, y, 1, "PCA")

Xp = predict(lda, X);
iris_plot(Xp, y, 2, "LDA")

Xp = predict(ica, X);
iris_plot(Xp, y, 3, "ICA")

Xp = predict(fa, X);
iris_plot(Xp, y, 4, "FA")

Xp = predict(mds);
iris_plot(Xp, y, 5, "MDS")

plot!()

##### Flux.jl ##### (https://fluxml.ai/Flux.jl/stable/)

Pkg.add(["Flux", "ProgressBars", "Plots"]);
using Flux, ProgressBars, Plots, Statistics

# This is the most prominent deep learning package in Julia with great CUDA support.
# This simple tutorial will train a small network to solve the XOR task.
# Below we generate the input and target data.

input = rand(Float32, 2, 1000)
truth = [xor(col[1]>0.5, col[2]>0.5) for col in eachcol(input)]
target = Flux.onehotbatch(truth, [true, false])
loader = Flux.DataLoader((input, target), batchsize=64, shuffle=true);

# The model is very simply, it has 3 hidden units with tanh activation, a batchnorm and
# a softmax at the end.

model = Chain(
    Dense(2 => 3, tanh),   # activation function inside layer
    BatchNorm(3),
    Dense(3 => 2),
    softmax)

# These should be random outputs since the network is initialized randomly

initial_predictions = model(input)

# Here is where a lot of hyperparameters go, we'll keep it simple and use Adam with a momentum parameter of 0.01

optim = Flux.setup(Flux.Adam(0.01), model);

# This is the main training loop

losses = []
for epoch in ProgressBar(1:1_000)
    for (x, y) in loader
        loss, grads = Flux.withgradient(model) do m # This is the forward pass
            # Evaluate model and loss inside gradient context:
            y_hat = m(x)
            Flux.crossentropy(y_hat, y)
        end
        Flux.update!(optim, model, grads[1]) # This is the backprop step
        push!(losses, loss)  # logging, outside gradient context
    end
end

final_predictions = model(input)

# We plot the results

plot(losses; xaxis=(:log10, "iteration"), yaxis="loss", label="per batch")
n_batches = length(loader); n_epocs = length(losses);
plot!(n_batches:n_batches:n_epocs, mean.(Iterators.partition(losses, n_batches)), label="epoch mean")

plot(layout=(1, 3), size=(1000, 330))
p_true = scatter!(input[1,:], input[2,:], sp=1, zcolor=truth, title="True classification", legend=false)
p_raw =  scatter!(input[1,:], input[2,:], sp=2, zcolor=initial_predictions[1,:], title="Untrained network", label="", clims=(0,1))
p_done = scatter!(input[1,:], input[2,:], sp=3, zcolor=final_predictions[1,:], title="Trained network", legend=false)

##### DifferentialEquations.jl ##### (https://docs.sciml.ai/DiffEqDocs/stable/)

Pkg.add(["DifferentialEquations"]);
using DifferentialEquations

# This is perhaps the most successful and well-known package in Julia. It is the gold-standard
# if you're numerically solving any type of differential equation. 
# It works by setting up a differential equation problem, and then using 'solve' to solve it.
# To set up the problem, we need 3 things:
# - the dynamics equation
# - the initial conditions
# - the timepoints/time segment to evaluate at
# In this first very simple equation, we have du/dt = 1.01u
# The dynamics function represents the right hand side of this equation and takes in three arguments:
# - u the variable of interest (which may be a vector)
# - p the parameters of the model
# - t the timepoint
# We then set the initial conditions (here to 0.5) and the time range to between 0 and 1

f(u, p, t) = 1.01 * u;
u0 = 1 / 2;
tspan = (0.0, 1.0);
prob = ODEProblem(f, u0, tspan)

# To solve the problem, we simply pass 'prob' to 'solve'
# There are many arguments solve can take such as which algorithm to use, or which tolerance.

sol = solve(prob)

# In the multivariate case, the dynamics function 'f' would be slightly different.
# Below is the Lorenz attractor (3-dimensional). In this case, we have an extra argument: du
# du is the vector of gradients that we will fill in. We do this in place for memory reasons
# This means the function does not return anything (hence the '!')

function lorenz!(du, u, p, t)
    du[1] = 10.0 * (u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
end
prob = ODEProblem(lorenz!, [1.0; 0.0; 0.0], (0.0, 100.0))
sol = solve(prob)
plot(sol, idxs = (1, 2, 3))

# Example using Hodgkin-Huxley model
# The example below implements the 4-dimensional HH model (https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model)

# Potassium ion-channel rate functions
alpha_n(v) = (0.02 * (v - 25.0)) / (1.0 - exp((-1.0 * (v - 25.0)) / 9.0));
beta_n(v) = (-0.002 * (v - 25.0)) / (1.0 - exp((v - 25.0) / 9.0));

# Sodium ion-channel rate functions
alpha_m(v) = (0.182 * (v + 35.0)) / (1.0 - exp((-1.0 * (v + 35.0)) / 9.0));
beta_m(v) = (-0.124 * (v + 35.0)) / (1.0 - exp((v + 35.0) / 9.0));

alpha_h(v) = 0.25 * exp((-1.0 * (v + 90.0)) / 12.0);
beta_h(v) = (0.25 * exp((v + 62.0) / 6.0)) / exp((v + 90.0) / 12.0);

function HH!(du, u, p, t)
    gK, gNa, gL, EK, ENa, EL, C, I = p
    v, n, m, h = u
    # membrane potential
    du[1] = (-(gK * (n^4.0) * (v - EK)) - (gNa * (m^3.0) * h * (v - ENa)) -
             (gL * (v - EL)) + I) / C
    # channel activations n, m, h
    du[2] = (alpha_n(v) * (1.0 - n)) - (beta_n(v) * n)
    du[3] = (alpha_m(v) * (1.0 - m)) - (beta_m(v) * m)
    du[4] = (alpha_h(v) * (1.0 - h)) - (beta_h(v) * h)
end

# n, m & h steady-states
n_inf(v) = alpha_n(v) / (alpha_n(v) + beta_n(v));
m_inf(v) = alpha_m(v) / (alpha_m(v) + beta_m(v));
h_inf(v) = alpha_h(v) / (alpha_h(v) + beta_h(v));

#    gK,   gNa,  gL,  EK,    ENa,  EL,    C, I
p = [35.0, 40.0, 0.3, -77.0, 55.0, -65.0, 1, 0];
u0 = [-60, n_inf(-60), m_inf(-60), h_inf(-60)];
tspan = (0.0, 1000);

# This sets the input current to 1 once time has reached 100
current_step = PresetTimeCallback(100, integrator -> integrator.p[8] += 1)

prob = ODEProblem(HH!, u0, tspan, p, callback=current_step)

sol = solve(prob, Tsit5());
plot(sol, vars = 1)

plot(sol, vars = [2, 3, 4], tspan = (105.0, 130.0))

