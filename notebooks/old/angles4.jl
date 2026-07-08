using DrWatson
using DiscreteCalculus
using CairoMakie
using StaticArrays
using VertexModel
using LinearAlgebra
using SparseArrays
using Random
using Colors 
using JLD2
using Dates
using FromFile
using InvertedIndices
using LaTeXStrings
using Printf
using REPL.TerminalMenus
using CircularArrays
using OrdinaryDiffEq
using StatsBase
using FFTW
using CurveFit

function polynomialFit(x, u)
    return sum(u.*x.^collect(0:length(u)-1))
end
function exponentialFit(x, u)
    return u.k[1] + sum(u.p.*exp.(x.*u.λ)) 
end
function exponentialDeriv(x, u)
    return sum(u.p.*u.λ.*exp.(x.*u.λ)) 
end

@from "$(srcdir("AblateCells.jl"))" using AblateCells
@from "$(srcdir("Stresses.jl"))" using Stresses

# f₁(r) = 1/r 
# f₂(r) = 1/(r^2) 
# # ϕ(r, θ, θ₀, a, b) = f₁(r)*a + f₂(r)*b*cos(2*(θ-θ₀))
# ϕ(r, θ, θ₀, a, b) = a/b + cos(2*(θ-θ₀))

dateString = "25-12-02-16-58-52"

testFiles = filter(x->(occursin("Divided",x)||occursin("Ablated",x))&&occursin(".jld2",x), readdir(datadir("displacementFields", dateString)))
dividedCells = [i[(length(dateString)+1+length("Divided"))+1:end-5] for i in testFiles if occursin("Divided", i)]
ablatedCells = [i[(length(dateString)+1+length("Ablated"))+1:end-5] for i in testFiles if occursin("Ablated", i)]
testCells = parse.(Int64, dividedCells∩ablatedCells)

acceptedCells = []

divθs = []
maxθsSmoothed = []
maxθsExponential = []
maxθsPolynomial = []
maxθsRaw = []

for ablatedCell in testCells
    # Import and process data
    importedData = load(datadir("displacementFields", dateString, "$(dateString)_InitialSystem.jld2"))
    @unpack integ1 = importedData
    (params1, matrices1) = integ1.p 
    # @unpack A, B = matrices2 
    R1 = reinterpret(SVector{2,Float64}, integ1.u)
    cellCentres = findCellCentresOfMass(R1, matrices1.A, matrices1.B)
    importedDataAblated = load(datadir("displacementFields", dateString, "$(dateString)_Ablated$(ablatedCell).jld2"))
    @unpack Rablated, Aablated, Bablated, Fablated, ablationCOM = importedDataAblated
    cellCentresAblated = findCellCentresOfMass(Rablated, Aablated, Bablated)
    cellPolygonsAblated = findCellPolygons(Rablated, Aablated, Bablated)
    Δrᵢ = cellCentresAblated.-cellCentres[Not(ablatedCell)]
    radiusVectorsAblated = ([cellCentres[i].-ablationCOM for i=1:size(matrices1.B,1)])[Not(ablatedCell)]
    directionsAblated = normalize.(Δrᵢ).⋅normalize.(radiusVectorsAblated)
    θs = atan.(getindex.(radiusVectorsAblated, 1), getindex.(radiusVectorsAblated, 2))
    # centralcells = collect(1:size(Bablated,1)) # findall(x->norm(x)<0.33*maximum(norm.(radiusVectorsAblated)), radiusVectorsAblated)
    centralcells = findall(x->norm(x)<0.33*maximum(norm.(radiusVectorsAblated)), radiusVectorsAblated)
    θsTruncated = θs[centralcells]
    θsSorted = CircularArray(sort(θsTruncated))
    radiusVectorsSorted = CircularArray(radiusVectorsAblated[sortperm(θsTruncated)])

    # Raw data
    directionsTruncated = directionsAblated[centralcells]
    directionsSorted = CircularArray(directionsTruncated[sortperm(θsTruncated)])
    signs = sign.(directionsSorted)
    difs = signs.-signs[0:end-1]
    intersectInds = findall(x->x!=0, difs)
    intersects = Float64[]
    for ind in intersectInds
        dy = directionsSorted[ind]-directionsSorted[ind-1]
        dx = θsSorted[ind] - θsSorted[ind-1]
        # 0 = directionsTruncated[ind-1] + xDif*dy/dx
        push!(intersects, (θsSorted[ind-1] - directionsSorted[ind-1]*dx/dy))
    end
    push!(maxθsRaw, θsSorted[findmax(directionsSorted)[2]])
    
    # Smoothed data 
    h1 = fit(Histogram, θsSorted, weights(directionsSorted), -π:π/20:π+π/50)
    h2 = fit(Histogram, θsSorted, -π:π/20:π+π/50)
    binEdges = collect(h1.edges[1])
    deleteat!(binEdges, findall(x->x==0, h2.weights))
    binMidpoints = CircularArray(binEdges[1:end-1] .+ (binEdges[2:end]-binEdges[1:end-1])/2.0)
    smoothedData = CircularArray([h1.weights[i]/h2.weights[i] for i=1:length(h2.weights) if h2.weights[i]!=0])
    lines!(axes[3], binMidpoints, smoothedData, color=:green)
    signsSmoothed = CircularArray(sign.(smoothedData))
    difs = signsSmoothed.-signsSmoothed[0:end-1]
    intersectInds = findall(x->x!=0, difs)
    intersectsSmoothed = Float64[]
    for ind in intersectInds
        dy = smoothedData[ind]-smoothedData[ind-1]
        dx = binMidpoints[ind]-binMidpoints[ind-1]
        push!(intersectsSmoothed, (binMidpoints[ind-1] - smoothedData[ind-1]*dx/dy))
    end
    push!(maxθsSmoothed, binMidpoints[findmax(smoothedData)[2]])
    
    # Exponential fit 
    prob = CurveFitProblem(Vector(θsSorted), Vector(directionsSorted))
    sol = solve(prob, ExpSumFitAlgorithm(n=20, withconst=true))
    xs = collect(-π:0.01:π)
    ys = []
    for i=1:length(xs)
        push!(ys, exponentialFit(xs[i], sol.u))
    end
    push!(maxθsExponential, xs[findmax(real.(ys[4:end-3]))[2]])

    # Polynomial fit 
    prob = CurveFitProblem(Vector(θsSorted), Vector(directionsSorted))
    sol = solve(prob, PolynomialFitAlgorithm(degree=20))
    xs = collect(-π:0.01:π)
    ys = []
    for i=1:length(xs)
        push!(ys, polynomialFit(xs[i], sol.u))
    end
    push!(maxθsPolynomial, xs[findmax(ys[4:end-3])[2]])

    eigenVals, eigenVecs = eigen(matrices1.cellShapeTensor[ablatedCell]) 
    divθ = atan(eigenVecs[1,1], eigenVecs[2,1])
    push!(divθs, divθ)
    
end

#%%

fig = Figure()
ax = Axis(fig[1,1], aspect=DataAspect())
scatter!(ax, mod.(divθs.+3π/2, π), mod.(maxθsRaw.+π, π), color=:black)
scatter!(ax, mod.(divθs.+3π/2, π), mod.(maxθsSmoothed.+π, π), color=:red)
scatter!(ax, mod.(divθs.+3π/2, π), mod.(maxθsExponential.+π, π), color=:green)
scatter!(ax, mod.(divθs.+3π/2, π), mod.(maxθsPolynomial.+π, π), color=:blue)
ax.xlabel("Long axis angle of ablated cell")
ax.ylabel("Peak expansion angle")
xlims!(ax, (0,π))
ylims!(ax, (0,π))
display(fig)