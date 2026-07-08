using DrWatson
using DiscreteCalculus
# using CairoMakie
using GLMakie
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

options = ["yes", "no"]
menu = RadioMenu(options, pagesize=length(options))

f₁(r) = 1/r 
f₂(r) = 1/(r^2) 
# ϕ(r, θ, θ₀, a, b) = f₁(r)*a + f₂(r)*b*cos(2*(θ-θ₀))
ϕ(r, θ, θ₀, a, b) = a/b + cos(2*(θ-θ₀))

dateString = "25-12-02-16-58-52"

testFiles = filter(x->(occursin("Divided",x)||occursin("Ablated",x))&&occursin(".jld2",x), readdir(datadir("displacementFields", dateString)))
dividedCells = [i[(length(dateString)+1+length("Divided"))+1:end-5] for i in testFiles if occursin("Divided", i)]
ablatedCells = [i[(length(dateString)+1+length("Ablated"))+1:end-5] for i in testFiles if occursin("Ablated", i)]
testCells = parse.(Int64, dividedCells∩ablatedCells)

acceptedCells = []

GLMakie.activate!()
fig = Figure(size=(1000,1500))
display(fig)
axes = [Axis(fig[1,1], aspect=DataAspect())]
hidedecorations!(axes[1])
hidespines!(axes[1])
push!(axes, Axis(fig[1,2]))
push!(axes, Axis(fig[2,1]))
push!(axes, Axis(fig[2,2]))
push!(axes, Axis(fig[3,1]))
polarAxis = PolarAxis(fig[3, 2])
hidedecorations!(polarAxis)

for ablatedCell in testCells
    @show ablatedCell
    
    empty!.(axes)
    empty!(polarAxis)
    # Label(fig[1,:,Top()], "Cell $ablatedCell")

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

    # Monolayer visualisation
    for i=1:size(Bablated,1)
        if i∈centralcells
            poly!(axes[1], cellPolygonsAblated[i], color=directionsAblated[i], colormap=:managua, colorrange=(-1.0,1.0), strokewidth=1, strokecolor=(:black,0.1))
        else
            poly!(axes[1], cellPolygonsAblated[i], color=(:black,0.1), strokewidth=1, strokecolor=(:black,0.1))
        end
    end

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
    lines!(axes[2], θsSorted, directionsSorted, color=:red)
    scatter!(axes[2], intersects, zeros(length(intersects)), color=:red)

    # Smoothed data 
    h1 = fit(Histogram, θsSorted, weights(directionsSorted), -π:π/20:π+π/50)
    h2 = fit(Histogram, θsSorted, -π:π/20:π+π/50)
    binEdges = collect(h1.edges[1])
    deleteat!(binEdges, findall(x->x==0, h2.weights))
    binMidpoints = binEdges[1:end-1] .+ (binEdges[2:end]-binEdges[1:end-1])/2.0
    smoothedData = [h1.weights[i]/h2.weights[i] for i=1:length(h2.weights) if h2.weights[i]!=0]
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
    scatter!(axes[3], intersectsSmoothed, zeros(length(intersectsSmoothed)), color=:green)
    @show findmax(smoothedData)
    @show binMidpoints[findmax(smoothedData)[2]]
    
    # Exponential fit 
    prob = CurveFitProblem(Vector(θsSorted), Vector(directionsSorted))
    sol = solve(prob, ExpSumFitAlgorithm(n=20, withconst=true))
    xs = collect(-π:0.01:π)
    ys = []
    for i=1:length(xs)
        push!(ys, exponentialFit(xs[i], sol.u))
    end
    lines!(axes[4], xs, real.(ys))
    # @show findmax(real.(ys))
    # @show xs[findmax(real.(ys))[2]]

    # Polynomial fit 
    prob = CurveFitProblem(Vector(θsSorted), Vector(directionsSorted))
    sol = solve(prob, PolynomialFitAlgorithm(degree=20))
    xs = collect(-π:0.01:π)
    ys = []
    for i=1:length(xs)
        push!(ys, polynomialFit(xs[i], sol.u))
    end
    lines!(axes[5], xs, real.(ys))

    eigenVals, eigenVecs = eigen(matrices1.cellShapeTensor[ablatedCell]) 
    divθ = atan(eigenVecs[1,1], eigenVecs[2,1])
    for int in intersects
        lines!(polarAxis, [int, int], [0,1], color=:red)    
    end
    for int in intersectsSmoothed
        lines!(polarAxis, [int, int], [0,1], color=:green)    
    end
    lines!(polarAxis, [divθ+π/2, divθ-π/2, divθ-π/2], [1.0,0.0,1.0], color=:blue)
    
    lines!(polarAxis, [binMidpoints[findmax(smoothedData)[2]], 0.0, binMidpoints[findmax(smoothedData)[2]]-π], [1.0,0.0,1.0], color=:blue)
    
    @show divθ

    ylims!(axes[2], (minimum(directionsSorted),maximum(directionsSorted)))
    ylims!(axes[3], (minimum(directionsSorted),maximum(directionsSorted)))
    ylims!(axes[4], (minimum(directionsSorted),maximum(directionsSorted)))
    ylims!(axes[5], (minimum(directionsSorted),maximum(directionsSorted)))

    
    
    # `request` displays the menu and returns the index after the
    #   user has selected a choice
    choice = request("Accept?", menu)
    if options[choice] == "yes"
        push!(acceptedCells, ablatedCell)
    end

end

# display(fig)



# using NFFT
# scaledBinMidpoints = (binMidpoints.-binMidpoints[1])./(binMidpoints[end]-binMidpoints[1]+0.01).-0.5

# fHat = nfft(scaledBinMidpoints,smoothedData)
# y = nfft_adjoint(k, N, fHat)

# J, N = 8, 16
# k = range(-0.4, stop=0.4, length=J)  # nodes at which the NFFT is evaluated
# f = randn(ComplexF64, J)             # data to be transformed
# fHat = nfft(k, f)
# y = nfft_adjoint(k, N, fHat)

# # cs = fft(smoothedData)

# # fourierComponent(c, n, x) = c*(exp(im*n*x))# + exp(-im*n*x))

# # function fourierToCurve!(cs, xs, ys)
# #     for (i,x) in enumerate(xs)
# #         ys[i] = cs[1]
# #         for (n,c) in enumerate(cs)
# #             ys[i] += fourierComponent(c, n, x)
# #         end
# #     end
# #     return nothing 
# # end


# # xs = collect(0:2π/100:2π)
# # ys = Complex{Float64}.(zeros(size(xs)))

# # line = fourierToCurve!(cs, xs, ys)

# # fig2 = Figure()
# # axes[ ]= Axis(fig2[1,1])
# # lines!(axes[,] xs, ys)
# # display(fig2)