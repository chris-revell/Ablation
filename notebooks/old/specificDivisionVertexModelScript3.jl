# Julia packages
using DrWatson
using FromFile
using OrdinaryDiffEq
using LinearAlgebra
using JLD2
using SparseArrays
using StaticArrays
using CairoMakie
using Printf
using CairoMakie
using VertexModel
using GeometryBasics
using Random
using Colors
using InvertedIndices
using Dates
using DiscreteCalculus

@from "$(srcdir("AblateCells.jl"))" using AblateCells

function makeCellPolygons(R,params,matrices)
    cellPolygons = Vector{Point{2,Float64}}[]
    for i=1:params.nCells
        push!(cellPolygons,Point{2,Float64}.(R[matrices.cellVertexOrders[i]]))
    end
    return cellPolygons
end
getRandomColor(seed) = RGB(rand(Xoshiro(seed),3)...)

# integ0 = vertexModel(
#     nRows = 15,
#     nCycles = 2,
#     divisionToggle = 1,
#     outputTotal = 1,
#     outputToggle = 0,
#     frameDataToggle = 0,
#     frameImageToggle = 0,
#     printToggle = 1,
#     plotCells = 0,
#     energyModel = "quadratic",
# )

# #%%
# (params0, matrices0) = integ0.p 
# # @unpack A, B = matrices 
# R0 = reinterpret(SVector{2,Float64}, integ0.u)

# integ1 = vertexModel(
#     initialSystem = "argument",
#     nCycles = 1,
#     divisionToggle = 0,
#     outputTotal = 1,
#     outputToggle = 0,
#     frameDataToggle = 0,
#     frameImageToggle = 0,
#     printToggle = 1,
#     plotCells = 0,
#     energyModel = "quadratic",
#     R_in = R0,
#     A_in = matrices0.A, 
#     B_in = matrices0.B,
# )
# (params1, matrices1) = integ1.p 
# # @unpack A, B = matrices1 
# R1 = reinterpret(SVector{2,Float64}, integ1.u)

# dateString = "$(Dates.format(Dates.now(),"yy-mm-dd-HH-MM-SS"))"
dateString = "25-12-02-16-58-52"
# !isdir(datadir("division", dateString)) ? mkpath(datadir("division", dateString)) : nothing 
# jldsave(datadir("division", dateString, "$(dateString)_InitialSystem.jld2"); 
#             integ1,
#         )

data = load("/Users/christopher/Postdoc/Code/Ablation/data/division/$(dateString)/$(dateString)_InitialSystem.jld2")
@unpack integ1 = data
(params1, matrices1) = integ1.p 
R1 = reinterpret(SVector{2,Float64}, integ1.u)

systemCOM1 = sum(R1)./length(R1)

(pos, ind) = findmin(norm.([cell.-systemCOM1 for cell in matrices1.cellPositions]))
# ind = rand(findall(x->x==0, findPeripheralCells(matrices1.B)))


# Δr = f(r)(a+bcos(2θ-θ₀))
# Δr̄ = (1 + (b/a)*cos(2θ-θ₀))

radiusVectors = [cell.-matrices1.cellPositions[ind] for cell in matrices1.cellPositions]
radii = norm.(radiusVectors)
θs = atan.(getindex.(radiusVectors,1), getindex.(radiusVectors,2))  

displacementFunction(abratio, θ₀, θ, r) = (1/r)*(1+abratio*cos(2*θ-θ₀))

prefactors = displacementFunction.(10.0, 0.0, θs[Not(ind)], radii[Not(ind)])
plottedVectors = normalize.(radiusVectors[Not(ind)]).*prefactors

# arrowColours = normalize(radiusVectors[Not(ind)]).⋅normalize(plottedVectors)

fig = Figure(size=(1500,1500))
axes = Axis[]
push!(axes, Axis(fig, aspect=DataAspect()))
cellPolygons1 = makeCellPolygons(R1, params1, matrices1)
for i in 1:params1.nCells
    if i!=ind
        poly!(axes[end], cellPolygons1[i], color=(:green, 0.25), strokecolor=(:black, 1.0), strokewidth=2)
    else 
        poly!(axes[end], cellPolygons1[i], color=(:black, 0.5), strokecolor=(:black, 1.0), strokewidth=2)
    end
end
clims = (-maximum(abs.(prefactors)), maximum(abs.(prefactors)))
arrows!(axes[end], Point{2,Float64}.(matrices1.cellPositions[Not(ind)]), plottedVectors, color=prefactors, colorrange=clims, colormap=:bwr)
hidespines!(axes[end])
hidedecorations!(axes[end])

fig[1,1] = axes[end]
display(fig)

