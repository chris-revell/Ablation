# Julia packages
using DrWatson
using FromFile
using OrdinaryDiffEq
using LinearAlgebra
using JLD2
using SparseArrays
using StaticArrays
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
@from "$(srcdir("DivideCell.jl"))" using DivideCell

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
dateString = "25-12-19-12-12-33"
# !isdir(datadir("division", dateString)) ? mkpath(datadir("division", dateString)) : nothing 
# jldsave(datadir("division", dateString, "$(dateString)_InitialSystem.jld2"); 
#             integ1,
#         )

data = load("/Users/christopher/Postdoc/Code/Ablation/data/division/$(dateString)/$(dateString)_InitialSystem.jld2")
@unpack integ1 = data
(params1, matrices1) = integ1.p 
R1 = reinterpret(SVector{2,Float64}, integ1.u)

systemCOM1 = sum(R1)./length(R1)

# (pos, ind) = findmin(norm.([cell.-systemCOM1 for cell in matrices1.cellPositions]))
ind = rand(findall(x->x==0, findPeripheralCells(matrices1.B)))

Rtmp, Atmp, Btmp, shortvec = divideCell(R1, params1, matrices1, ind) 

#%%

integ2 = vertexModel(
    initialSystem = "argument",
    nCycles = 0.1,
    divisionToggle = 0,
    outputTotal = 1,
    outputToggle = 0,
    frameDataToggle = 0,
    frameImageToggle = 0,
    printToggle = 1,
    plotCells = 0,
    energyModel = "quadratic",
    R_in = Rtmp,
    A_in = Atmp,
    B_in = Btmp,
)
jldsave(datadir("division", dateString, "$(dateString)_Division_Cell$(ind).jld2"); 
            integ2,
        )
(params2, matrices2) = integ2.p 
# @unpack A, B = matrices2 
R2 = reinterpret(SVector{2,Float64}, integ2.u)

#%%

ablatedR, ablatedA, ablatedB = ablateCells(R1, matrices1.A, matrices1.B, [ind])
integ3 = vertexModel(
    initialSystem = "argument",
    nCycles = 0.1,
    divisionToggle = 0,
    outputTotal = 1,
    outputToggle = 0,
    frameDataToggle = 0,
    frameImageToggle = 0,
    printToggle = 1,
    plotCells = 0,
    energyModel = "quadratic",
    R_in = ablatedR,
    A_in = ablatedA,
    B_in = ablatedB,
)
jldsave(datadir("division", dateString, "$(dateString)_Ablation_Cell$(ind).jld2"); 
            integ3,
        )
(params3, matrices3) = integ3.p 
# @unpack A, B = matrices3 
R3 = reinterpret(SVector{2,Float64}, integ3.u)

#%%


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
hidespines!(axes[end])
hidedecorations!(axes[end])

push!(axes, Axis(fig, aspect=DataAspect()))
cellPolygons2 = makeCellPolygons(R2, params2, matrices2)
for i = 1:params2.nCells
    if i!=ind && i!= params2.nCells
        poly!(axes[end], cellPolygons2[i], color=(:green, 0.25), strokecolor=(:black, 1.0), strokewidth=2)
    else 
        poly!(axes[end], cellPolygons2[i], color=(:black, 0.5), strokecolor=(:black, 1.0), strokewidth=2)
    end
end
hidespines!(axes[end])
hidedecorations!(axes[end])


# systemCOM2 = sum(R2)./params2.nVerts
# COMDisplacement = systemCOM2.-systemCOM
# cellDisplacements = [matrices2.cellPositions[i].-matrices1.cellPositions[i].-COMDisplacement for i=1:params1.nCells]
cellDisplacements = [matrices2.cellPositions[i].-matrices1.cellPositions[i] for i=1:params1.nCells]
cellDisplacementNorms = norm.(cellDisplacements)
radiusVectors = [cell.-matrices1.cellPositions[ind] for cell in matrices1.cellPositions]
radii = norm.(radiusVectors)
arrowColours = normalize.(radiusVectors).⋅normalize.(cellDisplacements)
clims = (-1.0, 1.0)

push!(axes, Axis(fig))
scatter!(axes[end], log10.(radii[Not(ind)]), log10.(cellDisplacementNorms[Not(ind)]), color=arrowColours[Not(ind)], colorrange=clims, colormap=:managua)
xs = log10.(norm.(radiusVectors[Not(ind)]))
axes[end].xlabel = L"\log_{10}\left(R_i\right)"
axes[end].ylabel = L"\log_{10}\left(\Delta R_i\right)"
# ys = -1.5.-1.0.*xs
# lines!(axes[end], xs, ys, color=:blue)
# ys = -1.25.-1.0.*xs
# lines!(axes[end], xs, ys, color=:red)

push!(axes, Axis(fig, aspect=DataAspect()))
for i = 1:params2.nCells-1
    if i!=ind 
        poly!(axes[end], cellPolygons2[i], color=arrowColours[i], colorrange=(-1.0,1.0), colormap=:managua, strokecolor=(:black, 0.2), strokewidth=1)
    end
end
# arrows2d!(axes[end], Point{2,Float64}.(matrices1.cellPositions[Not(ind)]), Vec{2,Float64}.(cellDisplacements[Not(ind)]), color=arrowColours, lengthscale=20.0)
# arrows!(axes[end], Point{2,Float64}.(matrices1.cellPositions[Not(ind)]), Vec{2,Float64}.(cellDisplacements[Not(ind)]), color=arrowColours[Not(ind)], colorrange=clims, colormap=:managua, lengthscale=20.0, linewidth=5)
arrows2d!(axes[end], Point{2,Float64}.(matrices1.cellPositions[Not(ind)]), Vec{2,Float64}.(cellDisplacements[Not(ind)]), color=:black, lengthscale=20.0)
# scatter!(axes[end], Point{2,Float64}(systemCOM1))
lines!(axes[end], Point{2,Float64}.([matrices1.cellPositions[ind].-shortvec.*2, matrices1.cellPositions[ind].+shortvec.*2]), color=(:black,0.5), linewidth=4)
hidedecorations!(axes[end])
hidespines!(axes[end])





push!(axes, Axis(fig, aspect=DataAspect()))
cellPolygons3 = makeCellPolygons(R3, params3, matrices3)
for i = 1:params3.nCells
    poly!(axes[end], cellPolygons3[i], color=(:green, 0.25), strokecolor=(:black, 1.0), strokewidth=2)
end
hidespines!(axes[end])
hidedecorations!(axes[end])

cellDisplacements = matrices3.cellPositions.-matrices1.cellPositions[Not(ind)]
cellDisplacementNorms = norm.(cellDisplacements)
# C = abs.(matrices1.B)*abs.(matrices1.A)./2
# ablatedCellVertices = R1[findall(x->x!=0, C[ind, :])]
# ablationCentre = sum(ablatedCellVertices)./length(ablatedCellVertices)
radiusVectors = [cell.-matrices1.cellPositions[ind] for cell in matrices1.cellPositions]
radii = norm.(radiusVectors)
arrowColours = normalize.(radiusVectors[Not(ind)]).⋅normalize.(cellDisplacements)
clims = (-1.0, 1.0)

push!(axes, Axis(fig))
scatter!(axes[end], log10.(radii[Not(ind)]), log10.(cellDisplacementNorms), color=arrowColours, colorrange=clims, colormap=:managua)
xs = log10.(norm.(radiusVectors[Not(ind)]))
axes[end].xlabel = L"\log_{10}\left(R_i\right)"
axes[end].ylabel = L"\log_{10}\left(\Delta R_i\right)"
# ys = -1.5.-1.0.*xs
# lines!(axes[end], xs, ys, color=:blue)
# ys = -1.25.-1.0.*xs
# lines!(axes[end], xs, ys, color=:red)

push!(axes, Axis(fig, aspect=DataAspect()))
for i = 1:params3.nCells
    poly!(axes[end], cellPolygons3[i], color=arrowColours[i], colorrange=(-1.0,1.0), colormap=:managua, strokecolor=(:black, 0.2), strokewidth=1)
end
# arrows!(axes[end], Point{2,Float64}.(matrices3.cellPositions), Vec{2,Float64}.(cellDisplacements), color=arrowColours, colorrange=clims, colormap=:managua, lengthscale=20.0, linewidth=5)
arrows2d!(axes[end], Point{2,Float64}.(matrices3.cellPositions), Vec{2,Float64}.(cellDisplacements), color=:black, lengthscale=20.0)
# scatter!(axes[end], Point{2,Float64}(systemCOM1))
# lines!(axes[end], Point{2,Float64}.([matrices3.cellPositions[ind].-shortvec.*2, matrices3.cellPositions[ind].+shortvec.*2]), color=(:black,0.5), linewidth=4)
hidedecorations!(axes[end])
hidespines!(axes[end])


subfigureOrdering = CartesianIndex.([(1,1), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)])
for (n,i) in enumerate(subfigureOrdering)
    fig[i[1], i[2]] = axes[n]
end

display(fig)

!isdir(plotsdir("division")) ? mkpath(plotsdir("division")) : nothing 
save(plotsdir("division", "$(Dates.format(Dates.now(),"yy-mm-dd-HH-MM-SS"))_Cell$(ind).png"), fig)
display(fig)