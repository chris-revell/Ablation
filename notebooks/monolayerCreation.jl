using DrWatson
using DiscreteCalculus
using CairoMakie
using StaticArrays
using VertexModel#ablation
using LinearAlgebra
using SparseArrays
using Random
using Colors 
using JLD2
using Dates
using FromFile
using InvertedIndices
using OrdinaryDiffEq

@from "$(srcdir("AblateCells.jl"))" using AblateCells

#%%

subDir = "quadraticPotentialWithPressure"
systems = ["NoHole", "SingleHole", "DoubleHole", "Large"]
system = systems[4]

#%%
integ1 = vertexModel(abstol = 1e-9,
                    reltol = 1e-9,
                    nRows=11,
                    nCycles=(system=="Large" ? 3 : 2),
                    printToggle=1,
                    frameDataToggle=0,
                    frameImageToggle=0,
                    videoToggle=0,
                    setRandomSeed=12345,
                    divisionToggle=1,
                    pressureExternal=0.1,
                    energyModel="quadratic",
                )


R1 = reinterpret(SVector{2,Float64}, integ1.u) 
params1, matrices1 = integ1.p
A1 = matrices1.A
B1 = matrices1.B

cellPolygons1 = findCellPolygons(R1, A1, B1)
cellCentres1 = findCellCentresOfMass(R1, A1, B1)

fig = Figure(size=(500,1500))
ax = Axis(fig[1,1], aspect=DataAspect())
hidedecorations!(ax)
hidespines!(ax)
for i=1:size(B1,1)
    poly!(ax, cellPolygons1[i], color=(:black, 0.2), strokecolor=(:black, 1.0), strokewidth=2)
end
scatter!(ax, Point{2,Float64}.(cellCentres1), color=(:black,1.0), markersize=10)
annotations!(ax, string.(collect(1:params1.nCells)), Point{2,Float64}.(cellCentres1), fontsize=12, color=(:black,1.0))
display(fig)

#%%

systemCOM = sum(R1)./length(R1)
neighbourMatrix = dropzeros(B1*transpose(B1))

if system=="SingleHole"
    centralCell = findmin(norm.([cellCentres1[i].-systemCOM for i=1:size(B1,1)]))[2]
    ablatedCells = unique(findall(x->x!=0, neighbourMatrix[centralCell,:]))
    # for _ = 1:2
    #     neighbours = getindex.(findall(x->x!=0, neighbourMatrix[ablatedCells,:]),2)
    #     append!(ablatedCells, unique(neighbours))
    # end
elseif system=="DoubleHole"
    yrange = maximum(getindex.(R1,2))-minimum(getindex.(R1,2))
    holes = [systemCOM.+[0.0, yrange/6], systemCOM.+[0.0, -yrange/6]]
    centralCells = [findmin(norm.([cellCentres1[i].-hole for i=1:size(B1,1)]))[2] for hole in holes]
    ablatedCells = unique(getindex.(findall(x->x!=0, neighbourMatrix[centralCells,:]), 2))
    # for _ = 1:1
    #     neighbours = getindex.(findall(x->x!=0, neighbourMatrix[ablatedCells,:]),2)
    #     append!(ablatedCells, unique(neighbours))
    # end
else 
    ablatedCells = []
end
unique!(ablatedCells)
ablatedR, ablatedA, ablatedB = ablateCells(R1, A1, B1, ablatedCells)
cellPolygonsAblated = findCellPolygons(ablatedR, ablatedA, ablatedB)
cellCentresAblated = findCellCentresOfMass(ablatedR, ablatedA, ablatedB)

ax2 = Axis(fig[2,1], aspect=DataAspect())
for i=1:size(ablatedB,1)
    poly!(ax2, cellPolygonsAblated[i], color=(:black, 0.2), strokecolor=(:black, 1.0), strokewidth=2)
end
scatter!(ax2, Point{2,Float64}.(cellCentresAblated), color=(:black,1.0), markersize=10)
annotations!(ax2, string.(collect(1:size(B1,1)))[Not(ablatedCells)], Point{2,Float64}.(cellCentresAblated), fontsize=12, color=(:black,1.0))
hidedecorations!(ax2)
hidespines!(ax2)
display(fig)

#%%

integ2 = vertexModel(abstol = 1e-9,
                    reltol = 1e-9,
                    initialSystem="argument",
                    divisionToggle=0,
                    R_in=ablatedR,
                    A_in=ablatedA,
                    B_in=ablatedB,
                    pressureExternal=0.1,
                    nCycles=1.0,
                    outputToggle=0,
                    frameDataToggle=0,
                    frameImageToggle=0,
                    videoToggle=0,
                    printToggle=1,
                    energyModel="quadratic",
                    # termSteadyState=true,
                )

#%%
Rfinal = reinterpret(SVector{2,Float64}, integ2.u) 
params2, matrices2 = integ2.p
Afinal = matrices2.A
Bfinal = matrices2.B
Ffinal = matrices2.F
@show maximum(norm.(sum(Ffinal, dims=2)))
cellTensions = matrices2.cellTensions
cellPressures = matrices2.cellPressures
cellPerimeters = matrices2.cellPerimeters
cellAreas = matrices2.cellAreas

cellPolygons2 = findCellPolygons(Rfinal, Afinal, Bfinal)
cellCentres2 = findCellCentresOfMass(Rfinal, Afinal, Bfinal)

ax3 = Axis(fig[3,1], aspect=DataAspect())
hidedecorations!(ax3)
hidespines!(ax3)
for i=1:size(Bfinal,1)
    poly!(ax3, cellPolygons2[i], color=(:black, 0.2), strokecolor=(:black, 1.0), strokewidth=2)
end
scatter!(ax3, Point{2,Float64}.(cellCentres2), color=(:black,1.0), markersize=10)
annotations!(ax3, string.(collect(1:size(B1,1)))[Not(ablatedCells)], Point{2,Float64}.(cellCentres2), fontsize=12, color=(:black,1.0))
display(fig)

R = Rfinal
A = Afinal
B = Bfinal
F = Ffinal
# cellTensions = cellTensions
# cellPressures = cellPressures
# cellPerimeters = cellPerimeters
# cellAreas

jldsave(datadir(subDir, "$(system)_testSystem.jld2"); R,
    A, 
    B, 
    F, 
    cellTensions, 
    cellPressures, 
    cellPerimeters, 
    cellAreas,
)
