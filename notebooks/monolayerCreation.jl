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
getRandomColor(seed) = RGB(rand(Xoshiro(seed),3)...)

#%%
integ = vertexModel(nRows=5, nCycles=0.5, outputToggle=0, frameDataToggle=0, frameImageToggle=0, videoToggle=0, setRandomSeed=1234, divisionToggle=1)
# integ = vertexModel(solver=Vern9(), abstol = 1e-9, reltol = 1e-9, nRows=11, nCycles=1.0, outputToggle=0, frameDataToggle=0, frameImageToggle=0, videoToggle=0, setRandomSeed=1234)
R = reinterpret(SVector{2,Float64}, integ.u) 
params, matrices = integ.p
@unpack A, B = matrices 

# # Import system data
# conditionsDict    = load(datadir("oldPaper", "dataFinal.jld2"))
# @unpack nVerts,nCells,nEdges,pressureExternal,γ,λ,viscousTimeScale,realTimetMax,tMax,dt,outputInterval,outputTotal,realCycleTime,t1Threshold = conditionsDict["params"]
# matricesDict = load(datadir("oldPaper", "matricesFinal.jld2"))
# paramsDict = load(datadir("oldPaper", "params.jld2"))
# @unpack A,B,C,R,F,cellAreas,cellPressures,cellTensions,cellPerimeters = matricesDict["matrices"]
# params = paramsDict["params"]
# dropzeros!(A)
# dropzeros!(B)
# dropzeros!(C)

#%%
cellPolygons = findCellPolygons(R, A, B)
cellCentres = findCellCentresOfMass(R, A, B)
fig = Figure(size=(500,1500))
ax = Axis(fig[1,1], aspect=DataAspect())
hidedecorations!(ax)
hidespines!(ax)
for i=1:size(B,1)
    poly!(ax, cellPolygons[i], color=(getRandomColor(i), 0.5), strokecolor=(:black, 1.0), strokewidth=2)
end
scatter!(ax, Point{2,Float64}.(cellCentres), color=(:black,1.0), markersize=10)
annotations!(ax, string.(collect(1:params.nCells)), Point{2,Float64}.(cellCentres), fontsize=12, color=(:black,1.0))
display(fig)

ablatedCells = []
# ablatedCells = [176, 20, 22, 92, 160, 155, 6, 129, 100, 24, 8, 7, 97, 174, 154, 21, 5, 104, 145, 189]
# ablatedCells = [58, 140, 42, 40, 129, 38, 103, 174, 97, 23, 41, 39, 154, 155, 170, 107, 98, 43, 169, 116, 18, 145, 56]

ablatedR, ablatedA, ablatedB = ablateCells(R, A, B, ablatedCells)
cellPolygonsAblated = findCellPolygons(ablatedR, ablatedA, ablatedB)
cellCentresAblated = findCellCentresOfMass(ablatedR, ablatedA, ablatedB)

ax2 = Axis(fig[2,1], aspect=DataAspect())
for i=1:size(ablatedB,1)
    poly!(ax2, cellPolygonsAblated[i], color=(getRandomColor(i), 0.5), strokecolor=(:black, 1.0), strokewidth=2)
end
scatter!(ax2, Point{2,Float64}.(cellCentresAblated), color=(:black,1.0), markersize=10)
# scatter!(ax2, Point{2,Float64}.(ablatedR), color=(:black,1.0), markersize=10)
annotations!(ax2, string.(collect(1:size(B,1)))[Not(ablatedCells)], Point{2,Float64}.(cellCentresAblated), fontsize=12, color=(:black,1.0))
hidedecorations!(ax2)
hidespines!(ax2)
display(fig)


#%%

integ2 = vertexModel(solver=Vern9(), abstol = 1e-11, reltol = 1e-9, initialSystem="argument", divisionToggle=0, R_in=ablatedR, A_in=ablatedA, B_in=ablatedB, pressureExternal=params.pressureExternal, nCycles=1.0, outputToggle=0, frameDataToggle=0, frameImageToggle=0, videoToggle=0)
 
#%%
ablatedGrownR = reinterpret(SVector{2,Float64}, integ2.u) 
params2, matrices2 = integ2.p
ablatedGrownA = matrices2.A
ablatedGrownB = matrices2.B
ablatedGrownF = matrices2.F
ablatedGrownCellTensions = matrices2.cellTensions
ablatedGrownCellPressures = matrices2.cellPressures
ablatedGrownCellPerimeters = matrices2.cellPerimeters
ablatedGrownCellAreas = matrices2.cellAreas

cellPolygons2 = findCellPolygons(ablatedGrownR, ablatedGrownA, ablatedGrownB)
cellCentres2 = findCellCentresOfMass(ablatedGrownR, ablatedGrownA, ablatedGrownB)

ax3 = Axis(fig[3,1], aspect=DataAspect())
hidedecorations!(ax3)
hidespines!(ax3)
for i=1:size(ablatedGrownB,1)
    poly!(ax3, cellPolygons2[i], color=(getRandomColor(i), 0.5), strokecolor=(:black, 1.0), strokewidth=2)
end
scatter!(ax3, Point{2,Float64}.(cellCentres2), color=(:black,1.0), markersize=10)
annotations!(ax3, string.(collect(1:size(B,1)))[Not(ablatedCells)], Point{2,Float64}.(cellCentresAblated), fontsize=12, color=(:black,1.0))
# annotations!(ax3, string.(collect(1:size(ablatedGrownB,1))), Point{2,Float64}.(cellCentres2), fontsize=12, color=(:black,1.0))
display(fig)

# jldsave(datadir("$(Dates.format(Dates.now(),"yy-mm-dd-HH-MM-SS"))_testSystem.jld2"); ablatedR,
#     ablatedA,
#     ablatedB,
#     ablatedGrownR,
#     ablatedGrownA, 
#     ablatedGrownB, 
#     ablatedGrownF, 
#     ablatedGrownCellTensions, 
#     ablatedGrownCellPressures, 
#     ablatedGrownCellPerimeters, 
#     ablatedGrownCellAreas)

R = ablatedGrownR
A = ablatedGrownA
B = ablatedGrownB
F = ablatedGrownF
cellTensions = ablatedGrownCellTensions
cellPressures = ablatedGrownCellPressures
cellPerimeters = ablatedGrownCellPerimeters
cellAreas = ablatedGrownCellAreas

jldsave(datadir("Voronoi_testSystem4.jld2"); R,
    A, 
    B, 
    F, 
    cellTensions, 
    cellPressures, 
    cellPerimeters, 
    cellAreas)
