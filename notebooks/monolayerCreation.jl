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
integ = vertexModel(abstol = 1e-8,
                    reltol = 1e-8,
                    nRows=5,
                    nCycles=4,
                    printToggle=1,
                    frameDataToggle=0,
                    frameImageToggle=0,
                    videoToggle=0,
                    setRandomSeed=123,
                    divisionToggle=1,
                )
#%%

# getRandomColor(seed) = RGB(rand(Xoshiro(seed),3)...)

systems = ["NoHole", "SingleHole", "DoubleHole"]
system = systems[3]

R = reinterpret(SVector{2,Float64}, integ.u) 
params, matrices = integ.p
@unpack A, B = matrices 


cellPolygons = findCellPolygons(R, A, B)
cellCentres = findCellCentresOfMass(R, A, B)
fig = Figure(size=(500,1500))
ax = Axis(fig[1,1], aspect=DataAspect())
hidedecorations!(ax)
hidespines!(ax)
for i=1:size(B,1)
    poly!(ax, cellPolygons[i], color=(:black, 0.5), strokecolor=(:black, 1.0), strokewidth=2)
end
scatter!(ax, Point{2,Float64}.(cellCentres), color=(:black,1.0), markersize=10)
annotations!(ax, string.(collect(1:params.nCells)), Point{2,Float64}.(cellCentres), fontsize=12, color=(:black,1.0))
display(fig)

if system=="SingleHole"
    systemCOM = sum(R)./length(R)
    cellCentres = findCellCentresOfMass(R, A, B)
    centralCell = findmin(norm.([cellCentres[i].-systemCOM for i=1:size(B,1)]))[2]
    neighbourMatrix = dropzeros(B*transpose(B))
    ablatedCells = unique(findall(x->x!=0, neighbourMatrix[centralCell,:]))
    # for _ = 1:2
    #     neighbours = getindex.(findall(x->x!=0, neighbourMatrix[ablatedCells,:]),2)
    #     append!(ablatedCells, unique(neighbours))
    # end
elseif system=="DoubleHole"
    systemCOM = sum(R)./length(R)
    yrange = maximum(getindex.(R,2))-minimum(getindex.(R,2))
    holes = [systemCOM.+[0.0, yrange/5], systemCOM.+[0.0, -yrange/5]]
    cellCentres = findCellCentresOfMass(R, A, B)
    centralCells = [findmin(norm.([cellCentres[i].-hole for i=1:size(B,1)]))[2] for hole in holes]
    neighbourMatrix = dropzeros(B*transpose(B))
    ablatedCells = unique(getindex.(findall(x->x!=0, neighbourMatrix[centralCells,:]), 2))
    # for _ = 1:1
    #     neighbours = getindex.(findall(x->x!=0, neighbourMatrix[ablatedCells,:]),2)
    #     append!(ablatedCells, unique(neighbours))
    # end
else 
    ablatedCells = []
end
unique!(ablatedCells)
ablatedR, ablatedA, ablatedB = ablateCells(R, A, B, ablatedCells)
cellPolygonsAblated = findCellPolygons(ablatedR, ablatedA, ablatedB)
cellCentresAblated = findCellCentresOfMass(ablatedR, ablatedA, ablatedB)

ax2 = Axis(fig[2,1], aspect=DataAspect())
for i=1:size(ablatedB,1)
    poly!(ax2, cellPolygonsAblated[i], color=(:black, 0.5), strokecolor=(:black, 1.0), strokewidth=2)
end
scatter!(ax2, Point{2,Float64}.(cellCentresAblated), color=(:black,1.0), markersize=10)
annotations!(ax2, string.(collect(1:size(B,1)))[Not(ablatedCells)], Point{2,Float64}.(cellCentresAblated), fontsize=12, color=(:black,1.0))
hidedecorations!(ax2)
hidespines!(ax2)
display(fig)

#%%

integ2 = vertexModel(abstol = 1e-8,
                    reltol = 1e-8,
                    sstol = 1e-5,
                    initialSystem="argument",
                    divisionToggle=0,
                    R_in=ablatedR,
                    A_in=ablatedA,
                    B_in=ablatedB,
                    pressureExternal=0.0,
                    printToggle=1,
                    frameDataToggle=0,
                    frameImageToggle=0,
                    videoToggle=0,
                    termSteadyState=true,
                )

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
    poly!(ax3, cellPolygons2[i], color=(:black, 0.5), strokecolor=(:black, 1.0), strokewidth=2)
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

jldsave(datadir("$(system)_testSystem5.jld2"); R,
    A, 
    B, 
    F, 
    cellTensions, 
    cellPressures, 
    cellPerimeters, 
    cellAreas,
)

# R = reinterpret(SVector{2,Float64}, integ.u) 
# params, matrices = integ.p
# jldsave(datadir("Large_testSystem.jld2"); R,
#     matrices.A, 
#     matrices.B, 
#     matrices.F, 
#     matrices.cellTensions, 
#     matrices.cellPressures, 
#     matrices.cellPerimeters, 
#     matrices.cellAreas
# )

    # yrange = maximum(getindex.(R,2))-minimum(getindex.(R,2))
    # ablatedCells = findall(x->x<yrange/7.0, norm.([cellCentres[i].-systemCOM for i=1:size(B,1)]))
    
    # ablatedVerts = findall(x->x<yrange/9.0, norm.([R[k].-systemCOM for k=1:size(A,2)]))
    # C = findC(A,B)
    # ablatedCells = unique(getindex.(findall(x->x!=0, C[:,ablatedVerts]), 1))
# ablatedCells = findall(x->x<yrange/8.0, norm.([cellCentres[i].-holes[1] for i=1:size(B,1)]))
    # append!(ablatedCells, findall(x->x<yrange/8.0, norm.([cellCentres[i].-holes[2] for i=1:size(B,1)])))
    # unique!(ablatedCells)