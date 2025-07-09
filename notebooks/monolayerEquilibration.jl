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

getRandomColor(seed) = RGB(rand(Xoshiro(seed),3)...)

#%%
inFile = datadir("referenceSystems", "Large_testSystem.jld2")
importedData = load(inFile)
R_in = importedData["R"]
A_in = importedData["A"]
B_in = importedData["B"]
F_in = importedData["F"]

integ = vertexModel(abstol = 1e-6,
                    reltol = 1e-6,
                    initialSystem="argument",
                    divisionToggle=0,
                    R_in=R_in,
                    A_in=A_in,
                    B_in=B_in,
                    pressureExternal=0.0,
                    nCycles=0.5,
                    outputToggle=0,
                    frameDataToggle=0,
                    frameImageToggle=0,
                    videoToggle=0,
                    printToggle=1,
                    termSteadyState=true,
                )

@show maximum(norm.(sum(F, dims=2)))

R = reinterpret(SVector{2,Float64}, integ.u) 
params, matrices = integ.p
jldsave(datadir("LargeEquilibrated_testSystem.jld2"); R,
    matrices.A, 
    matrices.B, 
    matrices.F, 
    matrices.cellTensions, 
    matrices.cellPressures, 
    matrices.cellPerimeters, 
    matrices.cellAreas
)
