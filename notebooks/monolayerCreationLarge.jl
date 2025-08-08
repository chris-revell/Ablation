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

integ = vertexModel(abstol = 1e-8,
                    reltol = 1e-8,
                    nRows=5,
                    nCycles=6,
                    printToggle=1,
                    frameDataToggle=0,
                    frameImageToggle=0,
                    videoToggle=0,
                    setRandomSeed=123,
                    divisionToggle=1,
                )

R = reinterpret(SVector{2,Float64}, integ.u) 
params, matrices = integ.p
@unpack A, B = matrices 

integ2 = vertexModel(abstol = 1e-8,
                    reltol = 1e-8,
                    sstol = 3e-5,
                    initialSystem="argument",
                    divisionToggle=0,
                    R_in=R,
                    A_in=A,
                    B_in=B,
                    pressureExternal=0.0,
                    printToggle=1,
                    frameDataToggle=0,
                    frameImageToggle=0,
                    videoToggle=0,
                    termSteadyState=true,
                )

R = reinterpret(SVector{2,Float64}, integ2.u) 
params, matrices = integ2.p
jldsave(datadir("Large_testSystem.jld2"); R,
    matrices.A, 
    matrices.B, 
    matrices.F, 
    matrices.cellTensions, 
    matrices.cellPressures, 
    matrices.cellPerimeters, 
    matrices.cellAreas
)
