
# Julia packages
using DrWatson
using FromFile
using LinearAlgebra
using JLD2
using SparseArrays
using StaticArrays
using Printf
using DifferentialEquations
using SteadyStateDiffEq
using VertexModel 
using DiscreteCalculus
using CairoMakie
# using NonlinearSolveq
using Sundials

@from "$(srcdir("AblateCells.jl"))" using AblateCells

#%%

energyModel="quadratic"

integ1 = vertexModel(abstol = 1e-9,
                    # reltol = 1e-9,
                    nRows=11,
                    nCycles=0.2,
                    printToggle=1,
                    frameDataToggle=0,
                    frameImageToggle=0,
                    videoToggle=0,
                    setRandomSeed=1234,
                    divisionToggle=1,
                    pressureExternal=0.1,
                    energyModel=energyModel
                )

(params1, matrices1) = integ1.p
R1 = reinterpret(SVector{2,Float64}, integ1.u)
@show maximum(norm.(sum(matrices1.F, dims=2)))

cellPolygons = findCellPolygons(R1, matrices1.A, matrices1.B)
cellCentres = findCellCentresOfMass(R1, matrices1.A, matrices1.B)
fig = Figure(size=(500,1500))
ax = Axis(fig[1,1], aspect=DataAspect())
hidedecorations!(ax)
hidespines!(ax)
for i=1:size(matrices1.B,1)
    poly!(ax, cellPolygons[i], color=(:black, 0.5), strokecolor=(:black, 1.0), strokewidth=2)
end
# scatter!(ax, Point{2,Float64}.(cellCentres), color=(:black,1.0), markersize=10)
# annotations!(ax, string.(collect(1:params.nCells)), Point{2,Float64}.(cellCentres), fontsize=12, color=(:black,1.0))
display(fig)

# R = reinterpret(SVector{2,Float64}, integ.u)
probSS = SteadyStateProblem(model!,
        integ1.u,
        integ1.p,
        abstol = 1e-9,
        # reltol = 1e-9,
    )

sol = solve(probSS, DynamicSS(Rodas5P()))
# sol = solve(probSS, DynamicSS(CVODE_BDF()), dt = 0.01)
R2 = reinterpret(SVector{2,Float64}, sol.u)
(params2, matrices2) = sol.prob.p
@show maximum(norm.(sum(matrices2.F, dims=2)))

cellPolygons = findCellPolygons(R2, matrices2.A, matrices2.B)
cellCentres = findCellCentresOfMass(R2, matrices2.A, matrices2.B)
# fig = Figure(size=(500,1500))
ax = Axis(fig[2,1], aspect=DataAspect())
hidedecorations!(ax)
hidespines!(ax)
for i=1:size(matrices2.B,1)
    poly!(ax, cellPolygons[i], color=(:black, 0.5), strokecolor=(:black, 1.0), strokewidth=2)
end
# scatter!(ax, Point{2,Float64}.(cellCentres), color=(:black,1.0), markersize=10)
# annotations!(ax, string.(collect(1:params.nCells)), Point{2,Float64}.(cellCentres), fontsize=12, color=(:black,1.0))
display(fig)


𝐡1 = hNetwork(R1, matrices1.A, matrices1.B, matrices1.F)
𝐡2 = hNetwork(R2, matrices2.A, matrices2.B, matrices2.F)

maximum(norm.(𝐡1.-𝐡2))
#%%

Ā = abs.(A)
B̄ = abs.(B)

ϵ = SMatrix{2, 2, Float64}([
                0.0 1.0
                -1.0 0.0
            ])

cellPositions  = findCellCentresOfMass(R, A, B)
edgeTangents   = findEdgeTangents(R, A)
edgeLengths = findEdgeLengths(R, A)
cellPerimeters = findCellPerimeterLengths(R, A, B)
cellAreas = findCellAreas(R, A, B)

# Calculate cell pressures and tensions according to energy model choice 
if energyModel == "log"
    # Model per Cowley et al. 2024 Section 2a
    # Calculate cell boundary tensions
    cellTensions = matrices2.μ .* matrices2.Γ .* matrices2.cellL₀s .* log.(cellPerimeters ./ cellL₀s)
    # Calculate cell internal pressures
    cellPressures = matrices2.μ .* matrices2.cellA₀s .* log.(cellAreas ./ matrices2.cellA₀s)
else
    # Quadratic energy model
    # Calculate cell boundary tensions
    cellTensions = matrices2.μ .* matrices2.Γ .*(cellPerimeters - matrices2.cellL₀s)
    # Calculate cell internal pressures
    cellPressures = matrices2.μ .*(cellAreas - matrices2.cellA₀s)
end

fill!(F, @SVector zeros(2))
dropzeros!(F)

for k = 1:size(A,2)
    for j in nzrange(A, k)
        for i in nzrange(B, rowvals(A)[j])
            # Force components from cell pressure perpendicular to edge tangents 
            F[k, rowvals(B)[i]] += 0.5 * cellPressures[rowvals(B)[i]] * B[rowvals(B)[i], rowvals(A)[j]] * Ā[rowvals(A)[j], k] .* (ϵ * edgeTangents[rowvals(A)[j]])
            # Force components from cell membrane tension parallel to edge tangents 
            F[k, rowvals(B)[i]] -= cellTensions[rowvals(B)[i]] * B̄[rowvals(B)[i], rowvals(A)[j]] * A[rowvals(A)[j], k] .* edgeTangents[rowvals(A)[j]] ./ edgeLengths[rowvals(A)[j]]
            # Force on vertex from external pressure 
            # externalF[k] += peripheralVertices[k] * (0.5 * pressureExternal * B[rowvals(B)[i], rowvals(A)[j]] * Ā[rowvals(A)[j], k] .* (ϵ * edgeTangents[rowvals(A)[j]])) # 0 unless peripheralVertices != 0
        end
        # Force on vertex from peripheral tension
        # externalF[k] -= peripheralEdges[rowvals(A)[j]] * peripheralTension * (peripheryLength - sqrt(π * nCells)) * A[rowvals(A)[j], k] .* edgeTangents[rowvals(A)[j]] ./ edgeLengths[rowvals(A)[j]]
    end
    
    # dR[k] = (sum(@view F[k, :]) .+ externalF[k])
end
