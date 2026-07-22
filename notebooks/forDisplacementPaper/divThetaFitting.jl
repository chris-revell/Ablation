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
using CircularArrays
using StatsBase
using LsqFit
@from "$(srcdir("AblateCells.jl"))" using AblateCells
@from "$(srcdir("Stresses.jl"))" using Stresses

CairoMakie.activate!()

include(projectdir("notebooks", "forDisplacementPaper", "dateStringDefinition.jl"))

fittingParams = []
divθs = []

for (i, ablatedCell) in enumerate(testCells)
    
    @show ablatedCell
    
    importedDataAblated = load(datadir("displacementFields", dateString, "$(dateString)_Ablated$(ablatedCell).jld2");
                    typemap=Dict("VertexModel.../VertexModelContainers.jl.VertexModelContainers.MatricesContainer" => MatricesContainer, 
                                "VertexModel.../VertexModelContainers.jl.VertexModelContainers.ParametersContainer" => ParametersContainer
                    )
                )
                
    @unpack Rablated, Aablated, Bablated, Fablated, ablationCOM = importedDataAblated
    cellCentresAblated = findCellCentresOfMass(Rablated, Aablated, Bablated)
    cellPolygonsAblated = findCellPolygons(Rablated, Aablated, Bablated)
    Δrᵢ = cellCentresAblated.-cellCentres[Not(ablatedCell)]
    radiusVectorsAblated = ([cellCentres[i].-ablationCOM for i=1:size(matrices.B,1)])[Not(ablatedCell)]
    directionsAblated = normalize.(Δrᵢ).⋅normalize.(radiusVectorsAblated)
    θs = atan.(getindex.(radiusVectorsAblated, 1), getindex.(radiusVectorsAblated, 2))
    centralcells = findall(x->norm(x)<0.33*maximum(norm.(radiusVectorsAblated)), radiusVectorsAblated)
    θsTruncated = θs[centralcells]
    θsSorted = CircularArray(sort(θsTruncated))
    radiusVectorsSorted = CircularArray(radiusVectorsAblated[sortperm(θsTruncated)])

    # Raw data
    directionsTruncated = directionsAblated[centralcells]
    directionsSorted = CircularArray(directionsTruncated[sortperm(θsTruncated)])
    
    eigenVals, eigenVecs = eigen(matrices.cellShapeTensor[ablatedCell]) 
    divθ = atan(eigenVecs[1,1], eigenVecs[2,1])
    # p0 = [aOverb, divθ]
    # p0 = [0.0,0.0]
    # p0 = rand(2)
    p0 = rand(5)

    localFit = curve_fit(ϕ, θsSorted, directionsSorted, p0)

    bestFitDirections = ϕ(θsSorted, localFit.param)

    push!(fittingParams, localFit.param)
    push!(divθs, divθ)

end

fig2 = Figure()
ax2 = Axis(fig2[1,1])
divθsRestricted = (divθs.+2*π).%2.0*π
fittedDivθs = (getindex.(fittingParams, 5).+2*π).%2.0*π
scatter!(ax2, divθsRestricted, fittedDivθs, color=:red)
lines!(ax2, [0.0,2π], [0.0, 2π])
lines!(ax2, [0.0,2π], [0.0, 2π].-π/2)
lines!(ax2, [0.0,2π], [0.0, 2π].-π)
# scatter!(ax2, (divθs.+2*π).%2.0*π, (getindex.(fittingParams, 3).+2*π).%2.0*π, color=:green)
display(fig2)
dateStringOut = "$(Dates.format(Dates.now(),"yy-mm-dd-HH-MM-SS"))"
save(datadir("displacementFields", dateString, "divThetaFitting$(dateStringOut).pdf"), fig2)
