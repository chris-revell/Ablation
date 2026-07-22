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

fittingParams = Dict()
divθs = Dict()

for ablatedCell in ablatedCells
    
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

    fittingParams[ablatedCell] = localFit.param
    divθs[ablatedCell] = divθ

end

jldsave(datadir("displacementFields", dateString, "fittingParamsAblation.jld2"); fittingParams, divθs, Peffs_all, ζᵢ)