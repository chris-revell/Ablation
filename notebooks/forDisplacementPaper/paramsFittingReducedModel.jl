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
using NonlinearSolve
using DataFrames
using CSV
@from "$(srcdir("AblateCells.jl"))" using AblateCells
@from "$(srcdir("Stresses.jl"))" using Stresses

CairoMakie.activate!()

include(projectdir("notebooks", "forDisplacementPaper", "setupScript.jl"))

is = Int64[]
as = Float64[]
bs = Float64[]
cs = Float64[]
divθs = Float64[]
rs = Float64[]
Δs = Float64[]

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
    centralcells = findall(x->norm(x)<centralCellThreshold*maximum(norm.(radiusVectorsAblated)), radiusVectorsAblated)
    θsTruncated = θs[centralcells]
    θsSorted = sort(θsTruncated)
    radiusVectorsSorted = radiusVectorsAblated[sortperm(θsTruncated)]
    # Raw data
    directionsTruncated = directionsAblated[centralcells]
    directionsSorted = directionsTruncated[sortperm(θsTruncated)]
    eigenVals, eigenVecs = eigen(matrices.cellShapeTensor[ablatedCell]) 
    divθ = atan(eigenVecs[1,1], eigenVecs[2,1])

    lowerbounds = [-Inf, -Inf, -π/2]
    upperbounds = [Inf, Inf, π/2]
    u0 = zeros(3)
    p = hcat(θsSorted, directionsSorted)
    nlls_prob = NonlinearLeastSquaresProblem(lossFunctionReduced, u0, p; ub=upperbounds, lb=lowerbounds)
    sol = solve(
        nlls_prob, 
        LevenbergMarquardt(); #GaussNewton()
        maxiters = 1000, #show_trace = Val(true),
        trace_level = TraceWithJacobianConditionNumber(25),
    )

    exportParams = copy(sol.u)
    if exportParams[2]<0
        exportParams[2] = -exportParams[2]
        exportParams[3] = exportParams[3]-sign(exportParams[3])*π
    end
    
    iᵖinds = findall(x->x!=0, findPeripheralCells(matrices.B))
    rs_local = norm.([R[i] .- ablationCOM for i in iᵖinds])
    distanceToPeriphery = findmin(rs_local)[2]

    dummyθs = collect(-π:0.01:π)
    bestFitDirections = ϕReduced(θsSorted, exportParams)
    Δslocal = (directionsSorted.-bestFitDirections).^2

    push!(is, ablatedCell)
    push!(as, exportParams[1])
    push!(bs, exportParams[2])
    push!(cs, exportParams[3])
    push!(divθs, divθ)
    push!(rs, distanceToPeriphery)
    push!(Δs, sum(Δslocal)/length(θsSorted))

end

df = DataFrame(i=is,
    a = as, 
    b = bs, 
    c = cs, 
    divθ = divθs,
    r = rs,
    ζᵢ = ζᵢ[is],
    Peff = Peffs_all[is],
    Aᵢ = matrices.cellAreas[is],
    Lᵢ = matrices.cellPerimeters[is],
    Δ = Δs,
)

# jldsave(datadir("displacementFields", dateString, "fittingParamsAblation.jld2"); df)
CSV.write(datadir("displacementFields", dateString, "fittingParamsAblationReduced.csv"), df)
println("Saved at $(datadir("displacementFields", dateString, "fittingParamsAblationReduced.csv"))")
println("Done")