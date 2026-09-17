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
ds = Float64[]
es = Float64[]
divθs = Float64[]
rs = Float64[]
Δs = Float64[]

for dividedCell in dividedCells

    @show dividedCell

    importedDataDivided = load(datadir("displacementFields", dateString, "$(dateString)_Divided$(dividedCell).jld2");
                    typemap=Dict("VertexModel.../VertexModelContainers.jl.VertexModelContainers.MatricesContainer" => MatricesContainer, 
                                "VertexModel.../VertexModelContainers.jl.VertexModelContainers.ParametersContainer" => ParametersContainer
                    )
                )          
    @unpack Rdivided, Adivided, Bdivided, Fdivided, divisionCOM = importedDataDivided
    
    Idivided = size(Bdivided, 1)
    cellCentresDivided = findCellCentresOfMass(Rdivided, Adivided, Bdivided)
    cellPolygonsDivided = findCellPolygons(Rdivided, Adivided, Bdivided)
    Δrᵢ = cellCentresDivided[1:end-1].-cellCentres
    radiusVectorsDivided = ([cellCentres[i].-divisionCOM for i=1:size(matrices.B,1)])
    directionsDivided = normalize.(Δrᵢ).⋅normalize.(radiusVectorsDivided)
    θs = atan.(getindex.(radiusVectorsDivided, 1), getindex.(radiusVectorsDivided, 2))
    centralcells = [i for i in findall(x->norm(x)<centralCellThreshold*maximum(norm.(radiusVectorsDivided)), radiusVectorsDivided) if i∉[dividedCell, Idivided]]
    θsTruncated = θs[centralcells]
    θsSorted = sort(θsTruncated)
    radiusVectorsSorted = radiusVectorsDivided[sortperm(θsTruncated)]
    directionsTruncated = directionsDivided[centralcells]
    directionsSorted = directionsTruncated[sortperm(θsTruncated)]
    eigenVals, eigenVecs = eigen(matrices.cellShapeTensor[dividedCell]) 
    divθ = atan(eigenVecs[1,1], eigenVecs[2,1])

    lowerbounds = [-Inf, -Inf, -π/2, -Inf, -π/2]
    upperbounds = [Inf, Inf, π/2, Inf, π/2]
    u0 = zeros(5)
    p = hcat(θsSorted, directionsSorted)
    nlls_prob = NonlinearLeastSquaresProblem(lossFunction, u0, p; ub=upperbounds, lb=lowerbounds)
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
    if exportParams[4]<0
        exportParams[4] = -exportParams[4]
        exportParams[5] = exportParams[5]-sign(exportParams[5])*π
    end

    iᵖinds = findall(x->x!=0, findPeripheralCells(matrices.B))
    rs_local = norm.([R[i] .- divisionCOM for i in iᵖinds])
    distanceToPeriphery = findmin(rs_local)[2]

    dummyθs = collect(-π:0.01:π)
    bestFitDirections = ϕ(θsSorted, exportParams)
    Δslocal = (directionsSorted.-bestFitDirections).^2

    push!(is, dividedCell)
    push!(as, exportParams[1])
    push!(bs, exportParams[2])
    push!(cs, exportParams[3])
    push!(ds, exportParams[4])
    push!(es, exportParams[5])
    push!(divθs, divθ)
    push!(rs, distanceToPeriphery)
    push!(Δs, sum(Δslocal)/length(θsSorted))

end

df = DataFrame(i=is,
    a = as, 
    b = bs, 
    c = cs, 
    d = ds, 
    e = es,
    divθ = divθs,
    r = rs,
    ζᵢ = ζᵢ[is],
    Peff = Peffs_all[is],
    Aᵢ = matrices.cellAreas[is],
    Lᵢ = matrices.cellPerimeters[is],
    Δ = Δs,
)

# jldsave(datadir("displacementFields", dateString, "fittingParamsDivision.jld2"); df)
CSV.write(datadir("displacementFields", dateString, "fittingParamsDivision.csv"), df)
println("Saved at $(datadir("displacementFields", dateString, "fittingParamsDivision.csv"))")
println("Done")