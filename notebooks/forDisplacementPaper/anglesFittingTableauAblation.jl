using DrWatson
using DiscreteCalculus
using CairoMakie; CairoMakie.activate!()
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
@from "$(srcdir("AblateCells.jl"))" using AblateCells
@from "$(srcdir("Stresses.jl"))" using Stresses

include(projectdir("notebooks", "forDisplacementPaper", "setupScript.jl"))

nPanels = 24

fig = Figure()
axes = []
gridlayouts = []

for ablatedCell in (ablatedCells[sortedOrder])[1:min(nPanels, length(ablatedCells))]
    
    @show ablatedCell

    push!(gridlayouts, GridLayout(parent=fig))
    push!(axes, Axis(gridlayouts[end][1,1], aspect=DataAspect()))
    hidedecorations!(axes[end]); hidespines!(axes[end])
    
    importedDataAblated = load(datadir("displacementFields", dateString, "$(dateString)_Ablated$(ablatedCell).jld2");
                    typemap=Dict("VertexModel.../VertexModelContainers.jl.VertexModelContainers.MatricesContainer" => MatricesContainer, 
                                "VertexModel.../VertexModelContainers.jl.VertexModelContainers.ParametersContainer" => ParametersContainer
                    )
                )
                
    @unpack Rablated, Aablated, Bablated, Fablated, ablationCOM = importedDataAblated

    maxF = maximum(norm.(sum(Fablated, dims=2)))
    @show maxF

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

    # Monolayer visualisation
    for i=1:size(Bablated,1)
        if i∈centralcells
            poly!(axes[end], cellPolygonsAblated[i], color=directionsAblated[i], colormap=:managua, colorrange=(-1.0,1.0), strokewidth=1, strokecolor=(:black,0.1))
        else
            poly!(axes[end], cellPolygonsAblated[i], color=(:black,0.1), strokewidth=1, strokecolor=(:black,0.1))
        end
    end

    # Resize axis panel
    axwidth, axheight = widths(axes[end].scene.viewport[])
    axes[end].width = axwidth
    axes[end].height = axheight

    push!(axes, Axis(gridlayouts[end][1,2]))
    ylims!(axes[end], (-1.0,1.0))

    # Raw data
    directionsTruncated = directionsAblated[centralcells]
    directionsSorted = directionsTruncated[sortperm(θsTruncated)]
    scatter!(axes[end], θsSorted, directionsSorted, color=:red)
    eigenVals, eigenVecs = eigen(matrices.cellShapeTensor[ablatedCell]) 
    divθ = atan(abs(eigenVecs[1,1]), eigenVecs[2,1])
    vlines!(axes[end], [divθ, divθ-π])

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

    dummyθs = collect(-π:0.01:π)
    bestFitDirections = ϕ(dummyθs, exportParams)
    # bestFitDirections = ϕ(dummyθs, sol.u)
    lines!(axes[end], dummyθs, bestFitDirections, linewidth=2, color=(:blue,0.75))

    PeffString = @sprintf("%.5f", Peffs_all[ablatedCell])
    Label(gridlayouts[end][2,:], L"Cell\ %$(ablatedCell),\ p_{eff}=%$PeffString", fontsize=10)

    colsize!(gridlayouts[end], 1, Relative(0.6))
    colsize!(gridlayouts[end], 2, Aspect(1,1.0))

end

for i=1:min(nPanels, length(ablatedCells))
    fig[fld1(i,4), mod1(i,4)] = gridlayouts[i]
end
colsize!(fig.layout, 1, Aspect(1, 2.0))
colsize!(fig.layout, 2, Aspect(1, 2.0))
colsize!(fig.layout, 3, Aspect(1, 2.0))
colsize!(fig.layout, 4, Aspect(1, 2.0))


resize_to_layout!(fig)
display(fig)
dateStringOut = "$(Dates.format(Dates.now(),"yy-mm-dd-HH-MM-SS"))"
save(datadir("displacementFields", dateString, "tableauDirections$(dateStringOut)$(order).pdf"), fig)




    # p0 = [aOverb, divθ]
    # p0 = [0.0,0.0]
    # p0 = rand(2)
    # p0 = rand(5)
    # p0 = [0.1,0.1,0.1,0.1,0.1]
    # lb = [-Inf, 0.0, -π/2, 0.0, -π/2]
    # lb = [-Inf, 0.0, -π/2, 0.0, -π/2]
    # ub = [Inf, Inf, π/2, Inf, π/2]
    # ub = [Inf, Inf, π/2, Inf, π/2]
    # localFit = curve_fit(ϕ, θsSorted, directionsSorted, p0, lower=lb, upper=ub)
    # localFit = curve_fit(ϕ, θsSorted, directionsSorted, p0)

    # u0 = [0.1,0.1,0.1,0.1,0.1]
    # lbound = [-Inf, 0.0, -π/2, 0.0, -π/2]
    # ubound = [Inf, Inf, π/2, Inf, π/2]
    # prob = NonlinearLeastSquaresProblem{isinplace}(NonlinearFunction(f), u0, [θsSorted, directionsSorted]; lbound, ubound)
