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

include(projectdir("notebooks", "forDisplacementPaper", "setupScript.jl"))

fig = Figure()
axes = []
gridlayouts = []

fittingParams = []

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
    cellCentresAblated = findCellCentresOfMass(Rablated, Aablated, Bablated)
    cellPolygonsAblated = findCellPolygons(Rablated, Aablated, Bablated)
    Δrᵢ = cellCentresAblated.-cellCentres[Not(ablatedCell)]
    radiusVectorsAblated = ([cellCentres[i].-ablationCOM for i=1:size(matrices.B,1)])[Not(ablatedCell)]
    directionsAblated = normalize.(Δrᵢ).⋅normalize.(radiusVectorsAblated)
    θs = atan.(getindex.(radiusVectorsAblated, 1), getindex.(radiusVectorsAblated, 2))
    centralcells = findall(x->norm(x)<centralCellThreshold*maximum(norm.(radiusVectorsAblated)), radiusVectorsAblated)
    θsTruncated = θs[centralcells]
    θsSorted = CircularArray(sort(θsTruncated))
    radiusVectorsSorted = CircularArray(radiusVectorsAblated[sortperm(θsTruncated)])

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
    scatter!(axes[end], log10.(norm.(radiusVectorsAblated)), log10.(norm.(Δrᵢ)), color=:red, markersize=3)
    lines!(axes[end], [0.0,1.0], [-2.0,-3.0], color=:black, linewidth=1)
    lines!(axes[end], [0.0,1.0], [-2.0,-4.0], color=:black, linewidth=1)

    p0 = rand(3)
    localFit = curve_fit(r, norm.(radiusVectorsAblated), norm.(Δrᵢ), p0)
    bestFitDisplacements = r(norm.(radiusVectorsAblated), localFit.param)
    lines!(axes[end], log10.(norm.(radiusVectorsAblated)), log10.(bestFitDisplacements), color=(:blue,1.0), linewidth=1)

    push!(fittingParams, localFit.param)

    PeffString = @sprintf("%.5f", Peffs_all[ablatedCell])
    Label(gridlayouts[end][2,:], L"Cell\ %$(ablatedCell),\ p_{eff}=%$PeffString", fontsize=10)
    
    colsize!(gridlayouts[end], 1, Relative(0.6))
    colsize!(gridlayouts[end], 2, Aspect(1,1.0))

end

for i=1:min(nPanels, length(ablatedCells))
    fig[fld1(i,2), mod1(i,2)] = gridlayouts[i]
end

resize_to_layout!(fig)
display(fig)
dateStringOut = "$(Dates.format(Dates.now(),"yy-mm-dd-HH-MM-SS"))"
save(datadir("displacementFields", dateString, "tableauMagnitudes$(dateStringOut)$(order).pdf"), fig)