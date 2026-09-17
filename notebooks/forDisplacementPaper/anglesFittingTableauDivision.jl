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
@from "$(srcdir("AblateCells.jl"))" using AblateCells
@from "$(srcdir("Stresses.jl"))" using Stresses

include(projectdir("notebooks", "forDisplacementPaper", "setupScript.jl"))

nPanels = 12
fig = Figure()
axes = []
gridlayouts = []

fittingParams = []
divθs = []

if order == "PeffRising"
    sortedOrder = sortperm(Peffs_all[dividedCells])
elseif order == "PeffFalling"    
    sortedOrder = sortperm(Peffs_all[dividedCells], rev=true)
elseif order=="Radius"     
    sortedOrder = sortperm(radii_all[dividedCells])
end

for dividedCell in (dividedCells[sortedOrder])[1:min(nPanels, length(dividedCells))]
    
    @show dividedCell

    push!(gridlayouts, GridLayout(parent=fig))
    push!(axes, Axis(gridlayouts[end][1,1], aspect=DataAspect()))
    hidedecorations!(axes[end]); hidespines!(axes[end])
    
    importedDataDivided = load(datadir("displacementFields", dateString, "$(dateString)_Divided$(dividedCell).jld2");
                    typemap=Dict("VertexModel.../VertexModelContainers.jl.VertexModelContainers.MatricesContainer" => MatricesContainer, 
                                "VertexModel.../VertexModelContainers.jl.VertexModelContainers.ParametersContainer" => ParametersContainer
                    )
                )
                
    @unpack Rdivided, Adivided, Bdivided, Fdivided, divisionCOM = importedDataDivided
    Inew = size(Bdivided, 1)
    cellCentresDivided = findCellCentresOfMass(Rdivided, Adivided, Bdivided)
    cellPolygonsDivided = findCellPolygons(Rdivided, Adivided, Bdivided)
    Δrᵢ = cellCentresDivided[1:end-1].-cellCentres
    radiusVectorsDivided = ([cellCentres[i].-divisionCOM for i=1:size(matrices.B,1)])
    directionsDivided = normalize.(Δrᵢ).⋅normalize.(radiusVectorsDivided)
    θs = atan.(getindex.(radiusVectorsDivided, 1), getindex.(radiusVectorsDivided, 2))
    centralcells = [i for i in findall(x->norm(x)<centralCellThreshold*maximum(norm.(radiusVectorsDivided)), radiusVectorsDivided) if i∉[dividedCell, Inew]]
    
    # Monolayer visualisation
    for i=1:size(Bdivided,1)
        if i∈[dividedCell, Inew]
            poly!(axes[end], cellPolygonsDivided[i], color=(:black,0.1), strokewidth=1, strokecolor=(:black,0.1))
        elseif i∈centralcells
            poly!(axes[end], cellPolygonsDivided[i], color=directionsDivided[i], colormap=:managua, colorrange=(-1.0,1.0), strokewidth=1, strokecolor=(:black,0.1))
        else
            poly!(axes[end], cellPolygonsDivided[i], color=(:black,0.1), strokewidth=1, strokecolor=(:black,0.1))
        end
    end

    # Resize axis panel
    axwidth, axheight = widths(axes[end].scene.viewport[])
    axes[end].width = axwidth
    axes[end].height = axheight

    push!(axes, Axis(gridlayouts[end][1,2]))
    ylims!(axes[end], (-1.0,1.0))

    # Raw data
    θsTruncated = θs[centralcells]
    θsSorted = sort(θsTruncated)
    radiusVectorsSorted = radiusVectorsDivided[sortperm(θsTruncated)]

    directionsTruncated = directionsDivided[centralcells]
    directionsSorted = directionsTruncated[sortperm(θsTruncated)]
    scatter!(axes[end], θsSorted, directionsSorted, color=:red)
    eigenVals, eigenVecs = eigen(matrices.cellShapeTensor[dividedCell]) 
    divθ = atan(eigenVecs[1,1], eigenVecs[2,1])
    # p0 = [aOverb, divθ]
    # p0 = [0.0,0.0]
    # p0 = rand(2)
    p0 = rand(5)
    localFit = curve_fit(ϕ, θsSorted, directionsSorted, p0)

    bestFitDirections = ϕ(θsSorted, localFit.param)
    lines!(axes[end], θsSorted, bestFitDirections, linewidth=2, color=(:blue,0.75))

    PeffString = @sprintf("%.5f", Peffs_all[dividedCell])
    Label(gridlayouts[end][2,:], L"Cell\ %$(dividedCell),\ p_{eff}=%$PeffString", fontsize=10)

    colsize!(gridlayouts[end], 1, Relative(0.6))
    colsize!(gridlayouts[end], 2, Aspect(1,1.0))

    # push!(fittingParams, localFit.param)
    # push!(divθs, divθ)

end

for i=1:min(nPanels, length(dividedCells))
    fig[fld1(i,2), mod1(i,2)] = gridlayouts[i]
end

resize_to_layout!(fig)
display(fig)
dateStringOut = "$(Dates.format(Dates.now(),"yy-mm-dd-HH-MM-SS"))"
save(datadir("displacementFields", dateString, "tableauDirectionsDivided$(dateStringOut)$(order).pdf"), fig)
