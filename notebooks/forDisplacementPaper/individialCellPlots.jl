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
using VertexModel

@from "$(srcdir("AblateCells.jl"))" using AblateCells
@from "$(srcdir("Stresses.jl"))" using Stresses

# dateString = "26-07-07-16-49-05"
dateString = "26-06-19-08-32-27"

fig = Figure(size=(500,500), fontsize=12)
ax = Axis(fig[1,1])

for cell in orderedCells

    empty!(ax)

    importedDataAblated = load(datadir("displacementFields", dateString, "$(dateString)_Ablated$(cell).jld2");
                    typemap=Dict("VertexModel.../VertexModelContainers.jl.VertexModelContainers.MatricesContainer" => MatricesContainer, 
                                "VertexModel.../VertexModelContainers.jl.VertexModelContainers.ParametersContainer" => ParametersContainer
                    )
                )
    @unpack Rablated, Aablated, Bablated, Fablated, ablationCOM = importedDataAblated
    
    cellCentresAblated = findCellCentresOfMass(Rablated, Aablated, Bablated)
    cellPolygonsAblated = findCellPolygons(Rablated, Aablated, Bablated)
    Δrᵢ = cellCentresAblated.-cellCentres[Not(cell)]
    radiusVectorsAblated = [cellCentres[i].-ablationCOM for i=1:size(matrices.B,1)]
    directionsAblated = normalize.(Δrᵢ).⋅normalize.(radiusVectorsAblated[Not(cell)])

    p_effString = @sprintf("%.3f", p_eff[cell])

    for i=1:size(Bablated,1)
        poly!(ax, cellPolygonsAblated[i], color=directionsAblated[i], colormap=:managua, colorrange=(-1.0,1.0), strokewidth=1, strokecolor=(:black,0.1))
    end
    
    ax.xlabel = L"Cell\ %$(cell),\ P_{eff}=%$(p_effString)"

    save(datadir("displacementFields", dateString, "cell$(cell).png"), fig)
    # save(plotsdir("displacementFields", dateString, "cell$(cell).pdf"), fig)

end
