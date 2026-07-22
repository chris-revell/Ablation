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

# Import system data
importedData = load(datadir("displacementFields", dateString, "$(dateString)_InitialSystem.jld2");
                    typemap=Dict("VertexModel.../VertexModelContainers.jl.VertexModelContainers.MatricesContainer" => MatricesContainer, 
                                "VertexModel.../VertexModelContainers.jl.VertexModelContainers.ParametersContainer" => ParametersContainer
                    )
                )
@unpack R, params, matrices = importedData

testFiles = filter(x->(occursin("Divided",x)||occursin("Ablated",x))&&occursin(".jld2",x), readdir(datadir("displacementFields", dateString)))

dividedCells = [i[(length(dateString)+1+length("Divided"))+1:end-5] for i in testFiles if occursin("Divided", i)]
ablatedCells = [i[(length(dateString)+1+length("Ablated"))+1:end-5] for i in testFiles if occursin("Ablated", i)]
testCells = parse.(Int64, dividedCells∩ablatedCells)

cellCentres = findCellCentresOfMass(R, matrices.A, matrices.B)

𝐡 = hNetwork(R, matrices.A, matrices.B, matrices.F)
# Stress tensors 
σᵢ = σ(R, matrices.A, matrices.B, 𝐡)
# Deviatoric stress 
σDᵢ = [σᵢ[i] .- 0.5*tr(σᵢ[i]) for i=1:size(matrices.B,1)]
σDSᵢ = 0.5.*(σDᵢ .+ transpose.(σDᵢ))
# ~\ref{eq:shearstressexact}
ζᵢ = [sqrt(-det(σDSᵢ[i])) for i=1:size(matrices.B,1)]
p_eff = -0.5.*cocurlᶜ(R, matrices.A, matrices.B, 𝐡)

cellOrder = sortperm(p_eff[testCells])
orderedCells = testCells[cellOrder]

maxPanels = 25
nPanels = min(maxPanels,length(testCells))

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
