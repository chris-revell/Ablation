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
using DataFrames
using CSV
@from "$(srcdir("AblateCells.jl"))" using AblateCells
@from "$(srcdir("Stresses.jl"))" using Stresses

include(projectdir("notebooks", "forDisplacementPaper", "setupScript.jl"))  

# data = load(datadir("displacementFields", dateString, "fittingParamsAblation.jld2"))
# @unpack fittingParams, divθs, Peffs_all, ζᵢ, as, bs, cs, ds, es, divθsVec, ablatedCells = data

df = DataFrame(CSV.File(datadir("displacementFields", dateString, "fittingParamsAblation.csv")))


# maxima = [findmax(getindex.(paramTuples,m))[2] for m=1:5]
# cellsToUse = ablatedCells[Not(maxima)]

fig = Figure(size=(1000,1000), fontsize=12)
axes = Axis[]
parameterLetters = [L"a", L"b", L"d"]

# residualScaled = df[!, :Δ]./maximum(df[!,:Δ])
# colours = [(:red, 1.0.-r) for r in residualScaled]

opacities2 = minimum(df[!, :Δ])./df[!, :Δ]

# colours = [(:red, o) for o in opacities2]
colours = [(:red, 1.0) for o in opacities2]

for (i,p) in enumerate([1, 2, 4])
    push!(axes, Axis(fig[i, 1]))
    scatter!(axes[end], df[!,:ζᵢ], df[!,p+1], color=colours)
    axes[end].xlabel = L"\zeta_i"
    axes[end].ylabel = parameterLetters[i]

    push!(axes, Axis(fig[i, 2]))
    scatter!(axes[end], df[!,:Peff], df[!,p+1], color=colours)
    axes[end].xlabel = L"P_{eff,i}"
    axes[end].ylabel = parameterLetters[i]

    push!(axes, Axis(fig[i, 3]))
    scatter!(axes[end], df[!,:r], df[!,p+1], color=colours)
    axes[end].xlabel = L"Distance\ from\ periphery"
    axes[end].ylabel = parameterLetters[i]

    push!(axes, Axis(fig[i, 4]))
    scatter!(axes[end], df[!, :Aᵢ], df[!,p+1], color=colours)
    axes[end].xlabel = L"Cell\ area"
    axes[end].ylabel = parameterLetters[i]

    push!(axes, Axis(fig[i, 5]))
    scatter!(axes[end], df[!, :Lᵢ]./sqrt.(df[!, :Aᵢ]), df[!,p+1], color=colours)
    axes[end].xlabel = L"L_i / \sqrt A_i"
    axes[end].ylabel = parameterLetters[i]
end

# for a=6:15
#     ylims!(axes[a], (0.0,2.0))
# end
    
parameterLetters = [L"c", L"e"]
push!(axes, Axis(fig[4, 1]))
scatter!(axes[end], df[!,:divθ], df[!, :c], color=colours)
axes[end].xlabel = L"Cell\ short\ axis\ orientation"
axes[end].ylabel = parameterLetters[1]
push!(axes, Axis(fig[4, 2]))
scatter!(axes[end], df[!,:divθ], df[!, :e], color=colours)
axes[end].xlabel = L"Cell\ short\ axis\ orientation"
axes[end].ylabel = parameterLetters[2]

push!(axes, Axis(fig[5, 1]))
scatter!(axes[end], df[!,:c], df[!, :e], color=colours)
axes[end].xlabel = L"c"
axes[end].ylabel = L"e"


# # Resize axis panel
# axwidth, axheight = widths(axes[1].scene.viewport[])
# for ax in axes
#     ax.width = axwidth
#     ax.height = axheight
# end

colsize!(fig.layout, 1, Aspect(1, 1.0))
colsize!(fig.layout, 2, Aspect(1, 1.0))
colsize!(fig.layout, 3, Aspect(1, 1.0))
colsize!(fig.layout, 4, Aspect(1, 1.0))
colsize!(fig.layout, 5, Aspect(1, 1.0))

resize_to_layout!(fig)
display(fig)

dateStringOut = "$(Dates.format(Dates.now(),"yy-mm-dd-HH-MM-SS"))"
save(datadir("displacementFields", dateString, "fittingScatterPlots$(dateStringOut).pdf"), fig)

