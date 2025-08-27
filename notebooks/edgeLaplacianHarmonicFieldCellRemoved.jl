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

@from "$(srcdir("AblateCells.jl"))" using AblateCells

# ϵᵢ = SMatrix{2, 2, Float64}([
#                 0.0 1.0
#                 -1.0 0.0
#             ])
# ϵₖ = SMatrix{2, 2, Float64}([
#                 0.0 -1.0
#                 1.0 0.0
#             ])

inputSystem = "Large3"

inFile = datadir("referenceSystems", "quadraticPotentialNoPressure", "$(inputSystem)Ablated_testSystem.jld2")
importedData = load(inFile)
@unpack R2, A2, B2, F2, systemCOM2 = importedData

Lprimal = edgeLaplacianPrimal(R2, A2, B2)
Ldual = edgeLaplacianDual(R2, A2, B2)
eigenvectors_Lprimal = [col for col in  eachcol((eigen(Matrix(Lprimal))).vectors)]
eigenvalues_Lprimal = (eigen(Matrix(Lprimal))).values
eigenvectors_Ldual = [col for col in  eachcol((eigen(Matrix(Ldual))).vectors)]
eigenvalues_Ldual = (eigen(Matrix(Ldual))).values

𝐜ⱼ = findEdgeMidpoints(R2, A2)
𝐂ⱼ = findCellLinkMidpoints(R2, A2, B2)
boundaryEdges = findBoundaryEdges(B2)
# primalBasisParallel = findEdgeTangents(R2, A2)./(findEdgeLengths(R2, A2).^2)
# primalBasisPerp = [ϵᵢ*v for v in primalBasisParallel]
# dualBasisParallel = findCellLinks(R2, A2, B2)./(findCellLinkLengths(R2, A2, B2).^2)
# dualBasisPerp = [ϵₖ*v for v in dualBasisParallel]

#%%

fig = Figure(size=(1000,500), fontsize=24)
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"(%$l)" for l in individualLetters]

radii = norm.([𝐜ⱼ[j].-systemCOM2 for j=1:size(B2,2)])
dummyDists = collect(maximum(radii)/100:maximum(radii)/100:maximum(radii))

cellPolygons = findCellPolygons(R2, A2, B2)
edgeQuadrilaterals = findEdgeQuadrilaterals(R2, A2, B2)


# Primal network 
α = 1.0
β = 0.0
# edgeVectorsPrimal = eigenvectors_Lprimal[1].*(α.*primalBasisParallel .+ β.*primalBasisPerp)
edgeVectorsPrimalNorms = abs.(eigenvectors_Lprimal[1])./findEdgeLengths(R2, A2)
# @show edgeVectorsPrimalNorms .- norm.(edgeVectorsPrimal)
edgeVectorsPrimalNormsLims = (-2.0, log10(maximum(edgeVectorsPrimalNorms)))
push!(axes, Axis(fig[1,1], aspect=DataAspect(), alignmode=Inside()))
# scatter!(axes[end], Point{2,Float64}(centralCellCOM), color=:red)
for j=1:size(B2,2)
    poly!(axes[end], edgeQuadrilaterals[j], color=log10(edgeVectorsPrimalNorms[j]), strokecolor=(:white, 0.0), strokewidth=0, colormap=Reverse(:devon), colorrange=edgeVectorsPrimalNormsLims)
end
for i=1:size(B2,1)
    poly!(axes[end], cellPolygons[i], color=(:white, 0.0), strokecolor=(:black, 0.2), strokewidth=0.5)
end
scatter!(axes[end], [Point{2,Float64}(systemCOM2)], color=(:red,0.5))
hidedecorations!(axes[end])
hidespines!(axes[end])
Label(fig[2,1], popfirst!(subfigureLabels), fontsize=24) #"Single hole, 1st eigenmode, primal network, α=$(α), β=$(β)")
Colorbar(fig[1,2], colorrange=edgeVectorsPrimalNormsLims, colormap=Reverse(:devon), height=Relative(0.8), label=L"log_{10}\left(\chi_j\right)")
push!(axes, Axis(fig[1,3], xscale=log10, yscale=log10, aspect=AxisAspect(1), alignmode=Inside()))
scatter!(axes[end], radii, edgeVectorsPrimalNorms, color=(:blue,0.1))
lines!(axes[end], dummyDists, 0.3./dummyDists, color=(:black, 0.75), linestyle=:dash)
# lines!(axes[end], dummyDists, 1.0./(dummyDists).^2, color=(:black, 0.75), linestyle=:dash)
lines!(axes[end], dummyDists, 0.3./(dummyDists).^3, color=(:black, 0.75), linestyle=:dash)
ylims!(axes[end], (minimum(edgeVectorsPrimalNorms),1.0))
xlims!(axes[end], (minimum(radii),maximum(radii)))
axes[end].xlabel = L"r_j"
axes[end].ylabel = L"log_{10}\left(\chi_j\right)"
Label(fig[2,3], popfirst!(subfigureLabels), fontsize=24) #"Single hole, 1st eigenmode, primal network, α=$(α), β=$(β)")

rowsize!(fig.layout, 1, Relative(0.99))
rowsize!(fig.layout, 2, Relative(0.01))
colsize!(fig.layout, 1, Relative(0.5))
colsize!(fig.layout, 2, Relative(0.1))
colsize!(fig.layout, 3, Relative(0.4))

colgap!(fig.layout, 1, Relative(-0.05))
rowgap!(fig.layout, 1, Relative(-0.05))

resize_to_layout!(fig)

display(fig)

save(plotsdir("edgeLaplacianHarmonicFieldCellRemoved_$(inputSystem).png"), fig)
save(plotsdir("edgeLaplacianHarmonicFieldCellRemoved_$(inputSystem).pdf"), fig)


