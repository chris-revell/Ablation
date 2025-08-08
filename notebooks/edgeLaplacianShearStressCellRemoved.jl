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

@from "$(srcdir("AblateCells.jl"))" using AblateCells

ϵᵢ = SMatrix{2, 2, Float64}([
                0.0 1.0
                -1.0 0.0
            ])
ϵₖ = SMatrix{2, 2, Float64}([
                0.0 -1.0
                1.0 0.0
            ])

inFile = datadir("referenceSystems", "Large_testSystem.jld2")
importedData = load(inFile)
@unpack R, A, B, F = importedData

systemCOM = sum(R)./length(R)
cellCentres = findCellCentresOfMass(R, A, B)
centralCell = findmin(norm.([cellCentres[i].-systemCOM for i=1:size(B,1)]))[2]
centralCellCOM = cellCentres[centralCell]

R2, A2, B2 = ablateCells(R, A, B, [centralCell])
cellCentres2 = findCellCentresOfMass(R2, A2, B2)

#%%

function ζ(R, A, B, m, β)
    Lprimal = edgeLaplacianPrimal(R, A, B)
    w = transpose((eigen(Matrix(Lprimal))).vectors)
    𝐭̂ = normalize.(findEdgeTangents(R, A))
    ζᵐᵢ = zeros(size(B,1))
    for i=1:size(B,1)
       tmp = sum([B[i,j].*w[m, j].*(𝐭̂[j]*𝐭̂[j]') for j=1:size(B,2)])
       tmpDet = det(tmp)
       ζᵐᵢ[i] = β*sqrt(-tmpDet)
    end
    return ζᵐᵢ
end 
    
β = 1.0

ζᵢ = ζ(R2, A2, B2, 1, β)

fig = Figure(size=(1000,500), fontsize=24)
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"(%$l)" for l in individualLetters]

radii = norm.([cellCentres2[i].-centralCellCOM for i=1:size(B2,1)])
dummyDists = collect(maximum(radii)/100:maximum(radii)/100:maximum(radii))

cellPolygons = findCellPolygons(R2, A2, B2)
boundaryCells = findBoundaryCells(B2).==1

# 1st eigenmode
clims = (minimum(log10.(ζᵢ)), maximum(log10.(ζᵢ)))
push!(axes, Axis(fig[1,1], aspect=DataAspect(), alignmode=Inside()))
for i=1:size(B2,1)
    poly!(axes[end], cellPolygons[i], color=log10(ζᵢ[i]), colorrange = clims, colormap = Reverse(:devon), strokecolor=(:black, 0.2), strokewidth=0.5)
end
hidedecorations!(axes[end])
hidespines!(axes[end])
Label(fig[2,1], popfirst!(subfigureLabels), fontsize=24) #"Single hole, 1st eigenmode, primal network, α=$(α), β=$(β)")
Colorbar(fig[1,2], colorrange=clims, colormap=Reverse(:devon), height=Relative(0.8), alignmode=Inside())

push!(axes, Axis(fig[1,3], yscale=log10, xscale=log10, aspect=AxisAspect(1), alignmode=Inside()))
scatter!(axes[end], radii[Not(boundaryCells)], ζᵢ[Not(boundaryCells)], color=(:blue,0.3))
scatter!(axes[end], radii[boundaryCells], ζᵢ[boundaryCells], color=(:red,0.3))
lines!(axes[end], dummyDists, 0.1./dummyDists, color=(:black, 0.75), linestyle=:dash)
# lines!(axes[end], dummyDists, 1.0./(dummyDists).^2, color=(:black, 0.75), linestyle=:dash)
lines!(axes[end], dummyDists, 0.1./(dummyDists).^3, color=(:black, 0.75), linestyle=:dash)
ylims!(axes[end], (minimum(ζᵢ),1.0))
xlims!(axes[end], (minimum(radii),maximum(radii)))
axes[end].xlabel = L"r_i"
axes[end].ylabel = L"\zeta"
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

save(plotsdir("edgeLaplacianShearStressCellRemoved.png"), fig)
save(plotsdir("edgeLaplacianShearStressCellRemoved.pdf"), fig)


