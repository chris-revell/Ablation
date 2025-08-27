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

@from "$(srcdir("AblateCells.jl"))" using AblateCells
@from "$(srcdir("Stresses.jl"))" using Stresses

inputSystem = "Large2"

# inFile = datadir("referenceSystems", "quadraticPotentialNoPressure", "$(inputSystem)_testSystem.jld2")
# importedData = load(inFile)
# @unpack R, A, B, F = importedData

# systemCOM = sum(R)./length(R)
# cellCentres1 = findCellCentresOfMass(R, A, B)
# centralCell = findmin(norm.([cellCentres1[i].-systemCOM for i=1:size(B,1)]))[2]
# # centralCellCOM = cellCentres1[centralCell]

# neighbourMatrix = dropzeros(B*transpose(B))
# # neighbourCells = unique(findall(x->x!=0, neighbourMatrix[centralCell,:]))
# neighbourCellsTmp = neighbourMatrix[centralCell,:]
# neighbourCells = neighbourCellsTmp[neighbourCellsTmp.!=centralCell]
# ablatedCells = [centralCell]

# # Rtmp, Atmp, Btmp = ablateCells(R, A, B, [centralCell])
# Rtmp, Atmp, Btmp = ablateCells(R, A, B, ablatedCells)

# integ2 = vertexModel(abstol = 1e-9,
#                     reltol = 1e-9,
#                     initialSystem="argument",
#                     divisionToggle=0,
#                     R_in=Rtmp,
#                     A_in=Atmp,
#                     B_in=Btmp,
#                     pressureExternal=0.0,
#                     nCycles=0.5,
#                     outputToggle=0,
#                     frameDataToggle=0,
#                     frameImageToggle=0,
#                     videoToggle=0,
#                     printToggle=1,
#                     energyModel="quadratic",
#                     # termSteadyState=true,
#                 )
# #%%
# R2 = reinterpret(SVector{2,Float64}, integ2.u) 
# params2, matrices2 = integ2.p
# A2 = matrices2.A
# B2 = matrices2.B
# F2 = matrices2.F
# @show maximum(norm.(sum(F2, dims=2)))

# C = findC(A, B)
# holeVertices = findall(x->x!=0, C[centralCell,:])
# systemCOM2 = sum(R2[holeVertices])./length(holeVertices)

# jldsave(datadir("referenceSystems", "quadraticPotentialNoPressure", "$(inputSystem)Ablated_testSystem.jld2"); R2,
#     A2, 
#     B2, 
#     F2, 
#     systemCOM2,
# )


inFile = datadir("referenceSystems", "quadraticPotentialNoPressure", "$(inputSystem)Ablated_testSystem.jld2")
importedData = load(inFile)
@unpack R2, A2, B2, F2, systemCOM2 = importedData


#%%
    

cellCentres2 = findCellCentresOfMass(R2, A2, B2)
cellPolygons1 = findCellPolygons(R, A, B)
cellPolygons2 = findCellPolygons(R2, A2, B2)

zperp = 1.0

cellradii1 = norm.([cellCentres1[i].-systemCOM2 for i=1:size(B,1)])
cellradii2 = norm.([cellCentres2[i].-systemCOM2 for i=1:size(B2,1)])
dummyCellRadii1 = collect(maximum(cellradii1)/100:maximum(cellradii1)/100:maximum(cellradii1))
dummyCellRadii2 = collect(maximum(cellradii2)/100:maximum(cellradii2)/100:maximum(cellradii2))
edgeradii1 = norm.([findEdgeMidpoints(R, A)[j].-systemCOM2 for j=1:size(A,2)])
edgeradii2 = norm.([findEdgeMidpoints(R2, A2)[j].-systemCOM2 for j=1:size(A2,2)])
dummyEdgeRadii1 = collect(maximum(edgeradii1)/100:maximum(edgeradii1)/100:maximum(edgeradii1))
dummyEdgeRadii2 = collect(maximum(edgeradii2)/100:maximum(edgeradii2)/100:maximum(edgeradii2))

cellPolygons1 = findCellPolygons(R, A, B)
cellPolygons2 = findCellPolygons(R2, A2, B2)
boundaryCells1 = findBoundaryCells(B).==1
boundaryCells2 = findBoundaryCells(B2).==1

𝐡1 = hNetwork(R, A, B, F)
𝐡2 = hNetwork(R2, A2, B2, F2)

# Row 1: Isotropic stress 
pEff1 = pEff(R, A, B, 0.2, 0.75)
pEff2 = pEff(R2, A2, B2, 0.2, 0.75)

Lprimal1 = edgeLaplacianPrimal(R, A, B)
Lprimal2 = edgeLaplacianPrimal(R2, A2, B2)
𝐰ᵐ1 = [col for col in  eachcol((eigen(Matrix(Lprimal1))).vectors)]
𝐰ᵐ2 = [col for col in  eachcol((eigen(Matrix(Lprimal2))).vectors)]

χⱼ2 = abs.(𝐰ᵐ2[1])./findEdgeLengths(R2, A2)
χⱼ2Lims = (-2.0, log10(maximum(χⱼ2)))

# edgeQuadrilaterals1 = findEdgeQuadrilaterals(R, A, B)
edgeQuadrilaterals2 = findEdgeQuadrilaterals(R2, A2, B2)

ζᵢ2 = ζ(R2, A2, B2, 1, zperp)
ζᵢ2lims = (minimum(log10.(ζᵢ2)), maximum(log10.(ζᵢ2)))

σᵢ1 = σ(R, A, B, 𝐡1)
σᵢ2 = σ(R2, A2, B2, 𝐡2)
# cocurlᶜhMinusTrσ = cocurlᶜ(R, A, B, 𝐡1).-tr.(σᵢ1)
# @show maximum(abs.(cocurlᶜhMinusTrσ))
# aᵢ = findCellAreas(R,A,B)
# @show sum([aᵢ[i].*σᵢ1[i] for i=1:size(B,1)])
# aᵢ = findCellAreas(R2,A2,B2)
# @show sum([aᵢ[i].*σᵢ2[i] for i=1:size(B2,1)])

# Deviatoric stress 
σDᵢ1 = [σᵢ1[i] .- 0.5*tr(σᵢ1[i]) for i=1:size(B,1)]
# @show maximum(abs.(tr.(σDᵢ1))) # Validation 
σDᵢ2 = [σᵢ2[i] .- 0.5*tr(σᵢ2[i]) for i=1:size(B2,1)]
# @show maximum(abs.(tr.(σDᵢ2))) # Validation 
σDSᵢ1 = 0.5.*(σDᵢ1 .+ transpose.(σDᵢ1))
σDSᵢ2 = 0.5.*(σDᵢ2 .+ transpose.(σDᵢ2))

ζᵢ1 = [sqrt(-det(σDSᵢ1[i])) for i=1:size(B,1)]
ζᵢ2 = [sqrt(-det(σDSᵢ2[i])) for i=1:size(B2,1)]
ζᵢ1lims = (minimum(log10.(ζᵢ1)), maximum(log10.(ζᵢ1)))
ζᵢ2lims = (minimum(log10.(ζᵢ2)), maximum(log10.(ζᵢ2)))


# save(plotsdir("comparisons.png"), fig)
# save(plotsdir("comparisons.pdf"), fig)




#%%

difζ = ζᵢ2.-ζᵢ1[Not(ablatedCells)]
difζLims = (minimum(difζ), maximum(difζ))

fig = Figure(size=(1000,1000), fontsize=24)
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"(%$l)" for l in individualLetters]

push!(axes, Axis(fig[1,1], aspect=DataAspect()))
hidedecorations!(axes[end])
hidespines!(axes[end])
for i=1:size(B2,1)
    poly!(axes[end], cellPolygons2[i], color=difζ[i], colorrange = difζLims, colormap = :imola, strokecolor=(:black, 0.2), strokewidth=0.5)
end
scatter!(axes[end], [Point{2,Float64}(systemCOM2)], color=(:red, 0.75))
Colorbar(fig[1,2], colorrange=difζLims, colormap=:imola, height=Relative(0.8), label=L"\Delta\zeta_i")
Label(fig[2,1], popfirst!(subfigureLabels), fontsize=24) 
push!(axes, Axis(fig[1,3], aspect=AxisAspect(1), xscale=log10, yscale=log10))
scatter!(axes[end], cellradii2, abs.(difζ))
lines!(axes[end], dummyCellRadii2, 1.0./dummyCellRadii2)
lines!(axes[end], dummyCellRadii2, 1.0./(dummyCellRadii2).^2)
axes[end].ylabel = L"\Delta\zeta_i"
axes[end].xlabel = L"r_i"
xlims!(axes[end], (0.2, maximum(cellradii2)))
hidedecorations!(axes[end])
hidespines!(axes[end])
Label(fig[2,3], popfirst!(subfigureLabels), fontsize=24) 
# ylims!(axes[end], (minimum(abs.(difζ)),1.0))

difpEff = pEff2.-pEff1[Not(ablatedCells)]
difpEffLims = (minimum(difpEff), maximum(difpEff))
push!(axes, Axis(fig[3,1], aspect=DataAspect()))
hidedecorations!(axes[end])
hidespines!(axes[end])
for i=1:size(B2,1)
    poly!(axes[end], cellPolygons2[i], color=difpEff[i], colorrange = difpEffLims, colormap = :imola, strokecolor=(:black, 0.2), strokewidth=0.5)
end
Colorbar(fig[3,2], colorrange=difpEffLims, colormap=:imola, height=Relative(0.8), label=L"\Delta P_{eff}")
Label(fig[4,1], popfirst!(subfigureLabels), fontsize=24) 
push!(axes, Axis(fig[3,3], aspect=AxisAspect(1), xscale=log10, yscale=log10))
scatter!(axes[end], cellradii2, abs.(difpEff))
lines!(axes[end], dummyCellRadii2, 1.0./dummyCellRadii2)
lines!(axes[end], dummyCellRadii2, 1.0./(dummyCellRadii2).^2)
axes[end].ylabel = L"\Delta P_{eff}"
axes[end].xlabel = L"r_i"
xlims!(axes[end], (0.2, maximum(cellradii2)))
hidedecorations!(axes[end])
hidespines!(axes[end])
Label(fig[4,3], popfirst!(subfigureLabels), fontsize=24) 
# ylims!(axes[end], (minimum(abs.(difpEff)),1.0))

resize_to_layout!(fig)

save(plotsdir("$(inputSystem)_comparison.png"), fig)

display(fig)
