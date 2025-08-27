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
@from "$(srcdir("Stresses.jl"))" using Stresses

ϵᵢ = SMatrix{2, 2, Float64}([
                0.0 1.0
                -1.0 0.0
            ])
ϵₖ = SMatrix{2, 2, Float64}([
                0.0 -1.0
                1.0 0.0
            ])

# inFile = datadir("referenceSystems", "Large_testSystem.jld2")
inFile = datadir("referenceSystems", "quadraticPotentialNoPressure", "NoHole_testSystem.jld2")
importedData = load(inFile)
@unpack R, A, B, F = importedData

systemCOM = sum(R)./length(R)
cellCentres1 = findCellCentresOfMass(R, A, B)
centralCell = findmin(norm.([cellCentres1[i].-systemCOM for i=1:size(B,1)]))[2]
# centralCellCOM = cellCentres1[centralCell]

neighbourMatrix = dropzeros(B*transpose(B))
ablatedCells = unique(findall(x->x!=0, neighbourMatrix[centralCell,:]))

# Rtmp, Atmp, Btmp = ablateCells(R, A, B, [centralCell])
Rtmp, Atmp, Btmp = ablateCells(R, A, B, ablatedCells)

integ2 = vertexModel(abstol = 1e-9,
                    reltol = 1e-9,
                    initialSystem="argument",
                    divisionToggle=0,
                    R_in=Rtmp,
                    A_in=Atmp,
                    B_in=Btmp,
                    pressureExternal=0.0,
                    nCycles=0.5,
                    outputToggle=0,
                    frameDataToggle=0,
                    frameImageToggle=0,
                    videoToggle=0,
                    printToggle=1,
                    energyModel="quadratic",
                    # termSteadyState=true,
                )
#%%
R2 = reinterpret(SVector{2,Float64}, integ2.u) 
params2, matrices2 = integ2.p
A2 = matrices2.A
B2 = matrices2.B
F2 = matrices2.F
@show maximum(norm.(sum(F2, dims=2)))


#%%
    

cellCentres2 = findCellCentresOfMass(R2, A2, B2)
cellPolygons1 = findCellPolygons(R, A, B)
cellPolygons2 = findCellPolygons(R2, A2, B2)

zperp = 1.0


fig = Figure(size=(1000,2000), fontsize=24)
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"(%$l)" for l in individualLetters]

cellradii1 = norm.([cellCentres1[i].-systemCOM for i=1:size(B,1)])
cellradii2 = norm.([cellCentres2[i].-systemCOM for i=1:size(B2,1)])
dummyCellRadii1 = collect(maximum(cellradii1)/100:maximum(cellradii1)/100:maximum(cellradii1))
dummyCellRadii2 = collect(maximum(cellradii2)/100:maximum(cellradii2)/100:maximum(cellradii2))
edgeradii1 = norm.([findEdgeMidpoints(R, A)[j].-systemCOM for j=1:size(A,2)])
edgeradii2 = norm.([findEdgeMidpoints(R2, A2)[j].-systemCOM for j=1:size(A2,2)])
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

Label(fig[1,0], "pEff")

clims = (minimum(pEff1), maximum(pEff2))
push!(axes, Axis(fig[1,1], aspect=DataAspect(), alignmode=Inside()))
for i=1:size(B,1)
    poly!(axes[end], cellPolygons1[i], color=pEff1[i], colorrange = clims, colormap = :imola, strokecolor=(:black, 0.2), strokewidth=0.5)
end
hidedecorations!(axes[end])
hidespines!(axes[end])
Label(fig[2,1], popfirst!(subfigureLabels), fontsize=24)
Colorbar(fig[1,2], colorrange=clims, colormap=:imola, height=Relative(0.8), alignmode=Inside())

clims = (minimum(pEff2), maximum(pEff2))
push!(axes, Axis(fig[1,3], aspect=DataAspect(), alignmode=Inside()))
for i=1:size(B2,1)
    poly!(axes[end], cellPolygons2[i], color=pEff2[i], colorrange = clims, colormap = :imola, strokecolor=(:black, 0.2), strokewidth=0.5)
end
hidedecorations!(axes[end])
hidespines!(axes[end])
Label(fig[2,3], popfirst!(subfigureLabels), fontsize=24)
Colorbar(fig[1,4], colorrange=clims, colormap=:imola, height=Relative(0.8), alignmode=Inside())

push!(axes, Axis(fig[1,5], aspect=AxisAspect(1)))
scatter!(axes[end], cellradii1, pEff1, color=(:blue,0.2))
scatter!(axes[end], cellradii2, pEff2, color=(:red,0.2))
axes[end].xlabel = L"r_i"
axes[end].ylabel = L"pEff"
Label(fig[2,5], popfirst!(subfigureLabels), fontsize=24) 

# Row 2: Harmonic field 

Lprimal1 = edgeLaplacianPrimal(R, A, B)
Lprimal2 = edgeLaplacianPrimal(R2, A2, B2)
𝐰ᵐ1 = [col for col in  eachcol((eigen(Matrix(Lprimal1))).vectors)]
𝐰ᵐ2 = [col for col in  eachcol((eigen(Matrix(Lprimal2))).vectors)]


# Primal network 
# χⱼ1 = abs.(𝐰ᵐ1[1])./findEdgeLengths(R, A)
χⱼ2 = abs.(𝐰ᵐ2[1])./findEdgeLengths(R2, A2)
# χⱼ1Lims = (-2.0, log10(maximum(χⱼ1)))
χⱼ2Lims = (-2.0, log10(maximum(χⱼ2)))

# edgeQuadrilaterals1 = findEdgeQuadrilaterals(R, A, B)
edgeQuadrilaterals2 = findEdgeQuadrilaterals(R2, A2, B2)

push!(axes, Axis(fig[3,3], aspect=DataAspect()))
for j=1:size(B2,2)
    poly!(axes[end], edgeQuadrilaterals2[j], color=log10(χⱼ2[j]), strokecolor=(:white, 0.0), strokewidth=0, colormap=:imola, colorrange=χⱼ2Lims)
end
for i=1:size(B2,1)
    poly!(axes[end], cellPolygons2[i], color=(:white, 0.0), strokecolor=(:black, 0.2), strokewidth=0.5)
end
hidedecorations!(axes[end])
hidespines!(axes[end])
Label(fig[4,3], popfirst!(subfigureLabels), fontsize=24) #"Single hole, 1st eigenmode, primal network, α=$(α), zperp=$(zperp)")
Colorbar(fig[3,4], colorrange=χⱼ2Lims, colormap=:imola, height=Relative(0.8))

# push!(axes, Axis(fig[3,5], xscale=log10, yscale=log10, aspect=AxisAspect(1)))
# scatter!(axes[end], edgeradii2, χⱼ, color=(:blue,0.1))
# lines!(axes[end], dummyEdgeRadii2, 0.3./dummyEdgeRadii2, color=(:black, 0.75), linestyle=:dash)
# lines!(axes[end], dummyEdgeRadii2, 1.0./(dummyEdgeRadii2).^2, color=(:black, 0.75), linestyle=:dash)
# lines!(axes[end], dummyEdgeRadii2, 0.3./(dummyEdgeRadii2).^3, color=(:black, 0.75), linestyle=:dash)
# # ylims!(axes[end], (minimum(χⱼ),1.0))
# # xlims!(axes[end], (minimum(radii),maximum(radii)))
# axes[end].xlabel = L"r_j"
# axes[end].ylabel = L"\chi"
# Label(fig[2,3], popfirst!(subfigureLabels), fontsize=24) 


# Row 4: Harmonic component of shear stress 

Label(fig[5,0], L"\zeta_i^{(m)}, 3.4")

# ζᵢ1 = ζ(R, A, B, 1, zperp)
ζᵢ2 = ζ(R2, A2, B2, 1, zperp)
# ζᵢ1lims = (minimum(log10.(ζᵢ1)), maximum(log10.(ζᵢ1)))
ζᵢ2lims = (minimum(log10.(ζᵢ2)), maximum(log10.(ζᵢ2)))

# push!(axes, Axis(fig[5,1], aspect=DataAspect(), alignmode=Inside()))
# for i=1:size(B,1)
#     poly!(axes[end], cellPolygons1[i], color=log10(ζᵢ1[i]), colorrange = ζᵢ1lims, colormap = :imola, strokecolor=(:black, 0.2), strokewidth=0.5)
# end
# hidedecorations!(axes[end])
# hidespines!(axes[end])
# Label(fig[6,1], popfirst!(subfigureLabels), fontsize=24)
# Colorbar(fig[5,2], colorrange=ζᵢ1lims, colormap=:imola, height=Relative(0.8))
# push!(axes, Axis(fig[1,3], yscale=log10, xscale=log10, aspect=AxisAspect(1)))
# scatter!(axes[end], radii[Not(boundaryCells)], ζᵢ[Not(boundaryCells)], color=(:blue,0.3))
# scatter!(axes[end], radii[boundaryCells], ζᵢ[boundaryCells], color=(:red,0.3))
# lines!(axes[end], dummyDists, 0.1./dummyDists, color=(:black, 0.75), linestyle=:dash)
# # lines!(axes[end], dummyDists, 1.0./(dummyDists).^2, color=(:black, 0.75), linestyle=:dash)
# lines!(axes[end], dummyDists, 0.1./(dummyDists).^3, color=(:black, 0.75), linestyle=:dash)
# ylims!(axes[end], (minimum(ζᵢ),1.0))
# xlims!(axes[end], (minimum(radii),maximum(radii)))
# axes[end].xlabel = L"r_i"
# axes[end].ylabel = L"\zeta"
# Label(fig[2,3], popfirst!(subfigureLabels), fontsize=24) #"Single hole, 1st eigenmode, primal network, α=$(α), β=$(β)")

push!(axes, Axis(fig[5,3], aspect=DataAspect(), alignmode=Inside()))
for i=1:size(B2,1)
    poly!(axes[end], cellPolygons2[i], color=log10(ζᵢ2[i]), colorrange = ζᵢ2lims, colormap = :imola, strokecolor=(:black, 0.2), strokewidth=0.5)
end
hidedecorations!(axes[end])
hidespines!(axes[end])
Label(fig[6,3], popfirst!(subfigureLabels), fontsize=24)
Colorbar(fig[5,4], colorrange=ζᵢ2lims, colormap=:imola, height=Relative(0.8))
# push!(axes, Axis(fig[1,3], yscale=log10, xscale=log10, aspect=AxisAspect(1)))
# scatter!(axes[end], radii[Not(boundaryCells)], ζᵢ[Not(boundaryCells)], color=(:blue,0.3))
# scatter!(axes[end], radii[boundaryCells], ζᵢ[boundaryCells], color=(:red,0.3))
# lines!(axes[end], dummyDists, 0.1./dummyDists, color=(:black, 0.75), linestyle=:dash)
# # lines!(axes[end], dummyDists, 1.0./(dummyDists).^2, color=(:black, 0.75), linestyle=:dash)
# lines!(axes[end], dummyDists, 0.1./(dummyDists).^3, color=(:black, 0.75), linestyle=:dash)
# ylims!(axes[end], (minimum(ζᵢ),1.0))
# xlims!(axes[end], (minimum(radii),maximum(radii)))
# axes[end].xlabel = L"r_i"
# axes[end].ylabel = L"\zeta"
# Label(fig[2,3], popfirst!(subfigureLabels), fontsize=24) #"Single hole, 1st eigenmode, primal network, α=$(α), β=$(β)")



# Row 4: ??

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

Label(fig[7,0], L"\zeta_i")

push!(axes, Axis(fig[7,1], aspect=DataAspect(), alignmode=Inside()))
for i=1:size(B,1)
    poly!(axes[end], cellPolygons1[i], color=log10(ζᵢ1[i]), colorrange = ζᵢ1lims, colormap = :imola, strokecolor=(:black, 0.2), strokewidth=0.5)
end
hidedecorations!(axes[end])
hidespines!(axes[end])
Label(fig[8,1], popfirst!(subfigureLabels), fontsize=24)
Colorbar(fig[7,2], colorrange=ζᵢ1lims, colormap=:imola, height=Relative(0.8))
# push!(axes, Axis(fig[1,3], yscale=log10, xscale=log10, aspect=AxisAspect(1)))
# scatter!(axes[end], radii[Not(boundaryCells)], ζᵢ[Not(boundaryCells)], color=(:blue,0.3))
# scatter!(axes[end], radii[boundaryCells], ζᵢ[boundaryCells], color=(:red,0.3))
# lines!(axes[end], dummyDists, 0.1./dummyDists, color=(:black, 0.75), linestyle=:dash)
# # lines!(axes[end], dummyDists, 1.0./(dummyDists).^2, color=(:black, 0.75), linestyle=:dash)
# lines!(axes[end], dummyDists, 0.1./(dummyDists).^3, color=(:black, 0.75), linestyle=:dash)
# ylims!(axes[end], (minimum(ζᵢ),1.0))
# xlims!(axes[end], (minimum(radii),maximum(radii)))
# axes[end].xlabel = L"r_i"
# axes[end].ylabel = L"\zeta"
# Label(fig[2,3], popfirst!(subfigureLabels), fontsize=24) #"Single hole, 1st eigenmode, primal network, α=$(α), β=$(β)")

push!(axes, Axis(fig[7,3], aspect=DataAspect(), alignmode=Inside()))
for i=1:size(B2,1)
    poly!(axes[end], cellPolygons2[i], color=log10(ζᵢ2[i]), colorrange = ζᵢ2lims, colormap = :imola, strokecolor=(:black, 0.2), strokewidth=0.5)
end
hidedecorations!(axes[end])
hidespines!(axes[end])
Label(fig[8,3], popfirst!(subfigureLabels), fontsize=24)
Colorbar(fig[7,4], colorrange=ζᵢ2lims, colormap=:imola, height=Relative(0.8))
# push!(axes, Axis(fig[1,3], yscale=log10, xscale=log10, aspect=AxisAspect(1)))
# scatter!(axes[end], radii[Not(boundaryCells)], ζᵢ[Not(boundaryCells)], color=(:blue,0.3))
# scatter!(axes[end], radii[boundaryCells], ζᵢ[boundaryCells], color=(:red,0.3))
# lines!(axes[end], dummyDists, 0.1./dummyDists, color=(:black, 0.75), linestyle=:dash)
# # lines!(axes[end], dummyDists, 1.0./(dummyDists).^2, color=(:black, 0.75), linestyle=:dash)
# lines!(axes[end], dummyDists, 0.1./(dummyDists).^3, color=(:black, 0.75), linestyle=:dash)
# ylims!(axes[end], (minimum(ζᵢ),1.0))
# xlims!(axes[end], (minimum(radii),maximum(radii)))
# axes[end].xlabel = L"r_i"
# axes[end].ylabel = L"\zeta"
# Label(fig[2,3], popfirst!(subfigureLabels), fontsize=24) #"Single hole, 1st eigenmode, primal network, α=$(α), β=$(β)")


rowsize!(fig.layout, 1, Relative(0.23))
rowsize!(fig.layout, 2, Relative(0.02))
rowsize!(fig.layout, 3, Relative(0.23))
rowsize!(fig.layout, 4, Relative(0.02))
rowsize!(fig.layout, 5, Relative(0.23))
rowsize!(fig.layout, 6, Relative(0.02))
rowsize!(fig.layout, 7, Relative(0.23))
rowsize!(fig.layout, 8, Relative(0.02))

colsize!(fig.layout, 0, Relative(0.03))
colsize!(fig.layout, 1, Relative(0.4))
colsize!(fig.layout, 2, Relative(0.03))
colsize!(fig.layout, 3, Relative(0.4))
colsize!(fig.layout, 4, Relative(0.03))

# colgap!(fig.layout, 1, Relative(-0.05))
# rowgap!(fig.layout, 1, Relative(-0.05))

resize_to_layout!(fig)

display(fig)

save(plotsdir("comparisons.png"), fig)
save(plotsdir("comparisons.pdf"), fig)




#%%

difζ = ζᵢ2.-ζᵢ1[Not(ablatedCells)]
difζLims = (minimum(difζ), maximum(difζ))

fig = Figure(size=(1000,1000))
ax1 = Axis(fig[1,1])
for i=1:size(B2,1)
    poly!(ax1, cellPolygons2[i], color=difζ[i], colorrange = difζLims, colormap = :imola, strokecolor=(:black, 0.2), strokewidth=0.5)
end

ax2 = Axis(fig[1,2], xscale=log10, yscale=log10)
scatter!(ax2, cellradii2, abs.(difζ))
lines!(ax2, dummyCellRadii2, 1.0./dummyCellRadii2)
lines!(ax2, dummyCellRadii2, 1.0./(dummyCellRadii2).^2)
xlims!(ax2, (0.75, maximum(cellradii2)))
ylims!(ax2, (minimum(abs.(difζ)),1.0))

difpEff = pEff2.-pEff1[Not(ablatedCells)]
difpEffLims = (minimum(difpEff), maximum(difpEff))
ax3 = Axis(fig[2,1])
for i=1:size(B2,1)
    poly!(ax3, cellPolygons2[i], color=difpEff[i], colorrange = difpEffLims, colormap = :imola, strokecolor=(:black, 0.2), strokewidth=0.5)
end

ax4 = Axis(fig[2,2], xscale=log10, yscale=log10)
scatter!(ax4, cellradii2, abs.(difpEff))
lines!(ax4, dummyCellRadii2, 1.0./dummyCellRadii2)
lines!(ax4, dummyCellRadii2, 1.0./(dummyCellRadii2).^2)
xlims!(ax4, (0.75, maximum(cellradii2)))
ylims!(ax4, (minimum(abs.(difpEff)),1.0))
display(fig)
