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

ϵᵢ = SMatrix{2, 2, Float64}([
                0.0 1.0
                -1.0 0.0
            ])
ϵₖ = SMatrix{2, 2, Float64}([
                0.0 -1.0
                1.0 0.0
            ])

# inFile = datadir("referenceSystems", "Large_testSystem.jld2")
inFile1 = datadir("referenceSystems", "quadraticPotentialNoPressure", "NoHole_testSystem.jld2")
importedData1 = load(inFile1)
@unpack R, A, B, F, cellTensions, cellAreas, cellPerimeters, cellPressures = importedData1
R1 = R
A1 = A 
B1 = B
F1 = F
cellTensions1 = cellTensions
cellAreas1 = cellAreas
cellPerimeters1 = cellPerimeters
cellPressures1 = cellPressures

inFile2 = datadir("referenceSystems", "quadraticPotentialNoPressure", "SingleHole_testSystem.jld2")
importedData2 = load(inFile2)
@unpack R, A, B, F, cellTensions, cellAreas, cellPerimeters, cellPressures = importedData2
R2 = R
A2 = A 
B2 = B
F2 = F
cellTensions2 = cellTensions
cellAreas2 = cellAreas
cellPerimeters2 = cellPerimeters
cellPressures2 = cellPressures

ζᵢ1 = cellPressures1 .- cellTensions1.*cellPerimeters1./(2.0.*cellAreas1)
ζᵢ2 = cellPressures2 .- cellTensions2.*cellPerimeters2./(2.0.*cellAreas2)

fig = Figure(size=(1000,1000), fontsize=24)
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"(%$l)" for l in individualLetters]

radii = norm.([cellCentres[i].-centralCellCOM for i=1:size(B,1)])
radii2 = norm.([cellCentres2[i].-centralCellCOM for i=1:size(B2,1)])
dummyDists = collect(maximum(radii)/100:maximum(radii)/100:maximum(radii))
dummyDists2 = collect(maximum(radii2)/100:maximum(radii2)/100:maximum(radii2))

cellPolygons = findCellPolygons(R1, A1, B1)
cellPolygons2 = findCellPolygons(R2, A2, B2)
boundaryCells1 = findBoundaryCells(B1).==1
boundaryCells2 = findBoundaryCells(B2).==1

# Before
# clims = (minimum(log10.(ζᵢ1)), maximum(log10.(ζᵢ1)))
clims = (0.0, max(maximum(ζᵢ2), maximum(ζᵢ1)))
push!(axes, Axis(fig[1,1], aspect=DataAspect(), alignmode=Inside()))
for i=1:size(B,1)
    # poly!(axes[end], cellPolygons[i], color=log10(ζᵢ1[i]), colorrange = clims, colormap = :batlow, strokecolor=(:black, 0.2), strokewidth=0.5)
    poly!(axes[end], cellPolygons[i], color=ζᵢ1[i], colorrange = clims, colormap = :batlow, strokecolor=(:black, 0.2), strokewidth=0.5)
end
hidedecorations!(axes[end])
hidespines!(axes[end])
Label(fig[2,1], popfirst!(subfigureLabels), fontsize=24)
Colorbar(fig[1,2], colorrange=clims, colormap=:batlow, height=Relative(0.8), alignmode=Inside())

# push!(axes, Axis(fig[1,3], yscale=log10, xscale=log10, aspect=AxisAspect(1), alignmode=Inside()))
push!(axes, Axis(fig[1,3], aspect=AxisAspect(1), alignmode=Inside()))
scatter!(axes[end], radii[Not(boundaryCells)], ζᵢ1[Not(boundaryCells)], color=(:blue,0.3))
scatter!(axes[end], radii[boundaryCells], ζᵢ1[boundaryCells], color=(:red,0.3))
# lines!(axes[end], dummyDists, 0.1./dummyDists, color=(:black, 0.75), linestyle=:dash)
# lines!(axes[end], dummyDists, 1.0./(dummyDists).^2, color=(:black, 0.75), linestyle=:dash)
# lines!(axes[end], dummyDists, 0.1./(dummyDists).^3, color=(:black, 0.75), linestyle=:dash)
ylims!(axes[end], clims)
xlims!(axes[end], (minimum(radii2),maximum(radii2)))
axes[end].xlabel = L"r_i"
axes[end].ylabel = L"\zeta"
Label(fig[2,3], popfirst!(subfigureLabels), fontsize=24) #"Single hole, 1st eigenmode, primal network, α=$(α), β=$(β)")

# After
# clims = (minimum(log10.(ζᵢ2)), maximum(log10.(ζᵢ2)))
clims = (0.0, max(maximum(ζᵢ2), maximum(ζᵢ1)))
push!(axes, Axis(fig[3,1], aspect=DataAspect(), alignmode=Inside()))
for i=1:size(B2,1)
    # poly!(axes[end], cellPolygons2[i], color=log10(ζᵢ2[i]), colorrange = clims, colormap = :batlow, strokecolor=(:black, 0.2), strokewidth=0.5)
    poly!(axes[end], cellPolygons2[i], color=ζᵢ2[i], colorrange = clims, colormap = :batlow, strokecolor=(:black, 0.2), strokewidth=0.5)
end
hidedecorations!(axes[end])
hidespines!(axes[end])
Label(fig[4,1], popfirst!(subfigureLabels), fontsize=24)
Colorbar(fig[3,2], colorrange=clims, colormap=:batlow, height=Relative(0.8), alignmode=Inside())

# push!(axes, Axis(fig[3,3], yscale=log10, xscale=log10, aspect=AxisAspect(1), alignmode=Inside()))
push!(axes, Axis(fig[3,3], aspect=AxisAspect(1), alignmode=Inside()))
scatter!(axes[end], radii2[Not(boundaryCells2)], ζᵢ2[Not(boundaryCells2)], color=(:blue,0.3))
scatter!(axes[end], radii2[boundaryCells2], ζᵢ2[boundaryCells2], color=(:red,0.3))
# lines!(axes[end], dummyDists2, 0.1./dummyDists2, color=(:black, 0.75), linestyle=:dash)
# lines!(axes[end], dummyDists, 1.0./(dummyDists).^2, color=(:black, 0.75), linestyle=:dash)
# lines!(axes[end], dummyDists2, 0.1./(dummyDists2).^3, color=(:black, 0.75), linestyle=:dash)
ylims!(axes[end], clims)
xlims!(axes[end], (minimum(radii2),maximum(radii2)))
axes[end].xlabel = L"r_i"
axes[end].ylabel = L"\zeta"
Label(fig[4,3], popfirst!(subfigureLabels), fontsize=24) #"Single hole, 1st eigenmode, primal network, α=$(α), β=$(β)")


# Difference
difζ = ζᵢ2.-ζᵢ1[Not(ablatedCells)]
difζ2 = ζᵢ2.-ζᵢ1[Not(ablatedCells)]
clims = (-maximum(abs.(difζ)), maximum(abs.(difζ)))
push!(axes, Axis(fig[5,1], aspect=DataAspect(), alignmode=Inside()))
for i=1:size(B2,1)
    # poly!(axes[end], cellPolygons2[i], color=log10(ζᵢ2[i]), colorrange = clims, colormap = :batlow, strokecolor=(:black, 0.2), strokewidth=0.5)
    poly!(axes[end], cellPolygons2[i], color=difζ[i], colorrange = clims, colormap = :vik, strokecolor=(:black, 0.2), strokewidth=0.5)
end
hidedecorations!(axes[end])
hidespines!(axes[end])
Label(fig[6,1], popfirst!(subfigureLabels), fontsize=24)
Colorbar(fig[5,2], colorrange=clims, colormap=:vik, height=Relative(0.8), alignmode=Inside())

push!(axes, Axis(fig[5,3], aspect=AxisAspect(1), alignmode=Inside()))
scatter!(axes[end], radii2[Not(boundaryCells2)], difζ[Not(boundaryCells2)], color=(:blue,0.3))
scatter!(axes[end], radii2[boundaryCells2], difζ[boundaryCells2], color=(:red,0.3))
# lines!(axes[end], dummyDists2, 0.1./dummyDists2, color=(:black, 0.75), linestyle=:dash)
# lines!(axes[end], dummyDists, 1.0./(dummyDists).^2, color=(:black, 0.75), linestyle=:dash)
# lines!(axes[end], dummyDists2, 0.1./(dummyDists2).^3, color=(:black, 0.75), linestyle=:dash)
ylims!(axes[end], (-maximum(abs.(difζ)),maximum(abs.(difζ))))
xlims!(axes[end], (minimum(radii2),maximum(radii2)))
axes[end].xlabel = L"r_i"
axes[end].ylabel = L"\zeta"
Label(fig[6,3], popfirst!(subfigureLabels), fontsize=24)


rowsize!(fig.layout, 1, Relative(0.32))
rowsize!(fig.layout, 2, Relative(0.01))
rowsize!(fig.layout, 3, Relative(0.32))
rowsize!(fig.layout, 4, Relative(0.01))
rowsize!(fig.layout, 5, Relative(0.32))
rowsize!(fig.layout, 6, Relative(0.01))
colsize!(fig.layout, 1, Relative(0.5))
colsize!(fig.layout, 2, Relative(0.1))
colsize!(fig.layout, 3, Relative(0.4))

# colgap!(fig.layout, 1, Relative(-0.05))
# rowgap!(fig.layout, 1, Relative(-0.05))

resize_to_layout!(fig)

display(fig)

save(plotsdir("edgeLaplacianShearStressBeforeAfterCellRemoved.png"), fig)
save(plotsdir("edgeLaplacianShearStressBeforeAfterCellRemoved.pdf"), fig)
