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

inputSystem = "Large2"
inputDir = "quadraticPotentialNoPressure"; isdir(plotsdir(inputDir)) ? nothing : mkdir(plotsdir(inputDir))
# inputDir = "quadraticPotentialWithPressure"; isdir(plotsdir(inputDir)) ? nothing : mkdir(plotsdir(inputDir))

inFile = datadir("referenceSystems", inputDir, "$(inputSystem)_testSystem.jld2")
importedData = load(inFile)
@unpack R, A, B, F = importedData

systemCOM = sum(R)./length(R)
cellCentres1 = findCellCentresOfMass(R, A, B)
centralCell = findmin(norm.([cellCentres1[i].-systemCOM for i=1:size(B,1)]))[2]
# centralCellCOM = cellCentres1[centralCell]

neighbourMatrix = dropzeros(B*transpose(B))
# ablatedCells = unique(findall(x->x!=0, neighbourMatrix[centralCell,:]))
ablatedCells = [centralCell]

if isfile(datadir("referenceSystems", inputDir, "$(inputSystem)Ablated_testSystem.jld2"))
    inFile = datadir("referenceSystems", inputDir, "$(inputSystem)Ablated_testSystem.jld2")
    importedData = load(inFile)
    @unpack R2, A2, B2, F2, systemCOM2 = importedData
else
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

    C = findC(A, B)
    centralCellVertices = R[findall(x->x!=0, C[centralCell, :])]
    systemCOM2 = sum(centralCellVertices)./length(centralCellVertices)
end

cellCentres2 = findCellCentresOfMass(R2, A2, B2)

#%%
    
zperp = 1.0

ζᵢ = ζ(R2, A2, B2, 1, zperp)

fig = Figure(size=(1000,500), fontsize=24)
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"(%$l)" for l in individualLetters]

radii = norm.([cellCentres2[i].-systemCOM for i=1:size(B2,1)])
dummyDists = collect(maximum(radii)/100:maximum(radii)/100:maximum(radii))

cellPolygons = findCellPolygons(R2, A2, B2)
peripheralCells = findPeripheralCells(B2).==1

# 1st eigenmode
clims = (minimum(log10.(ζᵢ)), maximum(log10.(ζᵢ)))
push!(axes, Axis(fig[1,1], aspect=DataAspect(), alignmode=Inside()))
for i=1:size(B2,1)
    poly!(axes[end], cellPolygons[i], color=log10(ζᵢ[i]), colorrange = clims, colormap = Reverse(:devon), strokecolor=(:black, 0.2), strokewidth=0.5)
end
hidedecorations!(axes[end])
hidespines!(axes[end])
Label(fig[2,1], popfirst!(subfigureLabels), fontsize=24) #"Single hole, 1st eigenmode, primal network, α=$(α), zperp=$(zperp)")
Colorbar(fig[1,2], colorrange=clims, colormap=Reverse(:devon), height=Relative(0.8), alignmode=Inside())

push!(axes, Axis(fig[1,3], yscale=log10, xscale=log10, aspect=AxisAspect(1), alignmode=Inside()))
scatter!(axes[end], radii[Not(peripheralCells)], ζᵢ[Not(peripheralCells)], color=(:blue,0.3))
scatter!(axes[end], radii[peripheralCells], ζᵢ[peripheralCells], color=(:red,0.3))
lines!(axes[end], dummyDists, 0.1./dummyDists, color=(:black, 0.75), linestyle=:dash)
# lines!(axes[end], dummyDists, 1.0./(dummyDists).^2, color=(:black, 0.75), linestyle=:dash)
lines!(axes[end], dummyDists, 0.1./(dummyDists).^3, color=(:black, 0.75), linestyle=:dash)
ylims!(axes[end], (minimum(ζᵢ),1.0))
xlims!(axes[end], (minimum(radii),maximum(radii)))
axes[end].xlabel = L"r_i"
axes[end].ylabel = L"\zeta"
Label(fig[2,3], popfirst!(subfigureLabels), fontsize=24) #"Single hole, 1st eigenmode, primal network, α=$(α), zperp=$(zperp)")

rowsize!(fig.layout, 1, Relative(0.99))
rowsize!(fig.layout, 2, Relative(0.01))
colsize!(fig.layout, 1, Relative(0.5))
colsize!(fig.layout, 2, Relative(0.1))
colsize!(fig.layout, 3, Relative(0.4))

colgap!(fig.layout, 1, Relative(-0.05))
rowgap!(fig.layout, 1, Relative(-0.05))

resize_to_layout!(fig)

display(fig)

save(plotsdir(inputDir, "edgeLaplacianShearStressCellRemoved.png"), fig)
save(plotsdir(inputDir, "edgeLaplacianShearStressCellRemoved.pdf"), fig)

jldsave(datadir("referenceSystems", inputDir, "$(inputSystem)Ablated_testSystem.jld2"); R2,
    A2, 
    B2, 
    F2, 
    systemCOM2,
)
