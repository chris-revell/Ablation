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

neighbourMatrix = dropzeros(B*transpose(B))
ablatedCells = [centralCell]

if isfile(datadir("referenceSystems", inputDir, "$(inputSystem)Ablated_testSystem.jld2"))
    inFile = datadir("referenceSystems", inputDir, "$(inputSystem)Ablated_testSystem.jld2")
    importedData = load(inFile)
    @unpack R2, A2, B2, F2, systemCOM2 = importedData
else
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
                    )
    #%%
    params2, matrices2 = integ2.p
    @show maximum(norm.(sum(F2, dims=2)))
    R2 = reinterpret(SVector{2,Float64}, integ2.u)
    A2 = matrices2.A
    B2 = matrices2.B
    F2 = matrices2.F

    C = findC(A, B)
    centralCellVertices = R[findall(x->x!=0, C[centralCell, :])]
    systemCOM2 = sum(centralCellVertices)./length(centralCellVertices)
    
    jldsave(datadir("referenceSystems", inputDir, "$(inputSystem)Ablated_testSystem.jld2"); 
        R2,
        A2, 
        B2, 
        F2, 
        systemCOM2,
    )
end

cellCentres2 = findCellCentresOfMass(R2, A2, B2)

#%%
    
zperp = 1.0

ζᵐᵢ = ζ(R2, A2, B2, 1, zperp)

fig = Figure(size=(1000,500), fontsize=24)
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"(%$l)" for l in individualLetters]

radii = norm.([cellCentres2[i].-systemCOM2 for i=1:size(B2,1)])
dummyDists = collect(maximum(radii)/100:maximum(radii)/100:maximum(radii))

cellPolygons = findCellPolygons(R2, A2, B2)
iᵖ = findPeripheralCells(B2).==1

# 1st eigenmode
clims = (minimum(log10.(ζᵐᵢ[Not(iᵖ)])), maximum(log10.(ζᵐᵢ[Not(iᵖ)])))
push!(axes, Axis(fig[1,1], aspect=DataAspect(), alignmode=Inside()))
for i=1:size(B2,1)
    if iᵖ[i]==1
        poly!(axes[end], cellPolygons[i], color=(:black, 0.25), strokecolor=(:black, 0.1), strokewidth=0.5)
    else
        poly!(axes[end], cellPolygons[i], color=log10(ζᵐᵢ[i]), colorrange = clims, colormap = Reverse(:imola), strokecolor=(:black, 0.1), strokewidth=0.5)
    end
end
scatter!(axes[end], [Point{2,Float64}(systemCOM2)], color=(:red,0.5), markersize=5)
hidedecorations!(axes[end])
hidespines!(axes[end])
Label(fig[2,1], popfirst!(subfigureLabels), fontsize=24) 
Colorbar(fig[1,2], label=L"\log_{10}\left(|\zeta^{(1)}_i|\right)", colorrange=clims, colormap=Reverse(:imola), height=Relative(0.8), alignmode=Inside())

push!(axes, Axis(fig[1,3], aspect=AxisAspect(1), alignmode=Inside()))
scatter!(axes[end], log10.(radii[Not(iᵖ)]), log10.(ζᵐᵢ[Not(iᵖ)]), color=(:blue,0.3))
lines!(axes[end], log10.(dummyDists), log10.(0.15./dummyDists), color=(:black, 0.75), linestyle=:dash)
lines!(axes[end], log10.(dummyDists), log10.(0.15./(dummyDists).^3), color=(:black, 0.75), linestyle=:dash)
ylims!(axes[end], (minimum(log10.(ζᵐᵢ[Not(iᵖ)])),1.0))
xlims!(axes[end], (minimum(log10.(radii[Not(iᵖ)])),maximum(log10.(radii[Not(iᵖ)]))))
axes[end].xlabel = L"\log_{10}\left(R_i\right)"
axes[end].ylabel = L"\log_{10}\left(|\zeta^{(1)}_i|\right)"
Label(fig[2,3], popfirst!(subfigureLabels), fontsize=24) 

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


