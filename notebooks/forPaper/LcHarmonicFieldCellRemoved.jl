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

    jldsave(datadir("referenceSystems", inputDir, "$(inputSystem)Ablated_testSystem.jld2"); R2,
        A2, 
        B2, 
        F2, 
        systemCOM2,
    )
end

Lprimal = edgeLaplacianPrimalHat(R2, A2, B2)
wLprimal = [col for col in  eachcol((eigen(Matrix(Lprimal))).vectors)]
λLprimal = (eigen(Matrix(Lprimal))).values

jᵖ = findPeripheralEdges(B2)
𝐜ⱼ = findEdgeMidpoints(R2, A2)[jᵖ.==0]
𝐂ⱼ = findCellLinkMidpoints(R2, A2, B2)[jᵖ.==0]
tⱼ = findEdgeLengths(R2, A2)[jᵖ.==0]

#%%

fig = Figure(size=(1000,500), fontsize=24)
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"(%$l)" for l in individualLetters]

radii = norm.([𝐜ⱼ[j].-systemCOM2 for j=1:(size(B2,2)-sum(jᵖ))])
dummyDists = collect(maximum(radii)/100:maximum(radii)/100:maximum(radii))

cellPolygons = findCellPolygons(R2, A2, B2)
edgeQuadrilaterals = findEdgeQuadrilaterals(R2, A2, B2)[jᵖ.==0]


# Primal network 
zpar = 0.0
zperp = 1.0
# edgeVectorsPrimal = wLprimal[1].*(zpar.*primalBasisParallel .+ zperp.*primalBasisPerp)
edgeVectorsPrimalNorms = abs.(wLprimal[1])./tⱼ
# @show edgeVectorsPrimalNorms .- norm.(edgeVectorsPrimal)
edgeVectorsPrimalNormsLims = (-2.0, log10(maximum(edgeVectorsPrimalNorms)))
push!(axes, Axis(fig[1,1], aspect=DataAspect(), alignmode=Inside()))
# scatter!(axes[end], Point{2,Float64}(centralCellCOM), color=:red)
for j=1:(size(B2,2)-sum(jᵖ))
    poly!(axes[end], edgeQuadrilaterals[j], color=log10(edgeVectorsPrimalNorms[j]), strokecolor=(:white, 0.0), strokewidth=0, colormap=Reverse(:devon), colorrange=edgeVectorsPrimalNormsLims)
end
for i=1:size(B2,1)
    poly!(axes[end], cellPolygons[i], color=(:white, 0.0), strokecolor=(:black, 0.2), strokewidth=0.5)
end
scatter!(axes[end], [Point{2,Float64}(systemCOM2)], color=(:red,0.5), markersize=5)
hidedecorations!(axes[end])
hidespines!(axes[end])
Label(fig[2,1], popfirst!(subfigureLabels), fontsize=24) #"Single hole, 1st eigenmode, primal network, zpar=$(zpar), zperp=$(zperp)")
Colorbar(fig[1,2], colorrange=edgeVectorsPrimalNormsLims, colormap=Reverse(:devon), height=Relative(0.8), label=L"\log_{10}\left(\chi_j\right)")
push!(axes, Axis(fig[1,3], xscale=log10, yscale=log10, aspect=AxisAspect(1), alignmode=Inside()))
scatter!(axes[end], radii, edgeVectorsPrimalNorms, color=(:blue,0.1))
lines!(axes[end], dummyDists, 0.3./dummyDists, color=(:black, 0.75), linestyle=:dash)
# lines!(axes[end], dummyDists, 1.0./(dummyDists).^2, color=(:black, 0.75), linestyle=:dash)
lines!(axes[end], dummyDists, 0.3./(dummyDists).^3, color=(:black, 0.75), linestyle=:dash)
ylims!(axes[end], (minimum(edgeVectorsPrimalNorms),1.0))
xlims!(axes[end], (minimum(radii),maximum(radii)))
axes[end].xlabel = L"r_j"
axes[end].ylabel = L"\log_{10}\left(\chi_j\right)"
Label(fig[2,3], popfirst!(subfigureLabels), fontsize=24) #"Single hole, 1st eigenmode, primal network, zpar=$(zpar), zperp=$(zperp)")

rowsize!(fig.layout, 1, Relative(0.99))
rowsize!(fig.layout, 2, Relative(0.01))
colsize!(fig.layout, 1, Relative(0.5))
colsize!(fig.layout, 2, Relative(0.1))
colsize!(fig.layout, 3, Relative(0.4))

colgap!(fig.layout, 1, Relative(-0.05))
rowgap!(fig.layout, 1, Relative(-0.05))

resize_to_layout!(fig)

display(fig)

save(plotsdir(inputDir, "edgeLaplacianHarmonicFieldCellRemoved_$(inputSystem).png"), fig)
save(plotsdir(inputDir, "edgeLaplacianHarmonicFieldCellRemoved_$(inputSystem).pdf"), fig)
