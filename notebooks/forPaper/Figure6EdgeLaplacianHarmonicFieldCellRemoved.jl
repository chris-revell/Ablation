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
# ablatedCells = unique(findall(x->x!=0, neighbourMatrix[centralCell,:]))
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

Lprimal = edgeLaplacianPrimalHat(R2, A2, B2)
wLprimal = [col for col in  eachcol((eigen(Matrix(Lprimal))).vectors)]
λLprimal = (eigen(Matrix(Lprimal))).values
jᵖ = findPeripheralEdges(B2)
𝐜ⱼ = findEdgeMidpoints(R2, A2)[jᵖ.==0]
𝐂ⱼ = findCellLinkMidpoints(R2, A2, B2)[jᵖ.==0]
tⱼ = findEdgeLengths(R2, A2)[jᵖ.==0]
cellPolygons = findCellPolygons(R2, A2, B2)
edgeQuadrilaterals = findEdgeQuadrilaterals(R2, A2, B2)[jᵖ.==0]
radii = norm.([𝐜ⱼ[j].-systemCOM2 for j=1:(size(B2,2)-sum(jᵖ))])
dummyDists = collect(maximum(radii)/100:maximum(radii)/100:maximum(radii))

#%%

fig = Figure(size=(1200,500), fontsize=24)
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"(%$l)" for l in individualLetters]

# Only primal network 
zpar = 0.0
zperp = 1.0
χⱼ = abs.(wLprimal[1])./tⱼ
χⱼLims = (-2.0, log10(maximum(χⱼ)))
push!(axes, Axis(fig[1,1], aspect=DataAspect(), alignmode=Inside()))
# scatter!(axes[end], Point{2,Float64}(centralCellCOM), color=:red)
for i=1:size(B2,1)
    poly!(axes[end], cellPolygons[i], color=(:black, 0.25), strokecolor=(:black, 0.1), strokewidth=1.0)
end
j_ks = [findall(x->x!=0, A2[j,:]) for j=1:size(A2,1) if jᵖ[j]==0]
for j=1:(size(B2,2)-sum(jᵖ))
    poly!(axes[end], edgeQuadrilaterals[j], color=log10(χⱼ[j]), strokecolor=(:white, 0.0), strokewidth=0, colormap=Reverse(:devon), colorrange=χⱼLims)
    lines!(axes[end], Point{2,Float64}.(R2[j_ks[j]]), color=(:black, 0.5), linewidth=0.5)
end
scatter!(axes[end], [Point{2,Float64}(systemCOM2)], color=(:red,0.5), markersize=5)
hidedecorations!(axes[end])
hidespines!(axes[end])
Label(fig[2,1], popfirst!(subfigureLabels), fontsize=24)
Colorbar(fig[1,2], colorrange=χⱼLims, colormap=Reverse(:devon), height=Relative(0.8), label=L"\log_{10}\left(\chi_j\right)")

push!(axes, Axis(fig[1,3], aspect=AxisAspect(1), alignmode=Inside()))
edgeTangentsNormalised = normalize.(findEdgeTangents(R2, A2)[jᵖ.==0])
radiusVectorsNormalised = normalize.([𝐜ⱼ[j].-systemCOM2 for j=1:(size(B2,2)-sum(jᵖ))])
directions = abs.(radiusVectorsNormalised.⋅edgeTangentsNormalised)
scatter!(axes[end], log10.(radii), log10.(χⱼ), color=directions, colorrange=(0.0,1.0), colormap=:batlow)
lines!(axes[end], log10.(dummyDists), log10.(0.3./dummyDists), color=(:black, 0.75), linestyle=:dash)
lines!(axes[end], log10.(dummyDists), log10.(0.3./(dummyDists).^3), color=(:black, 0.75), linestyle=:dash)
ylims!(axes[end], (minimum(log10.(χⱼ)),1.0))
xlims!(axes[end], (minimum(log10.(radii)),maximum(log10.(radii))))
axes[end].xlabel = L"\log_{10}\left(c_j\right)"
axes[end].ylabel = L"\log_{10}\left(\chi_j\right)"
Label(fig[2,3], popfirst!(subfigureLabels), fontsize=24)
Colorbar(fig[1,4], colorrange=(0.0,1.0), colormap=:batlow, height=Relative(0.8), label=L"|\cos\left(\theta_j\right)|")

rowsize!(fig.layout, 1, Relative(0.99))
rowsize!(fig.layout, 2, Relative(0.01))
colsize!(fig.layout, 1, Relative(0.4))
colsize!(fig.layout, 2, Relative(0.1))
colsize!(fig.layout, 3, Relative(0.4))
colsize!(fig.layout, 2, Relative(0.1))

colgap!(fig.layout, 1, Relative(-0.05))
rowgap!(fig.layout, 1, Relative(-0.05))

resize_to_layout!(fig)

display(fig)

save(plotsdir(inputDir, "Figure6edgeLaplacianHarmonicFieldCellRemoved_$(inputSystem).png"), fig)
save(plotsdir(inputDir, "Figure6edgeLaplacianHarmonicFieldCellRemoved_$(inputSystem).pdf"), fig)
