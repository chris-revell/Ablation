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
using UnPack

@from "$(srcdir("AblateCells.jl"))" using AblateCells
@from "$(srcdir("Stresses.jl"))" using Stresses

inputSystem = "Large4"
inputDir = "quadraticPotentialNoPressure"; isdir(plotsdir(inputDir)) ? nothing : mkdir(plotsdir(inputDir))
# inputDir = "quadraticPotentialWithPressure"; isdir(plotsdir(inputDir)) ? nothing : mkdir(plotsdir(inputDir))

fileName = datadir("referenceSystems", inputDir, "$(inputSystem)_testSystem.jld2")
importedData = load(fileName)
@unpack R, A, B, F = importedData


if isfile(datadir("referenceSystems", inputDir, "$(inputSystem)Ablated2_testSystem.jld2"))
    inFile = datadir("referenceSystems", inputDir, "$(inputSystem)Ablated2_testSystem.jld2")
    importedData = load(inFile)
    @unpack R2, A2, B2, F2, R3, A3, B3, F3 = importedData
else
    𝐡 = hNetwork(R, A, B, F)
    cocurlᶜh = cocurlᶜ(R, A, B, 𝐡)
    C = findC(A, B)
    i_min = findmin(cocurlᶜh)[2]
    i_max = findmax(cocurlᶜh)[2]
    Rtmp, Atmp, Btmp = ablateCells(R, A, B, [i_min])
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
    params2, matrices2 = integ2.p
    # @show maximum(norm.(sum(F2, dims=2)))
    R2 = reinterpret(SVector{2,Float64}, integ2.u)
    A2 = matrices2.A
    B2 = matrices2.B
    F2 = matrices2.F
    ablatedCellVertices = R[findall(x->x!=0, C[i_min, :])]
    ablationCentre2 = sum(ablatedCellVertices)./length(ablatedCellVertices)
    Rtmp, Atmp, Btmp = ablateCells(R, A, B, [i_max])
    integ3 = vertexModel(abstol = 1e-9,
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
    params3, matrices3 = integ3.p
    # @show maximum(norm.(sum(F2, dims=2)))
    R3 = reinterpret(SVector{2,Float64}, integ3.u)
    A3 = matrices3.A
    B3 = matrices3.B
    F3 = matrices3.F
    ablatedCellVertices = R[findall(x->x!=0, C[i_max, :])]
    ablationCentre3 = sum(ablatedCellVertices)./length(ablatedCellVertices)
    jldsave(datadir("referenceSystems", inputDir, "$(inputSystem)Ablated2_testSystem.jld2"); 
        R2,
        A2, 
        B2, 
        F2, 
        ablationCentre2,
        R3,
        A3, 
        B3, 
        F3,
        ablationCentre3,
    )
end

I = size(B2,1)
J = size(B2,2)
K = size(A2,2)

cellCentres1 = findCellCentresOfMass(R, A, B)
cellCentres2 = findCellCentresOfMass(R2, A2, B2)
cellCentres3 = findCellCentresOfMass(R3, A3, B3)

cellPolygons2 = findCellPolygons(R2, A2, B2)
cellPolygons3 = findCellPolygons(R3, A3, B3)

𝐜ⱼ1 = findEdgeMidpoints(R, A)
𝐜ⱼ2 = findEdgeMidpoints(R2, A2)
𝐜ⱼ3 = findEdgeMidpoints(R3, A3)

# cellradii2 = norm.([cellCentres2[i].-ablationCentre2 for i=1:size(B2,1)])
# cellDummyDists2 = collect(maximum(cellradii2)/100:maximum(cellradii2)/100:maximum(cellradii2))
# vertexradii2 = norm.([R2[k].-ablationCentre2 for k=1:size(A2,2)])
# vertexDummyDists2 = collect(maximum(vertexradii2)/100:maximum(vertexradii2)/100:maximum(vertexradii2))
# edgeradii2 = norm.([𝐜ⱼ2[j].-ablationCentre2 for j=1:size(A2,1)])
# edgeDummyDists2 = collect(maximum(edgeradii2)/100:maximum(edgeradii2)/100:maximum(edgeradii2))
# ΔRₖ = R2.-R 
Δrᵢ2 = cellCentres2.-cellCentres1[Not(i_min)]
Δrᵢ3 = cellCentres3.-cellCentres1[Not(i_max)]
# Δrⱼ = 𝐜ⱼ2.-𝐜ⱼ1

directions2 = [normalize(Δrᵢ2[i])⋅normalize(cellCentres2[i].-ablationCentre2) for i=1:I]
directions3 = [normalize(Δrᵢ3[i])⋅normalize(cellCentres3[i].-ablationCentre3) for i=1:I]
clims = (-1.0,1.0)


fig = Figure(size=(1000,500), fontsize=24)
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"(%$l)" for l in individualLetters]


push!(axes, Axis(fig[1,1], aspect=DataAspect()))
for i=1:I 
    poly!(axes[end], cellPolygons2[i], color=(:black, 0.1), strokewidth=1, strokecolor=(:black,0.1))
end
arrows!(axes[end], Point{2,Float64}.(cellCentres2), Vec{2,Float64}.(Δrᵢ2), color=directions2, colormap=:bam, colorrange=clims, lengthscale=1.0)
hidedecorations!(axes[end])
hidespines!(axes[end])
Label(fig[1,1,Bottom()], popfirst!(subfigureLabels), fontsize=24) 
# Colorbar(fig[1,col][1,2], colormap=:bam, colorrange=clims, height=Relative(0.7))
push!(axes, Axis(fig[1,2], aspect=DataAspect()))
for i=1:I 
    poly!(axes[end], cellPolygons3[i], color=(:black, 0.1), strokewidth=1, strokecolor=(:black,0.1))
end
arrows!(axes[end], Point{2,Float64}.(cellCentres3), Vec{2,Float64}.(Δrᵢ3), color=directions3, colormap=:bam, colorrange=clims, lengthscale=1.0)
hidedecorations!(axes[end])
hidespines!(axes[end])
Label(fig[1,2,Bottom()], popfirst!(subfigureLabels), fontsize=24) 
# Colorbar(fig[1,col][1,2], colormap=:bam, colorrange=clims, height=Relative(0.7))



display(fig)

save(plotsdir(inputDir, "displacementCellRemovedByStress$(inputSystem).png"), fig)
save(plotsdir(inputDir, "displacementCellRemovedByStress$(inputSystem).pdf"), fig)


