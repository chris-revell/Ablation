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

inputSystems = ["Large1", "Large3", "Large4"]
inputDir = "quadraticPotentialNoPressureMultiples"; isdir(plotsdir(inputDir)) ? nothing : mkdir(plotsdir(inputDir))
# inputDir = "quadraticPotentialWithPressure"; isdir(plotsdir(inputDir)) ? nothing : mkdir(plotsdir(inputDir))

fig = Figure(size=(1500,2000), fontsize=24)
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"(%$l)" for l in individualLetters]

for (col,inputSystem) in enumerate(inputSystems)

    fileName = datadir("referenceSystems", inputDir, "$(inputSystem)_testSystem.jld2")
    importedData = load(fileName)
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

    I = size(B2,1)
    J = size(B2,2)
    K = size(A2,2)

    cellCentres1 = findCellCentresOfMass(R, A, B)
    cellCentres2 = findCellCentresOfMass(R2, A2, B2)

    cellPolygons2 = findCellPolygons(R2, A2, B2)

    𝐜ⱼ1 = findEdgeMidpoints(R, A)
    𝐜ⱼ2 = findEdgeMidpoints(R2, A2)

    cellradii2 = norm.([cellCentres2[i].-systemCOM2 for i=1:size(B2,1)])
    cellDummyDists2 = collect(maximum(cellradii2)/100:maximum(cellradii2)/100:maximum(cellradii2))
    vertexradii2 = norm.([R2[k].-systemCOM2 for k=1:size(A2,2)])
    vertexDummyDists2 = collect(maximum(vertexradii2)/100:maximum(vertexradii2)/100:maximum(vertexradii2))
    edgeradii2 = norm.([𝐜ⱼ2[j].-systemCOM2 for j=1:size(A2,1)])
    edgeDummyDists2 = collect(maximum(edgeradii2)/100:maximum(edgeradii2)/100:maximum(edgeradii2))
    ΔRₖ = R2.-R 
    Δrᵢ = cellCentres2.-cellCentres1[Not(ablatedCells)]
    Δrⱼ = 𝐜ⱼ2.-𝐜ⱼ1

    directions = [normalize(Δrᵢ[i])⋅normalize(cellCentres2[i].-systemCOM2) for i=1:I]
    clims = (-1.0,1.0)

    push!(axes, Axis(fig[1,col][1,1], aspect=DataAspect()))
    for i=1:I 
        poly!(axes[end], cellPolygons2[i], color=(:black, 0.2), strokewidth=1, strokecolor=(:black,0.2))
    end
    arrows!(axes[end], Point{2,Float64}.(cellCentres2), Vec{2,Float64}.(Δrᵢ), color=directions, colormap=:bam, colorrange=clims, lengthscale=1.0, align=:head)
    hidedecorations!(axes[end])
    hidespines!(axes[end])
    Label(fig[1,col,Bottom()], popfirst!(subfigureLabels), fontsize=24) 
    Colorbar(fig[1,col][1,2], colormap=:bam, colorrange=clims, height=Relative(0.7))

    push!(axes, Axis(fig[2,col], aspect=AxisAspect(1.25)))
    scatter!(axes[end], log10.(cellradii2), log10.(norm.(Δrᵢ)), color=directions, colormap=:bam, colorrange=clims)
    lines!(axes[end], log10.(cellDummyDists2), log10.(0.02./cellDummyDists2), color=(:black, 0.75), linestyle=:dash)
    axes[end].xlabel = L"\log_{10}\left(R_i\right)"
    axes[end].ylabel = L"\log_{10}\left(\Delta R_i\right)"
    ylims!(axes[end], (-4.2, -1.0))
    xlims!(axes[end], (-0.8, 1.0))
    axes[end].xticks = (-0.8:0.8:0.8, string.(-0.8:0.8:0.8))
    Label(fig[2,col,Bottom()], popfirst!(subfigureLabels), fontsize=24) 
end




inputSystems = ["Large5", "Large6", "Large7"]
for (col,inputSystem) in enumerate(inputSystems)

    fileName = datadir("referenceSystems", inputDir, "$(inputSystem)_testSystem.jld2")
    importedData = load(fileName)
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

    I = size(B2,1)
    J = size(B2,2)
    K = size(A2,2)

    cellCentres1 = findCellCentresOfMass(R, A, B)
    cellCentres2 = findCellCentresOfMass(R2, A2, B2)

    cellPolygons2 = findCellPolygons(R2, A2, B2)

    𝐜ⱼ1 = findEdgeMidpoints(R, A)
    𝐜ⱼ2 = findEdgeMidpoints(R2, A2)

    cellradii2 = norm.([cellCentres2[i].-systemCOM2 for i=1:size(B2,1)])
    cellDummyDists2 = collect(maximum(cellradii2)/100:maximum(cellradii2)/100:maximum(cellradii2))
    vertexradii2 = norm.([R2[k].-systemCOM2 for k=1:size(A2,2)])
    vertexDummyDists2 = collect(maximum(vertexradii2)/100:maximum(vertexradii2)/100:maximum(vertexradii2))
    edgeradii2 = norm.([𝐜ⱼ2[j].-systemCOM2 for j=1:size(A2,1)])
    edgeDummyDists2 = collect(maximum(edgeradii2)/100:maximum(edgeradii2)/100:maximum(edgeradii2))
    ΔRₖ = R2.-R 
    Δrᵢ = cellCentres2.-cellCentres1[Not(ablatedCells)]
    Δrⱼ = 𝐜ⱼ2.-𝐜ⱼ1

    directions = [normalize(Δrᵢ[i])⋅normalize(cellCentres2[i].-systemCOM2) for i=1:I]
    clims = (-1.0,1.0)

    push!(axes, Axis(fig[3,col][1,1], aspect=DataAspect()))
    for i=1:I 
        poly!(axes[end], cellPolygons2[i], color=(:black, 0.2), strokewidth=1, strokecolor=(:black,0.2))
    end
    arrows!(axes[end], Point{2,Float64}.(cellCentres2), Vec{2,Float64}.(Δrᵢ), color=directions, colormap=:bam, colorrange=clims, lengthscale=1.0, align=:head)
    hidedecorations!(axes[end])
    hidespines!(axes[end])
    Label(fig[3,col,Bottom()], popfirst!(subfigureLabels), fontsize=24) 
    Colorbar(fig[3,col][1,2], colormap=:bam, colorrange=clims, height=Relative(0.7))

    push!(axes, Axis(fig[4,col], aspect=AxisAspect(1.25)))
    scatter!(axes[end], log10.(cellradii2), log10.(norm.(Δrᵢ)), color=directions, colormap=:bam, colorrange=clims)
    lines!(axes[end], log10.(cellDummyDists2), log10.(0.02./cellDummyDists2), color=(:black, 0.75), linestyle=:dash)
    axes[end].xlabel = L"\log_{10}\left(R_i\right)"
    axes[end].ylabel = L"\log_{10}\left(\Delta R_i\right)"
    ylims!(axes[end], (-4.2, -1.0))
    xlims!(axes[end], (-0.8, 1.0))
    axes[end].xticks = (-0.8:0.8:0.8, string.(-0.8:0.8:0.8))
    Label(fig[4,col,Bottom()], popfirst!(subfigureLabels), fontsize=24) 
end



# rowsize!(fig.layout, 1, Relative(0.95))
# rowsize!(fig.layout, 2, Relative(0.05))
# # rowsize!(fig.layout, 3, Relative(0.4))
# # rowsize!(fig.layout, 4, Relative(0.1))
# colsize!(fig.layout, 1, Relative(0.45))
# colsize!(fig.layout, 2, Relative(0.5))
# colsize!(fig.layout, 3, Relative(0.05))
# # colsize!(fig.layout, 3, Relative(0.4))

# colgap!(fig.layout, 1, Relative(-0.05))
# rowgap!(fig.layout, 1, Relative(-0.01))
# rowgap!(fig.layout, 3, Relative(-0.01))

# resize_to_layout!(fig)

display(fig)

save(plotsdir(inputDir, "displacement.png"), fig)
save(plotsdir(inputDir, "displacement.pdf"), fig)


