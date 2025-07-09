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
using CircularArrays
using FromFile

println("Require for all i and all k:")
println("cocurlᶜ𝐜 = 2")
println("curlᶜ𝐜 = 0")
println("cocurlᵛ𝐂 = 2")
println("curlᵛ𝐂 = 0")

fig = Figure(size=(2000,1500))
axes = Axis[]

inputSystems = ["NoHole", "SingleHole", "DoubleHole"]

spokesOrNot = "spokes"
suppressOrNot = "suppress"

for (col, inputSystem) in enumerate(inputSystems)
    # Import system data
    if inputSystem == "OldSystem"
        # conditionsDict    = load(datadir("referenceSystems", "oldPaper", "dataFinal.jld2"))
        # @unpack  = conditionsDict["params"]
        matricesDict = load(datadir("referenceSystems", "oldPaper", "matricesFinal.jld2"))
        @unpack A,B,C,R,F,cellAreas,cellPressures,cellTensions,cellPerimeters = matricesDict["matrices"]
        cellEffectivePressures = cellPressures .- cellTensions.*cellPerimeters./(2.0.*cellAreas)
        dropzeros!(A)
        dropzeros!(B)
        dropzeros!(C)
    else 
        fileName = datadir("referenceSystems", "$(inputSystem)_testSystem5.jld2")
        importedData = load(fileName)
        R = importedData["R"]
        A = importedData["A"]
        B = importedData["B"]
        F = importedData["F"]
        cellTensions = importedData["cellTensions"]
        cellPressures = importedData["cellPressures"]
        cellPerimeters = importedData["cellPerimeters"]
        cellAreas = importedData["cellAreas"]
        cellEffectivePressures = cellPressures .+ cellTensions.*cellPerimeters./(2.0.*cellAreas)
    end

    𝐜ⱼ = findEdgeMidpoints(R, A)
    𝐂ⱼ = findCellLinkMidpoints(R, A, B)
    cellPolygons = findCellPolygons(R, A, B)
    cellLinkTriangles = findCellLinkTriangles(R, A, B)

    cocurlᶜ𝐜 = cocurlᶜ(R, A, B, 𝐜ⱼ)
    curlᶜ𝐜   = curlᶜ(R, A, B, 𝐜ⱼ)
    cocurlᵛ𝐂 = (spokesOrNot=="spokes" ? cocurlᵛspokes(R, A, B, 𝐂ⱼ) : cocurlᵛ(R, A, B, 𝐂ⱼ))
    curlᵛ𝐂   = (spokesOrNot=="spokes" ? curlᵛspokes(R, A, B, 𝐂ⱼ) : curlᵛ(R, A, B, 𝐂ⱼ))


    internalVertices = findBoundaryVertices(A, B).!=1
    internalCells = findBoundaryCells(B).!=1

    @show inputSystem
    # printstyled("cocurlᶜc max: $(maximum(cocurlᶜ𝐜)), min: $(minimum(cocurlᶜ𝐜))\n"; color = (abs(maximum(cocurlᶜ𝐜)-2.0) < 1e-8 && abs(minimum(cocurlᶜ𝐜)-2.0) > -1e-8 ? :green : :red))
    printstyled("cocurlᶜc max: $(maximum(cocurlᶜ𝐜[internalCells])), min: $(minimum(cocurlᶜ𝐜[internalCells]))\n"; color = (abs(maximum(cocurlᶜ𝐜[internalCells])-(-2.0)) < 1e-8 && abs(minimum(cocurlᶜ𝐜[internalCells])-(-2.0)) > -1e-8 ? :green : :red))
    # printstyled("curlᶜ𝐜 max: $(maximum(curlᶜ𝐜)), min: $(minimum(curlᶜ𝐜))\n"; color = (maximum(curlᶜ𝐜) < 1e-8 && minimum(curlᶜ𝐜) > -1e-8 ? :green : :red))
    printstyled("curlᶜ𝐜 max: $(maximum(curlᶜ𝐜[internalCells])), min: $(minimum(curlᶜ𝐜[internalCells]))\n"; color = (maximum(curlᶜ𝐜[internalCells]) < 1e-8 && minimum(curlᶜ𝐜[internalCells]) > -1e-8 ? :green : :red))
    # printstyled("cocurlᵛ𝐂 max: $(maximum(cocurlᵛ𝐂)), min: $(minimum(cocurlᵛ𝐂))\n"; color = (abs(maximum(cocurlᵛ𝐂)-2.0) < 1e-8 && abs(minimum(cocurlᵛ𝐂)-2.0) > -1e-8 ? :green : :red))
    printstyled("cocurlᵛ𝐂 max: $(maximum(cocurlᵛ𝐂[internalVertices])), min: $(minimum(cocurlᵛ𝐂[internalVertices]))\n"; color = (abs(maximum(cocurlᵛ𝐂[internalVertices])-(-2.0)) < 1e-8 && abs(minimum(cocurlᵛ𝐂[internalVertices])-(-2.0)) > -1e-8 ? :green : :red))
    # printstyled("curlᵛ𝐂 max: $(maximum(curlᵛ𝐂)), min: $(minimum(curlᵛ𝐂))\n"; color = (maximum(curlᵛ𝐂) < 1e-8 && minimum(curlᵛ𝐂) > -1e-8 ? :green : :red))
    printstyled("curlᵛ𝐂 max: $(maximum(curlᵛ𝐂[internalVertices])), min: $(minimum(curlᵛ𝐂[internalVertices]))\n"; color = (maximum(curlᵛ𝐂[internalVertices]) < 1e-8 && minimum(curlᵛ𝐂[internalVertices]) > -1e-8 ? :green : :red))

    max1 = max(0.0001, maximum(abs.(cocurlᶜ𝐜)))
    lims1 = (-max1, max1)
    push!(axes, Axis(fig[1, 2*col-1], aspect=DataAspect()))
    for i=1:size(B,1)
        poly!(axes[end], cellPolygons[i], color=cocurlᶜ𝐜[i], colorrange = lims1, colormap = :bwr, strokecolor=(:black, 1.0), strokewidth=2)
    end
    Colorbar(fig[1,2*col], colorrange=lims1, colormap=:bwr, height=Relative(0.8))
    Label(fig[1,2*col-1,Bottom()], L"\mathrm{cocurl}^c c",fontsize=24)

    max2 = max(0.0001, maximum(abs.(curlᶜ𝐜)))
    lims2 = (-max2, max2)
    push!(axes, Axis(fig[2, 2*col-1], aspect=DataAspect()))
    for i=1:size(B,1)
        poly!(axes[end], cellPolygons[i], color=curlᶜ𝐜[i], colorrange = lims2, colormap = :bwr, strokecolor=(:black, 1.0), strokewidth=2)
    end
    Colorbar(fig[2,2*col], colorrange=lims2, colormap=:bwr, height=Relative(0.8))
    Label(fig[2,2*col-1,Bottom()], L"\mathrm{curl}^c c",fontsize=24)
    
    max3 = max(0.0001, maximum(abs.(cocurlᵛ𝐂)))
    lims3 = (-max3, max3)
    push!(axes, Axis(fig[3, 2*col-1], aspect=DataAspect()))
    for k=1:size(A,2)
        poly!(axes[end], cellLinkTriangles[k], color=cocurlᵛ𝐂[k], colorrange = lims3, colormap = :bwr, strokecolor=(:white, 0.0), strokewidth=0)
    end
    for i=1:size(B,1)
        poly!(axes[end], cellPolygons[i], color=(:white, 0.0), strokecolor=(:black, 1.0), strokewidth=2)
    end
    Colorbar(fig[3,2*col], colorrange=lims3, colormap=:bwr, height=Relative(0.8))
    Label(fig[3,2*col-1,Bottom()], L"\mathrm{cocurl}^v C",fontsize=24)

    max4 = max(0.0001, maximum(abs.(curlᵛ𝐂)))
    lims4 = (-max4, max4)
    push!(axes, Axis(fig[4, 2*col-1], aspect=DataAspect()))
    for k=1:size(A,2)
        poly!(axes[end], cellLinkTriangles[k], color=curlᵛ𝐂[k], colorrange = lims4, colormap = :bwr, strokecolor=(:white, 0.0), strokewidth=0)
    end
    for i=1:size(B,1)
        poly!(axes[end], cellPolygons[i], color=(:white, 0.0), strokecolor=(:black, 1.0), strokewidth=2)
    end
    Colorbar(fig[4,2*col], colorrange=lims4, colormap=:bwr, height=Relative(0.8))
    Label(fig[4,2*col-1,Bottom()], L"\mathrm{curl}^v C",fontsize=24)

end

hidedecorations!.(axes)
hidespines!.(axes)
display(fig)
save(plotsdir("validation_$(suppressOrNot)_$(spokesOrNot).png"), fig)