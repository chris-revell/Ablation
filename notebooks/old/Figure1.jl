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

inputSystems = ["NoHole", "SingleHole", "DoubleHole", "Voronoi", "OldSystem"]

for inputSystem in inputSystems
    # Import system data
    if inputSystem == "OldSystem"
        conditionsDict    = load(datadir("referenceSystems", "oldPaper", "dataFinal.jld2"))
        @unpack nVerts,nCells,nEdges,pressureExternal,γ,λ,viscousTimeScale,realTimetMax,tMax,dt,outputInterval,outputTotal,realCycleTime,t1Threshold = conditionsDict["params"]
        matricesDict = load(datadir("referenceSystems", "oldPaper", "matricesFinal.jld2"))
        @unpack A,B,C,R,F,cellAreas,cellPressures,cellTensions,cellPerimeters = matricesDict["matrices"]
        cellEffectivePressures = cellPressures .- cellTensions.*cellPerimeters./(2.0.*cellAreas)
        dropzeros!(A)
        dropzeros!(B)
        dropzeros!(C)
    else 
        fileName = datadir("referenceSystems", "$(inputSystem)_testSystem4.jld2")
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
    
    nCells = size(B,1)
    nEDges = size(B,2)
    nVerts = size(A,2)
    cellAreas = findCellAreas(R, A, B)
    linkTriangles = findCellLinkTriangles(R, A, B)
    linkTriangleAreas = findCellLinkTriangleAreas(R, A, B)
    cellPolygons = findCellPolygons(R, A, B)
    𝐡 = hNetwork(R, A, B, F)

    curlᶜh = curlᶜ(R, A, B, 𝐡)
    curlᶜhLims = (-max(maximum(abs.(curlᶜh)), 0.1), max(maximum(abs.(curlᶜh)), 0.1))
    curlᵛh = curlᵛ(R, A, B, 𝐡)
    curlᵛhLims = (-maximum(abs.(curlᵛh)), maximum(abs.(curlᵛh)))
    divᶜh = divᶜ(R, A, B, 𝐡)
    divᶜhLims = (-maximum(abs.(divᶜh)), maximum(abs.(divᶜh)))
    divᵛh = divᵛ(R, A, B, 𝐡)
    divᵛhLims = (-maximum(abs.(divᵛh)), maximum(abs.(divᵛh)))
    cocurlᶜh = cocurlᶜ(R, A, B, 𝐡)
    cocurlᶜhLims = (-maximum(abs.(cocurlᶜh)), maximum(abs.(cocurlᶜh)))
    cocurlᵛh = cocurlᵛ(R, A, B, 𝐡)
    cocurlᵛhLims = (-maximum(abs.(cocurlᵛh)), maximum(abs.(cocurlᵛh)))
    codivᶜh = codivᶜ(R, A, B, 𝐡)
    codivᶜhLims = (-maximum(abs.(codivᶜh)), maximum(abs.(codivᶜh)))
    codivᵛh = codivᵛ(R, A, B, 𝐡)
    codivᵛhLims = (-maximum(abs.(codivᵛh)), maximum(abs.(codivᵛh)))
    # derivs = [curlᶜh, curlᵛh, divᶜh, divᵛh, cocurlᶜh, cocurlᵛh, codivᶜh, codivᵛh]

    #%%

    fig = Figure(size=(1000,2000))
    axes = Axis[]

    push!(axes, Axis(fig[1,1], aspect=DataAspect()))
    for i=1:nCells
        poly!(axes[end],cellPolygons[i],color=cocurlᶜh[i],colorrange=cocurlᶜhLims,colormap=:bwr,strokewidth=1,strokecolor=(:black,0.25))
    end
    Colorbar(fig[1,2], colorrange=cocurlᶜhLims, colormap=:bwr)
    Label(fig[1,1,Bottom()],L"\{cocurl^c h\}_i",fontsize = 24)

    push!(axes, Axis(fig[2,1], aspect=DataAspect()))
    for k=1:nVerts
        poly!(axes[end],linkTriangles[k],color=cocurlᵛh[k],colorrange=cocurlᵛhLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
    end
    for i=1:nCells
        poly!(axes[end],cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
    end
    Colorbar(fig[2,2],limits=cocurlᵛhLims,colormap=:bwr)
    Label(fig[2,1,Bottom()], L"\{cocurl^v \breve{h}\}_k", fontsize = 24)

    push!(axes, Axis(fig[3,1], aspect=DataAspect()))
    for i=1:nCells
        poly!(axes[end],cellPolygons[i],color=curlᶜh[i],colorrange=curlᶜhLims,colormap=:bwr,strokewidth=1,strokecolor=(:black,0.25))
    end
    Colorbar(fig[3,2], colorrange=curlᶜhLims, colormap=:bwr)
    Label(fig[3,1,Bottom()],L"\{curl^c h\}_i",fontsize = 24)

    push!(axes, Axis(fig[4,1], aspect=DataAspect()))
    for k=1:nVerts
        poly!(axes[end],linkTriangles[k],color=curlᵛh[k],colorrange=curlᵛhLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
    end
    for i=1:nCells
        poly!(axes[end],cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
    end
    Colorbar(fig[4,2],limits=curlᵛhLims,colormap=:bwr)
    Label(fig[4,1,Bottom()], L"\{curl^v \breve{h}\}_k", fontsize = 24)



    push!(axes, Axis(fig[1,3], aspect=DataAspect()))
    for i=1:nCells
        poly!(axes[end],cellPolygons[i],color=-divᶜh[i],colorrange=divᶜhLims,colormap=:bwr,strokewidth=1,strokecolor=(:black,0.25))
    end
    Colorbar(fig[1,4], colorrange=divᶜhLims, colormap=:bwr)
    Label(fig[1,3,Bottom()],L"-\{div^c h\}_i",fontsize = 24)

    push!(axes, Axis(fig[2,3], aspect=DataAspect()))
    @show divᵛh
    for k=1:nVerts
        poly!(axes[end],linkTriangles[k],color=-divᵛh[k],colorrange=divᵛhLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
    end
    for i=1:nCells
        poly!(axes[end],cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
    end
    Colorbar(fig[2,4],limits=divᵛhLims,colormap=:bwr)
    Label(fig[2,3,Bottom()], L"-\{div^v \breve{h}\}_k", fontsize = 24)

    push!(axes, Axis(fig[3,3], aspect=DataAspect()))
    for i=1:nCells
        poly!(axes[end],cellPolygons[i],color=codivᶜh[i],colorrange=codivᶜhLims,colormap=:bwr,strokewidth=1,strokecolor=(:black,0.25))
    end
    Colorbar(fig[3,4], colorrange=codivᶜhLims, colormap=:bwr)
    Label(fig[3,3,Bottom()],L"\{cod^c h\}_i",fontsize = 24)

    push!(axes, Axis(fig[4,3], aspect=DataAspect()))
    for k=1:nVerts
        poly!(axes[end],linkTriangles[k],color=codivᵛh[k],colorrange=codivᵛhLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
    end
    for i=1:nCells
        poly!(axes[end],cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
    end
    Colorbar(fig[4,4],limits=codivᵛhLims,colormap=:bwr)
    Label(fig[4,3,Bottom()], L"\{cod^v \breve{h}\}_k", fontsize = 24)


    hidedecorations!.(axes)
    hidespines!.(axes)
    display(fig)
    save(datadir("Figure1$(inputSystem)Derivatives.png"), fig)
end