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

# inputSystems = ["OldSystem", "SingleHole", "DoubleHole"]#, "Voronoi", "OldSystem"]
inputSystems = ["Voronoi", "DoubleHole"]#, "Voronoi", "OldSystem"]
# inputSystems = ["OldSystem", "SingleHole", "DoubleHole"]#, "Voronoi", "OldSystem"]

fig = Figure(size=(1500, 1600))
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"($l)" for l in individualLetters]

gl0 = GridLayout(fig[1,1])
Label(gl0[0,2], "", fontsize=24, rotation=π/2)
Label(gl0[1,2], "Cells", fontsize=24, rotation=π/2)
Label(gl0[2,2], "Vertices", fontsize=24, rotation=π/2)
Label(gl0[3,2], "Cells", fontsize=24, rotation=π/2)
Label(gl0[4,2], "Vertices", fontsize=24, rotation=π/2)
Label(gl0[1:2,1], "Divergences", fontsize=24, rotation=π/2)
Label(gl0[3:4,1], "Curls", fontsize=24, rotation=π/2)
rowsize!(gl0, 0, Relative(0.04))
rowsize!(gl0, 1, Relative(0.24))
rowsize!(gl0, 2, Relative(0.24))
rowsize!(gl0, 3, Relative(0.24))
rowsize!(gl0, 4, Relative(0.24))


for (col,inputSystem) in enumerate(inputSystems)
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
        fileName = datadir("referenceSystems", "$(inputSystem)_testSystem.jld2")
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
    # for _ = 1:10
    #     𝐡 .+= hNetwork(R, A, B, F)
    # end
    # 𝐡 ./= 11.0

    boundaryVertices = findBoundaryVertices(A, B)
    boundaryCells = findBoundaryCells(B)

    curlᶜh = curlᶜ(R, A, B, 𝐡)
    curlᶜh[boundaryCells] .= 0
    curlᶜhLims = (-max(maximum(abs.(curlᶜh)), 0.1), max(maximum(abs.(curlᶜh)), 0.1))
    # curlᶜhLims = (-0.04, 0.04)
    curlᵛh = curlᵛ(R, A, B, 𝐡)
    curlᵛh[boundaryVertices.==1] .= 0
    curlᵛhLims = (-maximum(abs.(curlᵛh)), maximum(abs.(curlᵛh)))
    # curlᵛhLims = (-0.04, 0.04)
    divᶜh = divᶜ(R, A, B, 𝐡)
    divᶜh[boundaryCells] .= 0
    divᶜhLims = (-maximum(abs.(divᶜh)), maximum(abs.(divᶜh)))
    # divᶜhLims = (-0.75, 0.75)
    divᵛh = divᵛ(R, A, B, 𝐡)
    divᵛh[boundaryVertices.==1] .= 0
    divᵛhLims = (-maximum(abs.(divᵛh)), maximum(abs.(divᵛh)))
    # divᵛhLims = (-0.75, 0.75)
    cocurlᶜh = cocurlᶜ(R, A, B, 𝐡)
    cocurlᶜh[boundaryCells] .= 0
    cocurlᶜhLims = (-maximum(abs.(cocurlᶜh)), maximum(abs.(cocurlᶜh)))
    # cocurlᶜhLims = (-0.75, 0.75)
    cocurlᵛh = cocurlᵛ(R, A, B, 𝐡)
    cocurlᵛh[boundaryVertices.==1] .= 0
    cocurlᵛhLims = (-maximum(abs.(cocurlᵛh)), maximum(abs.(cocurlᵛh)))
    # cocurlᵛhLims = (-0.75, 0.75)
    codᶜh = codᶜ(R, A, B, 𝐡)
    codᶜh[boundaryCells] .= 0
    codᶜhLims = (-maximum(abs.(codᶜh)), maximum(abs.(codᶜh)))
    # codᶜhLims = (-0.04, 0.04)
    codᵛh = codᵛ(R, A, B, 𝐡)
    codᵛh[boundaryVertices.==1] .= 0
    codᵛhLims = (-maximum(abs.(codᵛh)), maximum(abs.(codᵛh)))
    # codᵛhLims = (-0.04, 0.04)

    gl = GridLayout(fig[1,1+col*2-1])

    push!(axes, Axis(gl[1,1], aspect=DataAspect()))
    for i=1:nCells
        poly!(axes[end],cellPolygons[i],color=cocurlᶜh[i],colorrange=cocurlᶜhLims,colormap=:bwr,strokewidth=1,strokecolor=(:black,0.25))
    end
    Colorbar(gl[1,2], colorrange=cocurlᶜhLims, colormap=:bwr)
    # Label(gl[1,1,Bottom()], L"\mathrm{curl}^c\, \mathsfbf{h}",fontsize=24)
    Label(gl[1,1,Bottom()], L"\mathrm{cocurl}^c\, \mathbf{h}",fontsize=24)

    push!(axes, Axis(gl[1,3], aspect=DataAspect()))
    for i=1:nCells
        poly!(axes[end],cellPolygons[i],color=-divᶜh[i],colorrange=divᶜhLims,colormap=:bwr,strokewidth=1,strokecolor=(:black,0.25))
    end
    Colorbar(gl[1,4], colorrange=divᶜhLims, colormap=:bwr)
    # Label(gl[1,3,Bottom()], L"-\mathrm{div}^c\, \mathsfbf{h}",fontsize=24)
    Label(gl[1,3,Bottom()], L"-\mathrm{div}^c\, \mathbf{h}",fontsize=24)

    push!(axes, Axis(gl[2,1], aspect=DataAspect()))
    for k=1:nVerts
        poly!(axes[end],linkTriangles[k],color=-divᵛh[k],colorrange=divᵛhLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
    end
    for i=1:nCells
        poly!(axes[end],cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
    end
    Colorbar(gl[2,2],limits=divᵛhLims,colormap=:bwr)
    # Label(gl[2,1,Bottom()], L"-\mathrm{div}^v\, \mathsfbf{h}", fontsize=24)
    Label(gl[2,1,Bottom()], L"-\mathrm{div}^v\, \mathbf{h}", fontsize=24)

    push!(axes, Axis(gl[2,3], aspect=DataAspect()))
    for k=1:nVerts
        poly!(axes[end],linkTriangles[k],color=cocurlᵛh[k],colorrange=cocurlᵛhLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
    end
    for i=1:nCells
        poly!(axes[end],cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
    end
    Colorbar(gl[2,4],limits=cocurlᵛhLims,colormap=:bwr)
    # Label(gl[2,3,Bottom()], L"\mathrm{cocurl}^v\, \mathsfbf{h}", fontsize=24)
    Label(gl[2,3,Bottom()], L"\mathrm{cocurl}^v\, \mathbf{h}", fontsize=24)

    push!(axes, Axis(gl[3,1], aspect=DataAspect()))
    for i=1:nCells
        poly!(axes[end],cellPolygons[i],color=curlᶜh[i],colorrange=curlᶜhLims,colormap=:bwr,strokewidth=1,strokecolor=(:black,0.25))
    end
    Colorbar(gl[3,2], colorrange=curlᶜhLims, colormap=:bwr)
    # Label(gl[3,1,Bottom()], L"\mathrm{curl}^c\, \mathsfbf{h}",fontsize=24)
    Label(gl[3,1,Bottom()], L"\mathrm{curl}^c\, \mathbf{h}",fontsize=24)

    push!(axes, Axis(gl[3,3], aspect=DataAspect()))
    for i=1:nCells
        poly!(axes[end],cellPolygons[i],color=codᶜh[i],colorrange=codᶜhLims,colormap=:bwr,strokewidth=1,strokecolor=(:black,0.25))
    end
    Colorbar(gl[3,4], colorrange=codᶜhLims, colormap=:bwr)
    # Label(gl[3,3,Bottom()], L"\mathrm{cod}^c\, \mathsfbf{h}",fontsize=24)
    Label(gl[3,3,Bottom()], L"\mathrm{cod}^c\, \mathbf{h}",fontsize=24)

    push!(axes, Axis(gl[4,1], aspect=DataAspect()))
    for k=1:nVerts
        poly!(axes[end],linkTriangles[k],color=codᵛh[k],colorrange=codᵛhLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
    end
    for i=1:nCells
        poly!(axes[end],cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
    end
    Colorbar(gl[4,2],limits=codᵛhLims,colormap=:bwr)
    # Label(gl[4,1,Bottom()], L"\mathrm{cod}^v\, \mathsfbf{h}", fontsize=24)
    Label(gl[4,1,Bottom()], L"\mathrm{cod}^v\, \mathbf{h}", fontsize=24)

    push!(axes, Axis(gl[4,3], aspect=DataAspect()))
    for k=1:nVerts
        poly!(axes[end],linkTriangles[k],color=curlᵛh[k],colorrange=curlᵛhLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
    end
    for i=1:nCells
        poly!(axes[end],cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
    end
    Colorbar(gl[4,4],limits=curlᵛhLims,colormap=:bwr)
    # Label(gl[4,3,Bottom()], L"\mathrm{curl}^v\, \mathrm{h}", fontsize=24)
    Label(gl[4,3,Bottom()], L"\mathrm{curl}^v\, \mathbf{h}", fontsize=24)

    Label(gl[0,1], "Primal", fontsize=24)
    Label(gl[0,3], "Dual", fontsize=24)
    
    rowsize!(gl, 0, Relative(0.04))
    rowsize!(gl, 1, Relative(0.24))
    rowsize!(gl, 2, Relative(0.24))
    rowsize!(gl, 3, Relative(0.24))
    rowsize!(gl, 4, Relative(0.24))

    colsize!(gl, 1, Relative(0.495))
    colsize!(gl, 2, Relative(0.005))
    colsize!(gl, 3, Relative(0.495))
    colsize!(gl, 4, Relative(0.005))
    
end

Label(fig[2,2, Top()], L"(a)", fontsize=36)
Label(fig[2,4, Top()], L"(b)", fontsize=36)
# Label(fig[2,1+5, Top()], "(c)", fontsize=36)
rowsize!(fig.layout, 1, Relative(0.95))
rowsize!(fig.layout, 2, Relative(0.05))

push!(axes, Axis(fig[:,3]))
vlines!(axes[end], [0.0], color=(:black,0.3), linewidth=5, linestyle=:solid)
# push!(axes, Axis(fig[:,1+4]))
# vlines!(axes[end], [0.0], color=(:black,0.3), linewidth=5, linestyle=:solid)

colsize!(fig.layout, 1, Relative(0.1))
colsize!(fig.layout, 2, Relative(0.44))
colsize!(fig.layout, 3, Relative(0.02))
colsize!(fig.layout, 4, Relative(0.44))
# colsize!(fig.layout, 1+4, Relative(0.05))
# resize_to_layout!(fig)

hidedecorations!.(axes)
hidespines!.(axes)
display(fig)
save(datadir("figureDerivatives2.png"), fig)