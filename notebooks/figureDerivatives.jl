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
# inputSystems = ["Voronoi", "DoubleHole"]#, "Voronoi", "OldSystem"]
inputSystems = ["SingleHole", "DoubleHole"]#, "Voronoi", "OldSystem"]
# inputSystems = ["OldSystem", "SingleHole", "DoubleHole"]#, "Voronoi", "OldSystem"]

fig = Figure(size=(1500, 1500))
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
        # conditionsDict    = load(datadir("referenceSystems", "oldPaper", "dataFinal.jld2"))
        # @unpack  = conditionsDict["params"]
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
    
    I = size(B,1)
    J = size(B,2)
    K = size(A,2)
    cellAreas = findCellAreas(R, A, B)
    linkTriangles = findCellLinkTriangles(R, A, B)
    linkTriangleAreas = findCellLinkTriangleAreas(R, A, B)
    cellPolygons = findCellPolygons(R, A, B)
    𝐡 = hNetwork(R, A, B, F)
    # 𝐡 = [SVector{2,Float64}(1.0,0.0) for _=1:J]
    
    boundaryVertices = findBoundaryVertices(A, B)
    boundaryCells = findBoundaryCells(B)

    curlᶜh = curlᶜ(R, A, B, 𝐡)
    # curlᶜh[boundaryCells] .= 0
    curlᶜhMax = max(maximum(abs.(curlᶜh)), 0.00001)
    curlᶜhLims = (-curlᶜhMax, curlᶜhMax)
    # curlᶜhLims = (-max(maximum(abs.(curlᶜh)), 0.1), max(maximum(abs.(curlᶜh)), 0.1))
    # curlᶜhLims = (-0.04, 0.04)
    curlᵛh = curlᵛboundary(R, A, B, 𝐡)
    # curlᵛh[boundaryVertices.==1] .= 0
    curlᵛhMax = max(maximum(abs.(curlᵛh)), 0.00001)
    curlᵛhLims = (-curlᵛhMax, curlᵛhMax)
    # curlᵛhLims = (-0.04, 0.04)
    divᶜh = divᶜ(R, A, B, 𝐡)
    # divᶜh[boundaryCells] .= 0
    divᶜhMax = max(maximum(abs.(divᶜh)), 0.00001)
    divᶜhLims = (-divᶜhMax, divᶜhMax)
    # divᶜhLims = (-0.75, 0.75)
    divᵛh = divᵛ(R, A, B, 𝐡)
    divᵛh[boundaryVertices.==1] .= 0
    divᵛhMax = max(maximum(abs.(divᵛh)), 0.00001)
    divᵛhLims = (-divᵛhMax, divᵛhMax)
    # divᵛhLims = (-0.75, 0.75)
    cocurlᶜh = cocurlᶜ(R, A, B, 𝐡)
    # cocurlᶜh[boundaryCells] .= 0
    cocurlᶜhMax = max(maximum(abs.(cocurlᶜh)), 0.00001)
    cocurlᶜhLims = (-cocurlᶜhMax, cocurlᶜhMax)
    # cocurlᶜhLims = (-0.75, 0.75)
    cocurlᵛh = cocurlᵛboundary(R, A, B, 𝐡)
    # cocurlᵛh[boundaryVertices.==1] .= 0
    cocurlᵛhMax = max(maximum(abs.(cocurlᵛh)), 0.00001)
    cocurlᵛhLims = (-cocurlᵛhMax, cocurlᵛhMax)
    # cocurlᵛhLims = (-0.75, 0.75)
    codivᶜh = codivᶜ(R, A, B, 𝐡)
    # codivᶜh[boundaryCells] .= 0
    codivᶜhMax = max(maximum(abs.(codivᶜh)), 0.00001)
    codivᶜhLims = (-codivᶜhMax, codivᶜhMax)
    # codivᶜhLims = (-0.04, 0.04)
    codivᵛh = codivᵛ(R, A, B, 𝐡)
    codivᵛh[boundaryVertices.==1] .= 0
    codivᵛhMax = max(maximum(abs.(codivᵛh)), 0.00001)
    codivᵛhLims = (-codivᵛhMax, codivᵛhMax)
    # codivᵛhLims = (-0.04, 0.04)

    gl = GridLayout(fig[1,1+col*2-1])

    push!(axes, Axis(gl[1,1], aspect=DataAspect()))
    for i=1:I
        poly!(axes[end],cellPolygons[i],color=cocurlᶜh[i],colorrange=cocurlᶜhLims,colormap=:bwr,strokewidth=1,strokecolor=(:black,0.25))
    end
    Colorbar(gl[1,2], colorrange=cocurlᶜhLims, colormap=:bwr, height=Relative(0.7))
    # Label(gl[1,1,Bottom()], L"\mathrm{curl}^c\, \mathsfbf{h}",fontsize=24)
    # Label(gl[1,1,Bottom()], L"\mathrm{cocurl}^c\, \mathbf{h}",fontsize=24)
    Label(gl[1,1,Bottom()], L"\mathrm{cocurl}^c",fontsize=24)

    push!(axes, Axis(gl[1,3], aspect=DataAspect()))
    for i=1:I
        poly!(axes[end],cellPolygons[i],color=-divᶜh[i],colorrange=divᶜhLims,colormap=:bwr,strokewidth=1,strokecolor=(:black,0.25))
    end
    Colorbar(gl[1,4], colorrange=divᶜhLims, colormap=:bwr, height=Relative(0.7))
    # Label(gl[1,3,Bottom()], L"-\mathrm{div}^c\, \mathsfbf{h}",fontsize=24)
    # Label(gl[1,3,Bottom()], L"-\mathrm{div}^c\, \mathbf{h}",fontsize=24)
    Label(gl[1,3,Bottom()], L"-\mathrm{div}^c",fontsize=24)

    push!(axes, Axis(gl[2,1], aspect=DataAspect()))
    for k=1:K
        poly!(axes[end],linkTriangles[k],color=-divᵛh[k],colorrange=divᵛhLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
    end
    for i=1:I
        poly!(axes[end],cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
    end
    Colorbar(gl[2,2],limits=divᵛhLims,colormap=:bwr, height=Relative(0.7))
    # Label(gl[2,1,Bottom()], L"-\mathrm{div}^v\, \mathsfbf{h}", fontsize=24)
    # Label(gl[2,1,Bottom()], L"-\mathrm{div}^v\, \mathbf{h}", fontsize=24)
    Label(gl[2,1,Bottom()], L"-\mathrm{div}^v", fontsize=24)

    push!(axes, Axis(gl[2,3], aspect=DataAspect()))
    for k=1:K
        poly!(axes[end],linkTriangles[k],color=cocurlᵛh[k],colorrange=cocurlᵛhLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
    end
    for i=1:I
        poly!(axes[end],cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
    end
    Colorbar(gl[2,4],limits=cocurlᵛhLims,colormap=:bwr, height=Relative(0.7))
    # Label(gl[2,3,Bottom()], L"\mathrm{cocurl}^v\, \mathsfbf{h}", fontsize=24)
    # Label(gl[2,3,Bottom()], L"\mathrm{cocurl}^v\, \mathbf{h}", fontsize=24)
    Label(gl[2,3,Bottom()], L"\mathrm{cocurl}^v", fontsize=24)

    push!(axes, Axis(gl[3,1], aspect=DataAspect()))
    for i=1:I
        poly!(axes[end],cellPolygons[i],color=curlᶜh[i],colorrange=curlᶜhLims,colormap=:bwr,strokewidth=1,strokecolor=(:black,0.25))
    end
    Colorbar(gl[3,2], colorrange=curlᶜhLims, colormap=:bwr, height=Relative(0.7))
    # Label(gl[3,1,Bottom()], L"\mathrm{curl}^c\, \mathsfbf{h}",fontsize=24)
    # Label(gl[3,1,Bottom()], L"\mathrm{curl}^c\, \mathbf{h}",fontsize=24)
    Label(gl[3,1,Bottom()], L"\mathrm{curl}^c",fontsize=24)

    push!(axes, Axis(gl[3,3], aspect=DataAspect()))
    for i=1:I
        poly!(axes[end],cellPolygons[i],color=codivᶜh[i],colorrange=codivᶜhLims,colormap=:bwr,strokewidth=1,strokecolor=(:black,0.25))
    end
    Colorbar(gl[3,4], colorrange=codivᶜhLims, colormap=:bwr, height=Relative(0.7))
    # Label(gl[3,3,Bottom()], L"\mathrm{codiv}^c\, \mathsfbf{h}",fontsize=24)
    # Label(gl[3,3,Bottom()], L"\mathrm{codiv}^c\, \mathbf{h}",fontsize=24)
    Label(gl[3,3,Bottom()], L"\mathrm{codiv}^c",fontsize=24)

    push!(axes, Axis(gl[4,1], aspect=DataAspect()))
    for k=1:K
        poly!(axes[end],linkTriangles[k],color=codivᵛh[k],colorrange=codivᵛhLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
    end
    for i=1:I
        poly!(axes[end],cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
    end
    Colorbar(gl[4,2],limits=codivᵛhLims,colormap=:bwr, height=Relative(0.7))
    # Label(gl[4,1,Bottom()], L"\mathrm{codiv}^v\, \mathsfbf{h}", fontsize=24)
    # Label(gl[4,1,Bottom()], L"\mathrm{codiv}^v\, \mathbf{h}", fontsize=24)
    Label(gl[4,1,Bottom()], L"\mathrm{codiv}^v", fontsize=24)

    push!(axes, Axis(gl[4,3], aspect=DataAspect()))
    for k=1:K
        poly!(axes[end],linkTriangles[k],color=curlᵛh[k],colorrange=curlᵛhLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
    end
    for i=1:I
        poly!(axes[end],cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
    end
    Colorbar(gl[4,4],limits=curlᵛhLims,colormap=:bwr, height=Relative(0.7))
    # Label(gl[4,3,Bottom()], L"\mathrm{curl}^v\, \mathrm{h}", fontsize=24)
    # Label(gl[4,3,Bottom()], L"\mathrm{curl}^v\, \mathbf{h}", fontsize=24)
    Label(gl[4,3,Bottom()], L"\mathrm{curl}^v", fontsize=24)

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
rowsize!(fig.layout, 1, Relative(0.98))
rowsize!(fig.layout, 2, Relative(0.02))

push!(axes, Axis(fig[:,3]))
vlines!(axes[end], [0.0], color=(:black,0.3), linewidth=5, linestyle=:solid)
# push!(axes, Axis(fig[:,1+4]))
# vlines!(axes[end], [0.0], color=(:black,0.3), linewidth=5, linestyle=:solid)

colsize!(fig.layout, 1, Relative(0.1))
colsize!(fig.layout, 2, Relative(0.44))
colsize!(fig.layout, 3, Relative(0.02))
colsize!(fig.layout, 4, Relative(0.44))
# colsize!(fig.layout, 1+4, Relative(0.05))
resize_to_layout!(fig)

hidedecorations!.(axes)
hidespines!.(axes)
display(fig)
save(datadir("figureDerivatives.png"), fig)