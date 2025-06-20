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

@from "$(srcdir("SingularValueDecomposition.jl"))" using SingularValueDecomposition

inputSystems = ["SingleHole", "DoubleHole"]#, "Voronoi", "OldSystem"]
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
Label(gl0[1:2,1], "", fontsize=24, rotation=π/2)
Label(gl0[3:4,1], "", fontsize=24, rotation=π/2)
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
    curlᶜh = curlᶜ(R, A, B, 𝐡)    
    curlᵛh = curlᵛ(R, A, B, 𝐡)    
    divᶜh = divᶜ(R, A, B, 𝐡)    
    divᵛh = divᵛ(R, A, B, 𝐡)    
    cocurlᶜh = cocurlᶜ(R, A, B, 𝐡)    
    cocurlᵛh = cocurlᵛ(R, A, B, 𝐡)    
    codᶜh = codᶜ(R, A, B, 𝐡)    
    codᵛh = codᵛ(R, A, B, 𝐡)    
    H = Diagonal(cellAreas)
    E = Diagonal(linkTriangleAreas)
    Lv = geometricLv(R, A, B)
    Lf = geometricLf(R, A, B)
    Lc = geometricLc(R, A, B)
    Lt = geometricLt(R, A, B)
    # ϕpar Lv -divᵛ
    ϕpar, ϕparspectrum = singularValueDecomposition(Lv, -1.0.*divᵛh, E)
    ϕparLims = (-maximum(abs.(ϕpar)), maximum(abs.(ϕpar)))
    # ϕperp Lv -codᵛ
    ϕperp, ϕperpspectrum = singularValueDecomposition(Lv, -1.0.*codᵛh, E)
    ϕperpLims = (-maximum(abs.(ϕperp)), maximum(abs.(ϕperp)))
    # upar Lf cocurlᶜ
    upar, uparspectrum = singularValueDecomposition(Lf, cocurlᶜh, H)
    uparLims = (-maximum(abs.(upar)), maximum(abs.(upar)))
    # uperp Lf curlᶜ
    uperp, uperpspectrum = singularValueDecomposition(Lf, curlᶜh, H)
    uperpLims = (-maximum(abs.(uperp)), maximum(abs.(uperp)))
    # ϕCapitalpar Lc -divᶜ
    ϕCapitalpar, ϕCapitalparspectrum = singularValueDecomposition(Lc, -1.0.*divᶜh, H)
    ϕCapitalparLims = (-maximum(abs.(ϕCapitalpar)), maximum(abs.(ϕCapitalpar)))
    # ϕCapitalperp Lc -codᶜ
    ϕCapitalperp, ϕCapitalperpspectrum = singularValueDecomposition(Lc, -1.0.*codᶜh, H)
    ϕCapitalperpLims = (-maximum(abs.(ϕCapitalperp)), maximum(abs.(ϕCapitalperp)))
    # Upar Lt cocurlᵛ
    Upar, Uparspectrum = singularValueDecomposition(Lt, cocurlᵛh, E)
    UparLims = (-maximum(abs.(Upar)), maximum(abs.(Upar)))
    # Uperp Lt curlᵛ
    Uperp, Uperpspectrum = singularValueDecomposition(Lt, curlᵛh, E)
    UperpLims = (-maximum(abs.(Uperp)), maximum(abs.(Uperp)))

    #%%
    gl = GridLayout(fig[1,1+col*2-1])

    push!(axes, Axis(gl[1,1], aspect=DataAspect()))
    for i=1:nCells
        poly!(axes[end],cellPolygons[i],color=upar[i],colorrange=uparLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
    end
    Colorbar(gl[1,2],limits=uparLims,colormap=:bwr)
    Label(gl[1,1,Bottom()], L"u^\parallel", fontsize = 24)

    push!(axes, Axis(gl[2,1], aspect=DataAspect()))
    for k=1:nVerts
        poly!(axes[end],linkTriangles[k],color=Upar[k],colorrange=UparLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
    end
    for i=1:nCells
        poly!(axes[end],cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
    end
    Colorbar(gl[2,2],limits=UparLims,colormap=:bwr)
    Label(gl[2,1,Bottom()], L"U^\parallel", fontsize = 24)

    push!(axes, Axis(gl[3,1], aspect=DataAspect()))
    for i=1:nCells
        poly!(axes[end],cellPolygons[i],color=uperp[i],colorrange=uperpLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
    end
    Colorbar(gl[3,2],limits=uperpLims,colormap=:bwr)
    Label(gl[3,1,Bottom()], L"u^\perp", fontsize = 24)

    push!(axes, Axis(gl[4,1], aspect=DataAspect()))
    for k=1:nVerts
        poly!(axes[end],linkTriangles[k],color=Uperp[k],colorrange=UperpLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
    end
    for i=1:nCells
        poly!(axes[end],cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
    end
    Colorbar(gl[4,2],limits=UperpLims,colormap=:bwr)
    Label(gl[4,1,Bottom()], L"U^\perp", fontsize = 24)



    push!(axes, Axis(gl[1,3], aspect=DataAspect()))
    for i=1:nCells
        poly!(axes[end],cellPolygons[i],color=ϕCapitalpar[i],colorrange=ϕCapitalparLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
    end
    Colorbar(gl[1,4],limits=ϕCapitalparLims,colormap=:bwr)
    Label(gl[1,3,Bottom()], L"\Phi^\parallel", fontsize = 24)

    push!(axes, Axis(gl[2,3], aspect=DataAspect()))
    for k=1:nVerts
        poly!(axes[end],linkTriangles[k],color=ϕpar[k],colorrange=ϕparLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
    end
    for i=1:nCells
        poly!(axes[end],cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
    end
    Colorbar(gl[2,4],limits=ϕparLims,colormap=:bwr)
    Label(gl[2,3,Bottom()], L"\phi^\parallel", fontsize = 24)

    push!(axes, Axis(gl[3,3], aspect=DataAspect()))
    for i=1:nCells
        poly!(axes[end],cellPolygons[i],color=-ϕCapitalperp[i],colorrange=ϕCapitalperpLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
    end
    Colorbar(gl[3,4],limits=ϕCapitalperpLims,colormap=:bwr)
    Label(gl[3,3,Bottom()], L"-\Phi^\perp", fontsize = 24)

    push!(axes, Axis(gl[4,3], aspect=DataAspect()))
    for k=1:nVerts
        poly!(axes[end],linkTriangles[k],color=-ϕperp[k],colorrange=ϕperpLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
    end
    for i=1:nCells
        poly!(axes[end],cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
    end
    Colorbar(gl[4,4],limits=ϕperpLims,colormap=:bwr)
    Label(gl[4,3,Bottom()], L"-\phi^\perp", fontsize = 24)
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

hidedecorations!.(axes)
hidespines!.(axes)
display(fig)
# save(datadir("figurePotentials.png"), fig)