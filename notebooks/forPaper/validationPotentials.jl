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

@from "$(srcdir("PenrosePseudoInversion.jl"))" using PenrosePseudoInversion

inputDir = "quadraticPotentialNoPressure"; isdir(plotsdir(inputDir)) ? nothing : mkdir(plotsdir(inputDir)); isdir(plotsdir(inputDir)) ? nothing : mkdir(plotsdir(inputDir))
inputSystems = ["NoHole", "SingleHole", "DoubleHole"]#, "Voronoi", "OldSystem"]

fig = Figure(size=(2500, 1500), fontsize=24)
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"($l)" for l in individualLetters]

gl0 = GridLayout(fig[1,1])
Label(gl0[0,1], "", fontsize=48, rotation=π/2)
Label(gl0[1,1], "Cells", fontsize=48, rotation=π/2)
Label(gl0[2,1], "Vertices", fontsize=48, rotation=π/2)
Label(gl0[3,1], "Cells", fontsize=48, rotation=π/2)
Label(gl0[4,1], "Vertices", fontsize=48, rotation=π/2)
# Label(gl0[1:2,1], "Divergences", fontsize=48, rotation=π/2)
# Label(gl0[3:4,1], "Curls", fontsize=48, rotation=π/2)
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
        fileName = datadir("referenceSystems", inputDir, "$(inputSystem)_testSystem.jld2")
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
    iⁱ = ones(Int64, I).-findPeripheralCells(B)
    cellAreas = findCellAreas(R, A, B)
    linkTriangles = findCellLinkTriangles(R, A, B)
    linkTriangleAreas = findCellLinkTriangleAreas(R, A, B)
    cellPolygons = findCellPolygons(R, A, B)
    𝐡 = hNetwork(R, A, B, F)
    # 𝐡 = [SVector{2,Float64}(1.0,0.0) for _=1:J]

    curlᶜh = curlᶜ(R, A, B, 𝐡)   
    curlᵛh = curlᵛspokes(R, A, B, 𝐡)   
    divᶜh = divᶜ(R, A, B, 𝐡)
    divᵛh = divᵛsuppress(R, A, B, 𝐡)
    cocurlᶜh = cocurlᶜ(R, A, B, 𝐡)   
    cocurlᵛh = cocurlᵛspokes(R, A, B, 𝐡)  
    codivᶜh = codivᶜ(R, A, B, 𝐡)
    codivᵛh = codivᵛsuppress(R, A, B, 𝐡)
    
    L̂v, Lvreindexing = geometricLvHatReduced(R, A, B)
    L̂f, Lfreindexing = geometricLfHatReduced(R, A, B)
    Lf = geometricLf(R, A, B)
    L̂c, Lcreindexing = geometricLcHatReduced(R, A, B)
    Lc = geometricLc(R, A, B)
    L̂t, Ltreindexing = geometricLtHatReduced(R, A, B)
    H = Diagonal(cellAreas[Lcreindexing])
    E = Diagonal(linkTriangleAreas[Lvreindexing])

    𝟙ᶜ = ones(I)
    𝟙ᵛ = ones(K)

    # ϕpar Lv -divᵛ
    ϕpar = L̂v\(-1.0.*divᵛh[Lvreindexing])
    # ϕperp Lv -codivᵛ
    ϕperp = L̂v\(-1.0.*codivᵛh[Lvreindexing])
    # upar Lf cocurlᶜ
    upar = penrosePseudoInversion(L̂f, cocurlᶜh[Lfreindexing], H)
    uparCorrection = penrosePseudoInversion(Lf, ones(I), H)
    upar .+= uparCorrection .* innerProd(𝟙ᶜ, H, cocurlᶜh) / innerProd(𝟙ᶜ, H, 𝟙ᶜ)
    # uperp Lf curlᶜ
    uperp = penrosePseudoInversion(L̂f, curlᶜh[Lfreindexing], H)
    # uperpCorrection = penrosePseudoInversion(Lf, ones(I), H)
    # uperp .+= uperpCorrection .* innerProd(𝟙ᶜ, H, curlᶜh) / innerProd(𝟙ᶜ, H, 𝟙ᶜ)
    # ϕCapitalpar Lc -divᶜ
    ϕCapitalpar = penrosePseudoInversion(L̂c, -1.0.*divᶜh[Lcreindexing], H)
    ϕCapitalparCorrection = Lc\𝟙ᶜ
    ϕCapitalpar .+= ϕCapitalparCorrection .* innerProd(𝟙ᶜ, H, -1.0.*divᶜh) / innerProd(𝟙ᶜ, H, 𝟙ᶜ)
    # ϕCapitalperp Lc -codivᶜ
    ϕCapitalperp = penrosePseudoInversion(L̂c, -1.0.*codivᶜh[Lcreindexing], H)
    ϕCapitalperpCorrection = Lc\𝟙ᶜ
    ϕCapitalperp .+= ϕCapitalperpCorrection .* innerProd(𝟙ᶜ, H, -1.0.*codivᶜh) / innerProd(𝟙ᶜ, H, 𝟙ᶜ)
    # Upar Lt cocurlᵛ
    Upar = L̂t\(cocurlᵛh[Ltreindexing])
    # Uperp Lt curlᵛ
    Uperp = L̂t\(curlᵛh[Ltreindexing])
    
    divᵛhDif = L̂v*ϕpar .+ divᵛh[Lvreindexing]
    divᵛhDifMax = max(maximum(abs.(divᵛhDif)), 0.00001)
    @show maximum(abs.(divᵛhDif))
    divᵛhDifLims = (-divᵛhDifMax, divᵛhDifMax)
    
    codivᵛhDif = (L̂v*ϕperp) .+ codivᵛh[Lvreindexing]
    codivᵛhDifMax = max(maximum(abs.(codivᵛhDif)), 0.00001)
    @show maximum(abs.(codivᵛhDif))
    codivᵛhDifLims = (-codivᵛhDifMax, codivᵛhDifMax)

    cocurlᶜhDif = L̂f*upar .- cocurlᶜh
    cocurlᶜhDifMax = max(maximum(abs.(cocurlᶜhDif)), 0.00001)
    @show maximum(abs.(cocurlᶜhDif))
    cocurlᶜhDifLims = (-cocurlᶜhDifMax, cocurlᶜhDifMax)

    curlᶜhDif = L̂f*uperp .- curlᶜh
    curlᶜhDifMax = max(maximum(abs.(curlᶜhDif)), 0.00001)
    @show maximum(abs.(curlᶜhDif))
    curlᶜhDifLims = (-curlᶜhDifMax, curlᶜhDifMax)

    divᶜhDif = L̂c*ϕCapitalpar .+ divᶜh
    divᶜhDifMax = max(maximum(abs.(divᶜhDif)), 0.00001)
    @show maximum(abs.(divᶜhDif[iⁱ.==1]))
    divᶜhDifLims = (-divᶜhDifMax, divᶜhDifMax)

    codivᶜhDif = L̂c*ϕCapitalperp .+ codivᶜh
    codivᶜhDifMax = max(maximum(abs.(codivᶜhDif)), 0.00001)
    @show maximum(abs.(codivᶜhDif[iⁱ.==1]))
    codivᶜhDifLims = (-codivᶜhDifMax, codivᶜhDifMax)

    cocurlᵛhDif = L̂t*Upar .- cocurlᵛh[Ltreindexing]
    cocurlᵛhDifMax = max(maximum(abs.(cocurlᵛhDif)), 0.00001)
    @show maximum(abs.(cocurlᵛhDif))
    cocurlᵛhDifLims = (-cocurlᵛhDifMax, cocurlᵛhDifMax)
    
    Uperp = L̂t\(curlᵛh[Ltreindexing])
    curlᵛhDif = L̂t*Uperp .- curlᵛh[Ltreindexing]
    curlᵛhDifMax = max(maximum(abs.(curlᵛhDif)), 0.00001)
    @show maximum(abs.(curlᵛhDif))
    curlᵛhDifLims = (-curlᵛhDifMax, curlᵛhDifMax)

    #%%

    gl = GridLayout(fig[1,1+col*2-1])
    
    push!(axes, Axis(gl[1,1], aspect=DataAspect()))
    for i=1:I
        poly!(axes[end],cellPolygons[i],color=cocurlᶜhDif[i],colorrange=cocurlᶜhDifLims,colormap=:bam,strokewidth=1,strokecolor=(:black,0.25))
    end
    Label(gl[1,1,Bottom()], L"\Delta\mathrm{cocurl}^c",fontsize=36)
    Colorbar(gl[1,2], colorrange=cocurlᶜhDifLims, colormap=:bam, height=Relative(0.8))

    push!(axes, Axis(gl[1,3], aspect=DataAspect()))
    for i=1:I
        poly!(axes[end],cellPolygons[i],color=divᶜhDif[i],colorrange=divᶜhDifLims,colormap=:bam,strokewidth=1,strokecolor=(:black,0.25))
    end
    Label(gl[1,3,Bottom()], L"\Delta\mathrm{div}^c",fontsize=36)
    Colorbar(gl[1,4], colorrange=divᶜhDifLims, colormap=:bam, height=Relative(0.8))

    push!(axes, Axis(gl[2,1], aspect=DataAspect()))
    for (k, kk) in enumerate(collect(1:K)[Lvreindexing])
        @show 
        poly!(axes[end],linkTriangles[kk],color=divᵛhDif[k],colorrange=divᵛhDifLims,colormap=:bam,strokewidth=1,strokecolor=(:white,0.0))
    end
    for i=1:I
        poly!(axes[end],cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
    end
    Label(gl[2,1,Bottom()], L"\Delta\mathrm{div}^v", fontsize=36)
    Colorbar(gl[2,2],colorrange=divᵛhDifLims,colormap=:bam, height=Relative(0.8))
    
    push!(axes, Axis(gl[2,3], aspect=DataAspect()))
    for (k, kk) in enumerate(collect(1:K)[Ltreindexing])
        poly!(axes[end],linkTriangles[kk],color=cocurlᵛhDif[k],colorrange=cocurlᵛhDifLims,colormap=:bam,strokewidth=1,strokecolor=(:white,0.0))
    end
    for i=1:I
        poly!(axes[end],cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
    end
    Label(gl[2,3,Bottom()], L"\Delta\mathrm{cocurl}^v", fontsize=36)
    Colorbar(gl[2,4],colorrange=cocurlᵛhDifLims,colormap=:bam, height=Relative(0.8))

    push!(axes, Axis(gl[3,1], aspect=DataAspect()))
    for i=1:I
        poly!(axes[end],cellPolygons[i],color=curlᶜhDif[i],colorrange=curlᶜhDifLims,colormap=:bam,strokewidth=1,strokecolor=(:black,0.25))
    end
    Label(gl[3,1,Bottom()], L"\Delta\mathrm{curl}^c",fontsize=36)
    Colorbar(gl[3,2], colorrange=curlᶜhDifLims, colormap=:bam, height=Relative(0.8))
    
    push!(axes, Axis(gl[3,3], aspect=DataAspect()))
    for i=1:I
        poly!(axes[end],cellPolygons[i],color=codivᶜhDif[i],colorrange=codivᶜhDifLims,colormap=:bam,strokewidth=1,strokecolor=(:black,0.25))
    end
    Label(gl[3,3,Bottom()], L"\Delta\mathrm{codiv}^c",fontsize=36)
    Colorbar(gl[3,4], colorrange=codivᶜhDifLims, colormap=:bam, height=Relative(0.8))

    push!(axes, Axis(gl[4,1], aspect=DataAspect()))
    for (k, kk) in enumerate(collect(1:K)[Lvreindexing])
        poly!(axes[end],linkTriangles[kk],color=codivᵛhDif[k],colorrange=codivᵛhDifLims,colormap=:bam,strokewidth=1,strokecolor=(:white,0.0))
    end
    for i=1:I
        poly!(axes[end],cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
    end
    Label(gl[4,1,Bottom()], L"\Delta\mathrm{codiv}^v", fontsize=36)
    Colorbar(gl[4,2],colorrange=codivᵛhDifLims,colormap=:bam, height=Relative(0.8))
    
    push!(axes, Axis(gl[4,3], aspect=DataAspect()))
    for (k, kk) in enumerate(collect(1:K)[Ltreindexing])
        poly!(axes[end],linkTriangles[kk],color=curlᵛhDif[k],colorrange=curlᵛhDifLims,colormap=:bam,strokewidth=1,strokecolor=(:white,0.0))
    end
    for i=1:I
        poly!(axes[end],cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
    end
    Label(gl[4,3,Bottom()], L"\Delta\mathrm{curl}^v", fontsize=36)
    Colorbar(gl[4,4],colorrange=curlᵛhDifLims,colormap=:bam, height=Relative(0.8))

    Label(gl[0,1], "Primal", fontsize=48)
    Label(gl[0,3], "Dual", fontsize=48)
    
    rowsize!(gl, 0, Relative(0.04))
    rowsize!(gl, 1, Relative(0.24))
    rowsize!(gl, 2, Relative(0.24))
    rowsize!(gl, 3, Relative(0.24))
    rowsize!(gl, 4, Relative(0.24))

    colsize!(gl, 1, Relative(0.49))
    colsize!(gl, 2, Relative(0.01))
    colsize!(gl, 3, Relative(0.49))
    colsize!(gl, 4, Relative(0.01))

end

Label(fig[2,2, Top()], L"(a)", fontsize=48)
Label(fig[2,4, Top()], L"(b)", fontsize=48)
Label(fig[2,6, Top()], L"(c)", fontsize=48)
rowsize!(fig.layout, 1, Relative(0.98))
rowsize!(fig.layout, 2, Relative(0.02))

push!(axes, Axis(fig[:,3]))
vlines!(axes[end], [0.0], color=(:black,0.3), linewidth=5, linestyle=:solid)
push!(axes, Axis(fig[:,5]))
vlines!(axes[end], [0.0], color=(:black,0.3), linewidth=5, linestyle=:solid)

colsize!(fig.layout, 1, Relative(0.05))
colsize!(fig.layout, 2, Relative(0.3))
colsize!(fig.layout, 3, Relative(0.01))
colsize!(fig.layout, 4, Relative(0.3))
colsize!(fig.layout, 5, Relative(0.01))
colsize!(fig.layout, 6, Relative(0.3))
resize_to_layout!(fig)

hidedecorations!.(axes)
hidespines!.(axes)
display(fig)
save(plotsdir(inputDir, "validationPotentials.png"), fig)
save(plotsdir(inputDir, "validationPotentials.pdf"), fig)