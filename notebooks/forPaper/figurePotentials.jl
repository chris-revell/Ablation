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
inputSystems = ["NoHole", "SingleHole", "DoubleHole"]

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
rowsize!(gl0, 0, Relative(0.04))
rowsize!(gl0, 1, Relative(0.24))
rowsize!(gl0, 2, Relative(0.24))
rowsize!(gl0, 3, Relative(0.24))
rowsize!(gl0, 4, Relative(0.24))


for (col,inputSystem) in enumerate(inputSystems)
    
    # Import system data
    fileName = datadir("referenceSystems", inputDir, "$(inputSystem)_testSystem.jld2")
    importedData = load(fileName)
    R = importedData["R"]
    A = importedData["A"]
    B = importedData["B"]
    F = importedData["F"]

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

    
    ϕparMax = max(maximum(abs.(ϕpar)), 0.0001)
    ϕparLims = (-ϕparMax, ϕparMax)
    ϕperpMax = max(maximum(abs.(ϕperp)), 0.0001)
    ϕperpLims = (-ϕperpMax, ϕperpMax)
    uparMax = max(maximum(abs.(upar)), 0.0001)
    uparLims = (-uparMax, uparMax)
    uperpMax = max(maximum(abs.(uperp)), 0.0001)
    uperpLims = (-uperpMax, uperpMax)
    ϕCapitalparMax = max(maximum(abs.(ϕCapitalpar)), 0.0001)
    ϕCapitalparLims = (-ϕCapitalparMax, ϕCapitalparMax)
    ϕCapitalperpMax = max(maximum(abs.(ϕCapitalperp)), 0.0001)
    ϕCapitalperpLims = (-ϕCapitalperpMax, ϕCapitalperpMax)
    UparMax = max(maximum(abs.(Upar)), 0.0001)
    UparLims = (-UparMax, UparMax)
    UperpMax = max(maximum(abs.(Uperp)), 0.0001)
    UperpLims = (-UperpMax, UperpMax)

    row1Max = maximum([ϕCapitalparMax,uparMax])
    row1Lims = (-row1Max, row1Max)
    row2Max = maximum([UparMax, ϕparMax])
    row2Lims = (-row2Max, row2Max)
    row3Max = max(uperpMax, ϕCapitalperpMax)
    row3Lims = (-row3Max, row3Max)
    row4Max = max(ϕperpMax, UperpMax)
    row4Lims = (-row4Max, row4Max)

    #%%
    gl = GridLayout(fig[1,1+col*2-1])

    push!(axes, Axis(gl[1,1], aspect=DataAspect()))
    for (i, ii) in enumerate(Lfreindexing)
        poly!(axes[end],cellPolygons[ii],color=upar[i],colorrange=uparLims,colormap=:bam,strokewidth=1,strokecolor=(:black,0.25))
    end
    Label(gl[1,1,Bottom()], L"-u^\parallel", fontsize = 36)
    Colorbar(gl[1,2],limits=uparLims,colormap=:bam, height=Relative(0.7))#, ticks=(uparLims[1]:uparLims[2]:uparLims[2]))
    push!(axes, Axis(gl[1,3], aspect=DataAspect()))
    for (i, ii) in enumerate(Lcreindexing)
        poly!(axes[end],cellPolygons[ii],color=ϕCapitalpar[i],colorrange=ϕCapitalparLims,colormap=:bam,strokewidth=1,strokecolor=(:black,0.25))
    end
    Label(gl[1,3,Bottom()], L"\Phi^\parallel", fontsize = 36)
    Colorbar(gl[1,4],limits=ϕCapitalparLims,colormap=:bam, height=Relative(0.7))#, ticks=(ϕCapitalparLims[1]:ϕCapitalparLims[2]:ϕCapitalparLims[2]))

    push!(axes, Axis(gl[2,1], aspect=DataAspect()))
    for (k, kk) in enumerate(Lvreindexing)
        poly!(axes[end],linkTriangles[kk],color=ϕpar[k],colorrange=ϕparLims,colormap=:bam,strokewidth=1,strokecolor=(:white,0.0))
    end
    for i=1:I
        poly!(axes[end],cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
    end
    Label(gl[2,1,Bottom()], L"\phi^\parallel", fontsize = 36)
    Colorbar(gl[2,2],limits=ϕparLims,colormap=:bam, height=Relative(0.7))#, ticks=(ϕparLims[1]:ϕparLims[2]:ϕparLims[2]))
    push!(axes, Axis(gl[2,3], aspect=DataAspect()))
    for (k, kk) in enumerate(Ltreindexing)
        poly!(axes[end],linkTriangles[kk],color=Upar[k],colorrange=UparLims,colormap=:bam,strokewidth=1,strokecolor=(:white,0.0))
    end
    for i=1:I
        poly!(axes[end],cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
    end
    Label(gl[2,3,Bottom()], L"-U^\parallel", fontsize = 36)
    Colorbar(gl[2,4],limits=UparLims,colormap=:bam, height=Relative(0.7))#, ticks=(UparLims[1]:UparLims[2]:UparLims[2]))

    push!(axes, Axis(gl[3,1], aspect=DataAspect()))
    for (i, ii) in enumerate(Lfreindexing)
        poly!(axes[end],cellPolygons[ii],color=uperp[i],colorrange=uperpLims,colormap=:bam,strokewidth=1,strokecolor=(:black,0.25))
    end
    Label(gl[3,1,Bottom()], L"-u^\perp", fontsize = 36)
    Colorbar(gl[3,2],limits=uperpLims,colormap=:bam, height=Relative(0.7))#, ticks=(uperpLims[1]:uperpLims[2]:uperpLims[2]))
    push!(axes, Axis(gl[3,3], aspect=DataAspect()))
    for (i, ii) in enumerate(Lcreindexing)
        poly!(axes[end],cellPolygons[ii],color=-ϕCapitalperp[i],colorrange=ϕCapitalperpLims,colormap=:bam,strokewidth=1,strokecolor=(:black,0.25))
    end
    Label(gl[3,3,Bottom()], L"-\Phi^\perp", fontsize = 36)
    Colorbar(gl[3,4],limits=ϕCapitalperpLims,colormap=:bam, height=Relative(0.7))#, ticks=(ϕCapitalperpLims[1]:ϕCapitalperpLims[2]:ϕCapitalperpLims[2]))

    push!(axes, Axis(gl[4,1], aspect=DataAspect()))
    for (k, kk) in enumerate(Lvreindexing)
        poly!(axes[end],linkTriangles[kk],color=-ϕperp[k],colorrange=ϕperpLims, colormap=:bam,strokewidth=1,strokecolor=(:white,0.0))
    end
    for i=1:I
        poly!(axes[end],cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
    end
    Label(gl[4,1,Bottom()], L"-\phi^\perp", fontsize = 36)
    Colorbar(gl[4,2],limits=ϕperpLims, colormap=:bam, height=Relative(0.7))#, ticks=(ϕperpLims[1]:ϕperpLims[2]:ϕperpLims[2]))
    push!(axes, Axis(gl[4,3], aspect=DataAspect()))
    for (k, kk) in enumerate(Ltreindexing)
        poly!(axes[end],linkTriangles[kk],color=Uperp[k],colorrange=UperpLims, colormap=:bam,strokewidth=1,strokecolor=(:white,0.0))
    end
    for i=1:I
        poly!(axes[end],cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
    end
    Label(gl[4,3,Bottom()], L"-U^\perp", fontsize = 36)
    Colorbar(gl[4,4],limits=UperpLims, colormap=:bam, height=Relative(0.7))#, ticks=(UperpLims[1]:UperpLims[2]:UperpLims[2]))
    
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
save(plotsdir(inputDir, "midpointsfigurePotentials.png"), fig)
save(plotsdir(inputDir, "midpointsfigurePotentials.pdf"), fig)