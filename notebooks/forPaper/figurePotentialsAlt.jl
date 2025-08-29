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
using Printf

@from "$(srcdir("PenrosePseudoInversion.jl"))" using PenrosePseudoInversion

inputDir = "quadraticPotentialNoPressure"; isdir(plotsdir(inputDir)) ? nothing : mkdir(plotsdir(inputDir)); isdir(plotsdir(inputDir)) ? nothing : mkdir(plotsdir(inputDir))
# inputDir = "quadraticPotentialWithPressure"; isdir(plotsdir(inputDir)) ? nothing : mkdir(plotsdir(inputDir)); isdir(plotsdir(inputDir)) ? nothing : mkdir(plotsdir(inputDir))
inputSystems = ["NoHole", "SingleHole", "DoubleHole"]#, "Voronoi", "OldSystem"]

fig = Figure(size=(2500, 1000), fontsize=48)
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"($l)" for l in individualLetters]

gl0 = GridLayout(fig[1,1])
Label(gl0[0,2], "", fontsize=48, rotation=π/2)
# Label(gl0[1,2], "Divergences", fontsize=48, rotation=π/2)
# Label(gl0[2,2], "Curls", fontsize=48, rotation=π/2)
# Label(gl0[3,2], "Cells", fontsize=48, rotation=π/2)
# Label(gl0[4,2], "Vertices", fontsize=48, rotation=π/2)
# Label(gl0[1:2,1], "Dual", fontsize=48, rotation=π/2)
# Label(gl0[3:4,1], "Curls", fontsize=48, rotation=π/2)
# rowsize!(gl0, 0, Relative(0.04))
# rowsize!(gl0, 1, Relative(0.24))
# rowsize!(gl0, 2, Relative(0.24))
# rowsize!(gl0, 3, Relative(0.24))
# rowsize!(gl0, 4, Relative(0.24))


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
    cellAreas = findCellAreas(R, A, B)
    linkTriangles = findCellLinkTriangles(R, A, B)
    linkTriangleAreas = findCellLinkTriangleAreas(R, A, B)
    cellPolygons = findCellPolygons(R, A, B)
    𝐡 = hNetwork(R, A, B, F)
    # 𝐡 = [SVector{2,Float64}(1.0,0.0) for _=1:J]
    
    boundaryVertices = findBoundaryVertices(A, B).==1
    boundaryCells = findBoundaryCells(B).==1

    curlᶜh = curlᶜ(R, A, B, 𝐡)
    curlᶜhMax = max(maximum(abs.(curlᶜh)), 0.0001)
    # curlᵛh = (spokesOrNot=="spokes" ? curlᵛspokes(R, A, B, 𝐡) : curlᵛ(R, A, B, 𝐡))
    curlᵛh = curlᵛspokes(R, A, B, 𝐡)
    curlᵛhMax = max(maximum(abs.(curlᵛh)), 0.0001)
    # divᶜh = (suppressOrNot=="suppress" ? divᶜsuppress(R, A, B, 𝐡) : divᶜ(R, A, B, 𝐡))
    divᶜh = divᶜ(R, A, B, 𝐡)
    divᶜhMax = max(maximum(abs.(divᶜh)), 0.0001)
    divᵛh = divᵛsuppress(R, A, B, 𝐡)
    divᵛhMax = max(maximum(abs.(divᵛh)), 0.0001)
    cocurlᶜh = cocurlᶜ(R, A, B, 𝐡)
    cocurlᶜhMax = max(maximum(abs.(cocurlᶜh)), 0.0001)
    # cocurlᵛh = (spokesOrNot=="spokes" ? cocurlᵛspokes(R, A, B, 𝐡) : cocurlᵛ(R, A, B, 𝐡))
    cocurlᵛh = cocurlᵛspokes(R, A, B, 𝐡)
    cocurlᵛhMax = max(maximum(abs.(cocurlᵛh)), 0.0001)
    # codivᶜh = (suppressOrNot=="suppress" ? codivᶜsuppress(R, A, B, 𝐡) : codivᶜ(R, A, B, 𝐡))
    codivᶜh = codivᶜ(R, A, B, 𝐡)
    codivᶜhMax = max(maximum(abs.(codivᶜh)), 0.0001)
    codivᵛh = codivᵛsuppress(R, A, B, 𝐡)
    codivᵛhMax = max(maximum(abs.(codivᵛh)), 0.0001)
    
    H = Diagonal(cellAreas)
    E = Diagonal(linkTriangleAreas)
    Lv = geometricLv(R, A, B)
    Lf = geometricLf(R, A, B)
    Lc = geometricLc(R, A, B)
    Lt = geometricLt(R, A, B)
    # ϕpar Lv -divᵛ
    ϕpar, ϕparspectrum = penrosePseudoInversion(Lv, -1.0.*divᵛh, E)
    # @show maximum(abs.(Lv*ϕpar .+ divᵛh))
    ϕparMax = max(maximum(abs.(ϕpar)), 0.0001)
    ϕparLims = (-ϕparMax, ϕparMax)
    # ϕperp Lv -codivᵛ
    ϕperp, ϕperpspectrum = penrosePseudoInversion(Lv, -1.0.*codivᵛh, E)
    # @show maximum(abs.(Lv*ϕperp .+ codivᵛh))
    ϕperpMax = max(maximum(abs.(ϕperp)), 0.0001)
    ϕperpLims = (-ϕperpMax, ϕperpMax)
    # upar Lf cocurlᶜ
    upar, uparspectrum = penrosePseudoInversion(Lf, cocurlᶜh, H)
    # @show maximum(abs.(Lf*upar .- cocurlᶜh))
    uparMax = max(maximum(abs.(upar)), 0.0001)
    uparLims = (-uparMax, uparMax)
    # uperp Lf curlᶜ
    uperp, uperpspectrum = penrosePseudoInversion(Lf, curlᶜh, H)
    # @show maximum(abs.(Lf*uperp .- curlᶜh))
    uperpMax = max(maximum(abs.(uperp)), 0.0001)
    uperpLims = (-uperpMax, uperpMax)
    # ϕCapitalpar Lc -divᶜ
    ϕCapitalpar, ϕCapitalparspectrum = penrosePseudoInversion(Lc, -1.0.*divᶜh, H)
    # @show maximum(abs.(Lc*ϕCapitalpar .+ divᶜh))
    ϕCapitalparMax = max(maximum(abs.(ϕCapitalpar)), 0.0001)
    ϕCapitalparLims = (-ϕCapitalparMax, ϕCapitalparMax)
    # ϕCapitalperp Lc -codivᶜ
    ϕCapitalperp, ϕCapitalperpspectrum = penrosePseudoInversion(Lc, -1.0.*codivᶜh, H)
    # @show maximum(abs.(Lc*ϕCapitalperp .+ codivᶜh))
    ϕCapitalperpMax = max(maximum(abs.(ϕCapitalperp)), 0.0001)
    ϕCapitalperpLims = (-ϕCapitalperpMax, ϕCapitalperpMax)
    # Upar Lt cocurlᵛ
    Upar, Uparspectrum = penrosePseudoInversion(Lt, cocurlᵛh, E)
    # @show maximum(abs.(Lt*Upar .- cocurlᵛh))
    UparMax = max(maximum(abs.(Upar)), 0.0001)
    UparLims = (-UparMax, UparMax)
    # Uperp Lt curlᵛ
    Uperp, Uperpspectrum = penrosePseudoInversion(Lt, curlᵛh, E)
    # @show maximum(abs.(Lt*Uperp .- curlᵛh))
    UperpMax = max(maximum(abs.(Uperp)), 0.0001)
    UperpLims = (-UperpMax, UperpMax)

    row1Max = maximum([ϕCapitalparMax, UparMax])
    row1Lims = (-row1Max, row1Max)
    row2Max = maximum([ϕCapitalperpMax, UperpMax])
    row2Lims = (-row2Max, row2Max)
    
    
    # row3Max = max(uperpMax, ϕCapitalperpMax)
    # row3Lims = (-row3Max, row3Max)
    # row4Max = max(ϕperpMax, UperpMax)
    # row4Lims = (-row4Max, row4Max)

    #%%
    gl = GridLayout(fig[1,1+col*2-1])

    push!(axes, Axis(gl[1,1], aspect=DataAspect()))
    for i=1:I
        poly!(axes[end],cellPolygons[i],color=ϕCapitalpar[i],colorrange=row1Lims,colormap=:bam,strokewidth=1,strokecolor=(:black,0.25))
    end
    Label(gl[1,1,Bottom()], L"\Phi^\parallel", fontsize = 24)
    push!(axes, Axis(gl[1,2], aspect=DataAspect()))
    for k=1:K
        poly!(axes[end],linkTriangles[k],color=Upar[k],colorrange=row1Lims,colormap=:bam,strokewidth=1,strokecolor=(:white,0.0))
    end
    for i=1:I
        poly!(axes[end],cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
    end
    Label(gl[1,2,Bottom()], L"U^\parallel", fontsize = 36)
    Colorbar(gl[1,3],limits=row1Lims,colormap=:bam, height=Relative(0.8), ticklabelsize=24, ticks = ([-row1Max,0.0,row1Max], map(x -> @sprintf("%.3f",x), [-row1Max,0.0,row1Max])))

    
    push!(axes, Axis(gl[2,1], aspect=DataAspect()))
    for i=1:I
        poly!(axes[end],cellPolygons[i],color=-ϕCapitalperp[i],colorrange=row2Lims,colormap=:bam,strokewidth=1,strokecolor=(:black,0.25))
    end
    Label(gl[2,1,Bottom()], L"-\Phi^\perp", fontsize = 24)
    push!(axes, Axis(gl[2,2], aspect=DataAspect()))
    for k=1:K
        poly!(axes[end],linkTriangles[k],color=Uperp[k],colorrange=row2Lims, colormap=:bam,strokewidth=1,strokecolor=(:white,0.0))
    end
    for i=1:I
        poly!(axes[end],cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
    end
    Label(gl[2,2,Bottom()], L"U^\perp", fontsize = 36)
    Colorbar(gl[2,3],limits=row2Lims, colormap=:bam, height=Relative(0.8), ticklabelsize=24, ticks = ([-row2Max,0.0,row2Max], map(x -> @sprintf("%.3f",x), [-row2Max,0.0,row2Max])))


    # push!(axes, Axis(gl[4,1], aspect=DataAspect()))
    # for k=1:K
    #     poly!(axes[end],linkTriangles[k],color=-ϕperp[k],colorrange=row2Lims, colormap=:bam,strokewidth=1,strokecolor=(:white,0.0))
    # end
    # for i=1:I
    #     poly!(axes[end],cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
    # end
    # Label(gl[4,1,Bottom()], L"-\phi^\perp", fontsize = 24)


    # push!(axes, Axis(gl[3,1], aspect=DataAspect()))
    # for i=1:I
    #     poly!(axes[end],cellPolygons[i],color=uperp[i],colorrange=row2Lims,colormap=:bam,strokewidth=1,strokecolor=(:black,0.25))
    # end
    # Label(gl[3,1,Bottom()], L"u^\perp", fontsize = 24)

    # push!(axes, Axis(gl[2,1], aspect=DataAspect()))
    # for k=1:K
    #     poly!(axes[end],linkTriangles[k],color=ϕpar[k],colorrange=row1Lims,colormap=:bam,strokewidth=1,strokecolor=(:white,0.0))
    # end
    # for i=1:I
    #     poly!(axes[end],cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
    # end
    # Label(gl[2,1,Bottom()], L"\phi^\parallel", fontsize = 24)

    # push!(axes, Axis(gl[1,1], aspect=DataAspect()))
    # for i=1:I
    #     poly!(axes[end],cellPolygons[i],color=upar[i],colorrange=row1Lims,colormap=:bam,strokewidth=1,strokecolor=(:black,0.25))
    # end
    # Label(gl[1,1,Bottom()], L"u^\parallel", fontsize = 24)

    
    Label(gl[0,1], "Cells", fontsize=48)
    Label(gl[0,2], "Vertices", fontsize=48)
    
    rowsize!(gl, 0, Relative(0.05))
    # rowsize!(gl, 1, Relative(0.46))
    # rowsize!(gl, 2, Relative(0.46))
    # rowsize!(gl, 3, Relative(0.24))
    # rowsize!(gl, 4, Relative(0.24))

    colsize!(gl, 1, Relative(0.495))
    colsize!(gl, 2, Relative(0.495))
    colsize!(gl, 3, Relative(0.01))
    
end

Label(fig[2,2, Top()], L"(a)", fontsize=48)
Label(fig[2,4, Top()], L"(b)", fontsize=48)
Label(fig[2,6, Top()], L"(c)", fontsize=48)
# Label(fig[2,1+5, Top()], "(c)", fontsize=48)
rowsize!(fig.layout, 1, Relative(0.98))
rowsize!(fig.layout, 2, Relative(0.02))

push!(axes, Axis(fig[:,3]))
vlines!(axes[end], [0.0], color=(:black,0.3), linewidth=5, linestyle=:solid)
push!(axes, Axis(fig[:,5]))
vlines!(axes[end], [0.0], color=(:black,0.3), linewidth=5, linestyle=:solid)

colsize!(fig.layout, 1, Relative(0.1))
colsize!(fig.layout, 2, Relative(0.29))
colsize!(fig.layout, 3, Relative(0.01))
colsize!(fig.layout, 4, Relative(0.29))
colsize!(fig.layout, 5, Relative(0.01))
colsize!(fig.layout, 6, Relative(0.29))

# resize_to_layout!(fig)

hidedecorations!.(axes)
hidespines!.(axes)
display(fig)
# save(plotsdir("figurePotentials_$(suppressOrNot)_$(spokesOrNot).png"), fig)
save(plotsdir(inputDir, "figurePotentialsAlt.png"), fig)
# save(plotsdir("figurePotentials_$(suppressOrNot)_$(spokesOrNot).pdf"), fig)
save(plotsdir(inputDir, "figurePotentialsAlt.pdf"), fig)