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
using Statistics

inputDir = "quadraticPotentialNoPressure"; isdir(plotsdir(inputDir)) ? nothing : mkdir(plotsdir(inputDir))
# inputDir = "quadraticPotentialWithPressure"; isdir(plotsdir(inputDir)) ? nothing : mkdir(plotsdir(inputDir))
inputSystems = ["NoHole", "SingleHole", "DoubleHole"]#, "Voronoi", "OldSystem"]

fig = Figure(size=(2500, 1500), fontsize=24)
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"($l)" for l in individualLetters]

gl0 = GridLayout(fig[1,1])
Label(gl0[0,2], "", fontsize=48, rotation=π/2)
Label(gl0[1,2], "Divergences", fontsize=48, rotation=π/2)
Label(gl0[2,2], "Curls", fontsize=48, rotation=π/2)
Label(gl0[3,2], "Divergences", fontsize=48, rotation=π/2)
Label(gl0[4,2], "Curls", fontsize=48, rotation=π/2)
Label(gl0[1:2,1], "Dual", fontsize=48, rotation=π/2)
Label(gl0[3:4,1], "Primary", fontsize=48, rotation=π/2)
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
        # cellTensions = importedData["cellTensions"]
        # cellPressures = importedData["cellPressures"]
        # cellPerimeters = importedData["cellPerimeters"]
        # cellAreas = importedData["cellAreas"]
        # cellEffectivePressures = cellPressures .+ cellTensions.*cellPerimeters./(2.0.*cellAreas)
    end
    
    I = size(B,1)
    J = size(B,2)
    K = size(A,2)
    linkTriangles = findCellLinkTriangles(R, A, B)
    cellPolygons = findCellPolygons(R, A, B)
    𝐡 = hNetwork(R, A, B, F)
    
    #%%
    # N_v-N_e+N_c=1-n_h
    # @show K-J+I
    # if inputSystem == "NoHole"
    #     @show 1
    # elseif inputSystem == "SingleHole"
    #     @show 0
    # elseif inputSystem == "DoubleHole"
    #     @show -1
    # end
    #%%
    
    boundaryVertices = findBoundaryVertices(A, B).==1
    boundaryCells = findBoundaryCells(B).==1

    # aᵢ = findCellAreas(R, A, B)
    # Eₖ = findCellLinkTriangleAreas(R, A, B)

    curlᶜh = curlᶜ(R, A, B, 𝐡)
    curlᶜhMax = max(maximum(abs.(curlᶜh)), 0.0001)
    # curlᶜhLims = (-curlᶜhMax, curlᶜhMax)
    # curlᵛh = (spokesOrNot=="spokes" ? curlᵛspokes(R, A, B, 𝐡) : curlᵛ(R, A, B, 𝐡))
    curlᵛh = curlᵛspokes(R, A, B, 𝐡)
    curlᵛhMax = max(maximum(abs.(curlᵛh)), 0.0001)
    # curlᵛhLims = (-curlᵛhMax, curlᵛhMax)
    # divᶜh = (suppressOrNot=="suppress" ? divᶜsuppress(R, A, B, 𝐡) : divᶜ(R, A, B, 𝐡))
    divᶜh = divᶜ(R, A, B, 𝐡)
    divᶜhMax = max(maximum(abs.(divᶜh)), 0.0001)
    # divᶜhLims = (-divᶜhMax, divᶜhMax)
    divᵛh = divᵛsuppress(R, A, B, 𝐡)
    divᵛhMax = max(maximum(abs.(divᵛh)), 0.0001)
    # divᵛhLims = (-divᵛhMax, divᵛhMax)
    cocurlᶜh = cocurlᶜ(R, A, B, 𝐡)
    cocurlᶜhMax = max(maximum(abs.(cocurlᶜh)), 0.0001)
    # cocurlᶜhLims = (-cocurlᶜhMax, cocurlᶜhMax)
    # cocurlᵛh = (spokesOrNot=="spokes" ? cocurlᵛspokes(R, A, B, 𝐡) : cocurlᵛ(R, A, B, 𝐡))
    cocurlᵛh = cocurlᵛspokes(R, A, B, 𝐡)
    cocurlᵛhMax = max(maximum(abs.(cocurlᵛh)), 0.0001)
    # cocurlᵛhLims = (-cocurlᵛhMax, cocurlᵛhMax)
    # codivᶜh = (suppressOrNot=="suppress" ? codivᶜsuppress(R, A, B, 𝐡) : codivᶜ(R, A, B, 𝐡))
    codivᶜh = codivᶜ(R, A, B, 𝐡)
    codivᶜhMax = max(maximum(abs.(codivᶜh)), 0.0001)
    # codivᶜhLims = (-codivᶜhMax, codivᶜhMax)
    codivᵛh = codivᵛsuppress(R, A, B, 𝐡)
    codivᵛhMax = max(maximum(abs.(codivᵛh)), 0.0001)
    # codivᵛhLims = (-codivᵛhMax, codivᵛhMax)

    # @show sum(aᵢ .* cocurlᶜh)
    # @show sum(aᵢ .* divᶜh)
    # @show sum(Eₖ .* divᵛh)
    # @show sum(Eₖ .* cocurlᵛh)
    # @show sum(aᵢ .* curlᶜh)
    # @show sum(aᵢ .* codivᶜh)
    # @show sum(Eₖ .* codivᵛh)
    # @show sum(Eₖ .* curlᵛh)
    
    row1Max = max(divᶜhMax, cocurlᵛhMax)
    row1Lims = (-row1Max, row1Max)
    row2Max = max(codivᶜhMax, curlᵛhMax)
    row2Lims = (-row2Max, row2Max)
    row3Max = max(cocurlᶜhMax, divᵛhMax)
    row3Lims = (-row3Max, row3Max)
    row4Max = max(curlᶜhMax, codivᵛhMax)
    row4Lims = (-row4Max, row4Max)
    

    gl = GridLayout(fig[1,1+col*2-1])


    push!(axes, Axis(gl[1,1], aspect=DataAspect()))
    for i=1:I
        poly!(axes[end],cellPolygons[i],color=-divᶜh[i],colorrange=row1Lims,colormap=:bam,strokewidth=1,strokecolor=(:black,0.25))
    end
    Label(gl[1,1,Bottom()], L"-\mathrm{div}^c",fontsize=36)

    push!(axes, Axis(gl[1,2], aspect=DataAspect()))
    for k=1:K
        poly!(axes[end],linkTriangles[k],color=cocurlᵛh[k],colorrange=row1Lims,colormap=:bam,strokewidth=1,strokecolor=(:white,0.0))
    end
    for i=1:I
        poly!(axes[end],cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
    end
    Label(gl[1,2,Bottom()], L"\mathrm{cocurl}^v", fontsize=36)
    Colorbar(gl[1,3], colorrange=row1Lims, colormap=:bam, height=Relative(0.8), ticklabelsize=24, ticks = ([-row1Max,0.0,row1Max], map(x -> @sprintf("%.3f",x), [-row1Max,0.0,row1Max])))

    push!(axes, Axis(gl[2,1], aspect=DataAspect()))
    for i=1:I
        poly!(axes[end],cellPolygons[i],color=codivᶜh[i],colorrange=row2Lims,colormap=:bam,strokewidth=1,strokecolor=(:black,0.25))
    end
    Label(gl[2,1,Bottom()], L"\mathrm{codiv}^c",fontsize=36)
    push!(axes, Axis(gl[2,2], aspect=DataAspect()))
    for k=1:K
        poly!(axes[end],linkTriangles[k],color=curlᵛh[k],colorrange=row2Lims,colormap=:bam,strokewidth=1,strokecolor=(:white,0.0))
    end
    for i=1:I
        poly!(axes[end],cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
    end
    Label(gl[2,2,Bottom()], L"\mathrm{curl}^v", fontsize=36)
    Colorbar(gl[2,3],limits=row2Lims,colormap=:bam, height=Relative(0.8), ticklabelsize=24, ticks = ([-row2Max,0.0,row2Max], map(x -> @sprintf("%.3f",x), [-row2Max,0.0,row2Max])))


    push!(axes, Axis(gl[3,1], aspect=DataAspect()))
    for i=1:I
        poly!(axes[end],cellPolygons[i],color=cocurlᶜh[i],colorrange=row3Lims,colormap=:bam,strokewidth=1,strokecolor=(:black,0.25))
    end
    Label(gl[3,1,Bottom()], L"\mathrm{cocurl}^c",fontsize=36)
    push!(axes, Axis(gl[3,2], aspect=DataAspect()))
    for k=1:K
        poly!(axes[end],linkTriangles[k],color=-divᵛh[k],colorrange=row3Lims,colormap=:bam,strokewidth=1,strokecolor=(:white,0.0))
    end
    for i=1:I
        poly!(axes[end],cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
    end
    Label(gl[3,2,Bottom()], L"-\mathrm{div}^v", fontsize=36)
    Colorbar(gl[3,3],limits=row3Lims,colormap=:bam, height=Relative(0.8), ticklabelsize=24, ticks = ([-row3Max,0.0,row3Max], map(x -> @sprintf("%.3f",x), [-row3Max,0.0,row3Max])))
    
    push!(axes, Axis(gl[4,1], aspect=DataAspect()))
    for i=1:I
        poly!(axes[end],cellPolygons[i],color=curlᶜh[i],colorrange=row4Lims,colormap=:bam,strokewidth=1,strokecolor=(:black,0.25))
    end
    Label(gl[4,1,Bottom()], L"\mathrm{curl}^c",fontsize=36)
    
    push!(axes, Axis(gl[4,2], aspect=DataAspect()))
    for k=1:K
        poly!(axes[end],linkTriangles[k],color=codivᵛh[k],colorrange=row4Lims,colormap=:bam,strokewidth=1,strokecolor=(:white,0.0))
    end
    for i=1:I
        poly!(axes[end],cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
    end
    Label(gl[4,2,Bottom()], L"\mathrm{codiv}^v", fontsize=36)
    Colorbar(gl[4,3],limits=row4Lims,colormap=:bam, height=Relative(0.8), ticklabelsize=24, ticks = ([-row4Max,0.0,row4Max], map(x -> @sprintf("%.3f",x), [-row4Max,0.0,row4Max])))

    

    Label(gl[0,1], "Cells", fontsize=48)
    Label(gl[0,2], "Vertices", fontsize=48)
    
    rowsize!(gl, 0, Relative(0.04))
    rowsize!(gl, 1, Relative(0.24))
    rowsize!(gl, 2, Relative(0.24))
    rowsize!(gl, 3, Relative(0.24))
    rowsize!(gl, 4, Relative(0.24))

    colsize!(gl, 1, Relative(0.495))
    colsize!(gl, 2, Relative(0.495))
    # colsize!(gl, 3, Relative(0.01))
    
end

Label(fig[2,2, Top()], L"(a)", fontsize=48)
Label(fig[2,4, Top()], L"(b)", fontsize=48)
Label(fig[2,6, Top()], L"(b)", fontsize=48)
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
resize_to_layout!(fig)

hidedecorations!.(axes)
hidespines!.(axes)
display(fig)

save(plotsdir(inputDir, "figureDerivativesAlt.png"), fig)
save(plotsdir(inputDir, "figureDerivativesAlt.pdf"), fig)