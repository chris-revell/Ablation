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
using Printf 

# function num_to_string(x, fmt="%.1g")
function numToString(x, fmt="%.1e")
    Printf.format(Printf.Format(fmt), x)
end

@from "$(srcdir("PenrosePseudoInversion.jl"))" using PenrosePseudoInversion

inputDir = "quadraticPotentialNoPressure"; isdir(plotsdir(inputDir)) ? nothing : mkdir(plotsdir(inputDir)); isdir(plotsdir(inputDir)) ? nothing : mkdir(plotsdir(inputDir))
inputSystems = ["NoHole", "SingleHole", "DoubleHole"]

fig = Figure(size=(1500, 2000), fontsize=24)
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"(%$l)" for l in individualLetters]

gl0 = GridLayout(fig[1,1])
Label(gl0[:,1], "", fontsize=36, rotation=π/2)
Label(gl0[1,2], "Cells", fontsize=36)
Label(gl0[1,3], "", fontsize=36)
Label(gl0[1,4], "Vertices", fontsize=36)
Label(gl0[1,5], "", fontsize=36)
Label(gl0[1,6], "Cells", fontsize=36)
Label(gl0[1,7], "", fontsize=36)
Label(gl0[1,8], "Vertices", fontsize=36)
Label(gl0[1,9], "", fontsize=36)

colsize!(gl0, 1, Relative(0.04))
colsize!(gl0, 2, Relative(0.24*0.8))
colsize!(gl0, 3, Relative(0.24*0.2))
colsize!(gl0, 4, Relative(0.24*0.8))
colsize!(gl0, 5, Relative(0.24*0.2))
colsize!(gl0, 6, Relative(0.24*0.8))
colsize!(gl0, 7, Relative(0.24*0.2))
colsize!(gl0, 8, Relative(0.24*0.8))
colsize!(gl0, 9, Relative(0.24*0.2))


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

    gl = GridLayout(fig[2*col, 1])
    glLocals = []
    panel = [0]

    panel[1] += 1
    push!(glLocals, GridLayout(gl[fldmod1(panel[1], 4)...]))
    push!(axes, Axis(glLocals[end][1,1], aspect=DataAspect()))
    for (i, ii) in enumerate(Lfreindexing)
        poly!(axes[end],
            cellPolygons[ii],
            color=upar[i],
            colorrange=uparLims,
            colormap=:bam,
            strokewidth=1,
            strokecolor=(:black,0.25)
        )
    end
    Label(glLocals[end][1,1,Bottom()], L"-u^\parallel", fontsize = 36)
    Colorbar(glLocals[end][1,2],
        limits=uparLims,
        colormap=:bam,
        height=Relative(0.8),
        ticks=([uparLims[1],0.0,uparLims[2]], [numToString(uparLims[1]),"0.0",numToString(uparLims[2])]),
        ticklabelsize=18,
        ticklabelrotation=π/2
    )
    
    panel[1] += 1
    push!(glLocals, GridLayout(gl[fldmod1(panel[1], 4)...]))
    push!(axes, Axis(glLocals[end][1,1], aspect=DataAspect()))
    for (k, kk) in enumerate(Lvreindexing)
        poly!(axes[end],
            linkTriangles[kk],
            color=ϕpar[k],
            colorrange=ϕparLims,
            colormap=:bam,
            strokewidth=1,
            strokecolor=(:white,0.0)
        )
    end
    for i=1:I
        poly!(axes[end],
            cellPolygons[i],
            color=(:white,0.0),
            strokewidth=1,
            strokecolor=(:black,0.25)
        )
    end
    Label(glLocals[end][1,1,Bottom()], L"\phi^\parallel", fontsize = 36)
    Colorbar(glLocals[end][1,2],
        limits=ϕparLims,
        colormap=:bam,
        height=Relative(0.8),
        ticks=([ϕparLims[1],0.0,ϕparLims[2]], [numToString(ϕparLims[1]),"0.0",numToString(ϕparLims[2])]), 
        ticklabelsize=18, 
        ticklabelrotation=π/2
    )
    
    panel[1] += 1
    push!(glLocals, GridLayout(gl[fldmod1(panel[1], 4)...]))
    push!(axes, Axis(glLocals[end][1,1], aspect=DataAspect()))
    for (i, ii) in enumerate(Lfreindexing)
        poly!(axes[end],
            cellPolygons[ii],
            color=uperp[i],
            colorrange=uperpLims,
            colormap=:bam,
            strokewidth=1,
            strokecolor=(:black,0.25)
        )
    end
    Label(glLocals[end][1,1,Bottom()], L"-u^\perp", fontsize = 36)
    Colorbar(glLocals[end][1,2],
        limits=uperpLims,
        colormap=:bam, 
        height=Relative(0.8), 
        ticks=([uperpLims[1],0.0,uperpLims[2]], [numToString(uperpLims[1]),"0.0",numToString(uperpLims[2])]),
        ticklabelsize=18,
        ticklabelrotation=π/2
    )
    
    panel[1] += 1
    push!(glLocals, GridLayout(gl[fldmod1(panel[1], 4)...]))
    push!(axes, Axis(glLocals[end][1,1], aspect=DataAspect()))
    for (k, kk) in enumerate(Lvreindexing)
        poly!(axes[end],
            linkTriangles[kk],
            color=-ϕperp[k],
            colorrange=ϕperpLims, 
            colormap=:bam,
            strokewidth=1,
            strokecolor=(:white,0.0)
        )
    end
    for i=1:I
        poly!(axes[end],
            cellPolygons[i],
            color=(:white,0.0),
            strokewidth=1,
            strokecolor=(:black,0.25)
        )
    end
    Label(glLocals[end][1,1,Bottom()], L"-\phi^\perp", fontsize = 36)
    Colorbar(glLocals[end][1,2],
        limits=ϕperpLims,
        colormap=:bam,
        height=Relative(0.8),
        ticks=([ϕperpLims[1],0.0,ϕperpLims[2]], [numToString(ϕperpLims[1]),"0.0",numToString(ϕperpLims[2])]),
        ticklabelsize=18,
        ticklabelrotation=π/2
    )
    
    panel[1] += 1
    push!(glLocals, GridLayout(gl[fldmod1(panel[1], 4)...]))
    push!(axes, Axis(glLocals[end][1,1], aspect=DataAspect()))
    for (i, ii) in enumerate(Lcreindexing)
        poly!(axes[end],
            cellPolygons[ii],
            color=ϕCapitalpar[i],
            colorrange=ϕCapitalparLims,
            colormap=:bam,
            strokewidth=1,
            strokecolor=(:black,0.25)
        )
    end
    Label(glLocals[end][1,1,Bottom()], L"\Phi^\parallel", fontsize = 36)
    Colorbar(glLocals[end][1,2],
        limits=ϕCapitalparLims,
        colormap=:bam,
        height=Relative(0.8),
        ticks=([ϕCapitalparLims[1],0.0,ϕCapitalparLims[2]], [numToString(ϕCapitalparLims[1]),"0.0",numToString(ϕCapitalparLims[2])]), 
        ticklabelsize=18,
        ticklabelrotation=π/2
    )

    panel[1] += 1
    push!(glLocals, GridLayout(gl[fldmod1(panel[1], 4)...]))
    push!(axes, Axis(glLocals[end][1,1], aspect=DataAspect()))
    for (k, kk) in enumerate(Ltreindexing)
        poly!(axes[end],
            linkTriangles[kk],
            color=Upar[k],
            colorrange=UparLims,
            colormap=:bam,
            strokewidth=1,
            strokecolor=(:white,0.0)
        )
    end
    for i=1:I
        poly!(axes[end],
            cellPolygons[i],
            color=(:white,0.0),
            strokewidth=1,
            strokecolor=(:black,0.25)
        )
    end
    Label(glLocals[end][1,1,Bottom()], L"-U^\parallel", fontsize = 36)
    Colorbar(glLocals[end][1,2],
        limits=UparLims,
        colormap=:bam, 
        height=Relative(0.8), 
        ticks=([UparLims[1],0.0,UparLims[2]], [numToString(UparLims[1]),"0.0",numToString(UparLims[2])]), 
        ticklabelsize=18, 
        ticklabelrotation=π/2
    )
    
    panel[1] += 1
    push!(glLocals, GridLayout(gl[fldmod1(panel[1], 4)...]))
    push!(axes, Axis(glLocals[end][1,1], aspect=DataAspect()))
    for (i, ii) in enumerate(Lcreindexing)
        poly!(axes[end],
            cellPolygons[ii],
            color=-ϕCapitalperp[i],
            colorrange=ϕCapitalperpLims,
            colormap=:bam,
            strokewidth=1,
            strokecolor=(:black,0.25)
        )
    end
    Label(glLocals[end][1,1,Bottom()], L"-\Phi^\perp", fontsize = 36)
    Colorbar(glLocals[end][1,2],
        limits=ϕCapitalperpLims,
        colormap=:bam,
        height=Relative(0.8),
        ticks=([ϕCapitalperpLims[1],0.0,ϕCapitalperpLims[2]], [numToString(ϕCapitalperpLims[1]),"0.0",numToString(ϕCapitalperpLims[2])]),
        ticklabelsize=18,
        ticklabelrotation=π/2
    )

    panel[1] += 1
    push!(glLocals, GridLayout(gl[fldmod1(panel[1], 4)...]))
    push!(axes, Axis(glLocals[end][1,1], aspect=DataAspect()))
    for (k, kk) in enumerate(Ltreindexing)
        poly!(axes[end],
            linkTriangles[kk],
            color=Uperp[k],
            colorrange=UperpLims,
            colormap=:bam,
            strokewidth=1,
            strokecolor=(:white,0.0)
        )
    end
    for i=1:I
        poly!(axes[end],
            cellPolygons[i],
            color=(:white,0.0),
            strokewidth=1,
            strokecolor=(:black,0.25)
            )
    end
    Label(glLocals[end][1,1,Bottom()], L"-U^\perp", fontsize = 36)
    Colorbar(glLocals[end][1,2],
        limits=UperpLims,
        colormap=:bam, 
        height=Relative(0.8), 
        ticks=([UperpLims[1],0.0,UperpLims[2]], [numToString(UperpLims[1]),"0.0",numToString(UperpLims[2])]), 
        ticklabelsize=18, 
        ticklabelrotation=π/2
    )
    
    glLabels = GridLayout(gl[:,0])
    Label(glLabels[1,1], "Primal", fontsize=36, rotation=π/2)
    Label(glLabels[2,1], subfigureLabels[col], fontsize=36)
    Label(glLabels[3,1], "Dual", fontsize=36, rotation=π/2)
    rowsize!(glLabels, 1, Relative(0.45))
    rowsize!(glLabels, 2, Relative(0.1))
    rowsize!(glLabels, 3, Relative(0.45))

    colsize!(gl, 0, Relative(0.04))
    colsize!(gl, 1, Relative(0.24))
    colsize!(gl, 2, Relative(0.24))
    colsize!(gl, 3, Relative(0.24))
    colsize!(gl, 4, Relative(0.24))

    for p=1:8
        colsize!(glLocals[p], 1, Relative(0.8))
        colsize!(glLocals[p], 2, Relative(0.2))
        rowsize!(glLocals[p], 1, Relative(1.0))
    end
    
    rowsize!(gl, 1, Relative(0.5))
    rowsize!(gl, 2, Relative(0.5))

end


push!(axes, Axis(fig[3,:]))
hlines!(axes[end], [0.0], color=(:black,0.3), linewidth=5, linestyle=:solid)
push!(axes, Axis(fig[5,:]))
hlines!(axes[end], [0.0], color=(:black,0.3), linewidth=5, linestyle=:solid)

rowsize!(fig.layout, 1, Relative(0.05))
rowsize!(fig.layout, 2, Relative(0.315))
rowsize!(fig.layout, 3, Relative(0.0025))
rowsize!(fig.layout, 4, Relative(0.315))
rowsize!(fig.layout, 5, Relative(0.0025))
rowsize!(fig.layout, 6, Relative(0.315))

resize_to_layout!(fig)

hidedecorations!.(axes)
hidespines!.(axes)
display(fig)
save(plotsdir(inputDir, "Figure8Potentials.png"), fig)
save(plotsdir(inputDir, "Figure8Potentials.pdf"), fig)