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

# function num_to_string(x, fmt="%.1g")
function numToString(x, fmt="%.1e")
    Printf.format(Printf.Format(fmt), x)
end

inputDir = "quadraticPotentialNoPressure"; isdir(plotsdir(inputDir)) ? nothing : mkdir(plotsdir(inputDir))
inputSystems = ["NoHole", "SingleHole", "DoubleHole"]#, "Voronoi", "OldSystem"]

fig = Figure(size=(1500, 2000), fontsize=24)
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"(%$l)" for l in individualLetters]

gl0 = GridLayout(fig[1,1])
Label(gl0[1,2:5], "Divergences", fontsize=36)
Label(gl0[1,6:9], "Curls", fontsize=36)
Label(gl0[2,2], "Cells", fontsize=36)
Label(gl0[2,3], "", fontsize=36)
Label(gl0[2,4], "Vertices", fontsize=36)
Label(gl0[2,5], "", fontsize=36)
Label(gl0[2,6], "Cells", fontsize=36)
Label(gl0[2,7], "", fontsize=36)
Label(gl0[2,8], "Vertices", fontsize=36)
Label(gl0[2,9], "", fontsize=36)

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
    kᵖ = findPeripheralVertices(A, B)
    kⁱ = 1 .- kᵖ
    Kⁱ = sum(kⁱ)
    iᵖ = findPeripheralCells(B)
    linkTriangles = findCellLinkTriangles(R, A, B)[kⁱ.==1]
    cellPolygons = findCellPolygons(R, A, B)
    𝐡 = hNetwork(R, A, B, F)
    
    curlᶜh = curlᶜ(R, A, B, 𝐡)
    curlᶜhMax = max(maximum(abs.(curlᶜh)), 0.0001)
    curlᶜhLims = (-curlᶜhMax, curlᶜhMax)
    curlᵛh = curlᵛspokes(R, A, B, 𝐡)[kⁱ.==1]
    curlᵛhMax = max(maximum(abs.(curlᵛh)), 0.0001)
    curlᵛhLims = (-curlᵛhMax, curlᵛhMax)
    divᶜh = divᶜ(R, A, B, 𝐡)
    divᶜhMax = max(maximum(abs.(divᶜh)), 0.0001)
    divᶜhLims = (-divᶜhMax, divᶜhMax)
    divᵛh = divᵛsuppress(R, A, B, 𝐡)[kⁱ.==1]
    divᵛhMax = max(maximum(abs.(divᵛh)), 0.0001)
    divᵛhLims = (-divᵛhMax, divᵛhMax)
    cocurlᶜh = cocurlᶜ(R, A, B, 𝐡)
    cocurlᶜhMax = max(maximum(abs.(cocurlᶜh)), 0.0001)
    cocurlᶜhLims = (-cocurlᶜhMax, cocurlᶜhMax)
    cocurlᵛh = cocurlᵛspokes(R, A, B, 𝐡)[kⁱ.==1]
    cocurlᵛhMax = max(maximum(abs.(cocurlᵛh)), 0.0001)
    cocurlᵛhLims = (-cocurlᵛhMax, cocurlᵛhMax)
    codivᶜh = codivᶜ(R, A, B, 𝐡)
    codivᶜhMax = max(maximum(abs.(codivᶜh)), 0.0001)
    codivᶜhLims = (-codivᶜhMax, codivᶜhMax)
    codivᵛh = codivᵛsuppress(R, A, B, 𝐡)[kⁱ.==1]
    codivᵛhMax = max(maximum(abs.(codivᵛh)), 0.0001)
    codivᵛhLims = (-codivᵛhMax, codivᵛhMax)

    gl = GridLayout(fig[2*col, 1])
    glLocals = []
    panel = [0]

    panel[1] += 1
    push!(glLocals, GridLayout(gl[fldmod1(panel[1], 4)...]))
    push!(axes, Axis(glLocals[end][1,1], aspect=DataAspect()))
    for i=1:I
        poly!(axes[end],
            cellPolygons[i],
            color=cocurlᶜh[i],
            colorrange=cocurlᶜhLims,
            colormap=:bam,
            strokewidth=1,
            strokecolor=(:black,0.25)
        )
    end
    Label(glLocals[end][1,1,Bottom()], L"-\mathrm{cocurl}^c",fontsize=36)
    Colorbar(glLocals[end][1,2],
        colorrange=cocurlᶜhLims,
        colormap=:bam,
        height=Relative(0.8),
        ticks=([cocurlᶜhLims[1],0.0,cocurlᶜhLims[2]], [numToString(cocurlᶜhLims[1]),"0.0",numToString(cocurlᶜhLims[2])]),
        ticklabelsize=18,
        ticklabelrotation=π/2
    )

    panel[1] += 1
    push!(glLocals, GridLayout(gl[fldmod1(panel[1], 4)...]))
    push!(axes, Axis(glLocals[end][1,1], aspect=DataAspect()))
    for k=1:Kⁱ
        poly!(axes[end],
            linkTriangles[k],
            color=-divᵛh[k],
            colorrange=divᵛhLims,
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
    Label(glLocals[end][1,1,Bottom()], L"-\mathrm{div}^v", fontsize=36)
    Colorbar(glLocals[end][1,2],
        limits=divᵛhLims,
        colormap=:bam,
        height=Relative(0.8),
        ticks=([divᵛhLims[1],0.0,divᵛhLims[2]], [numToString(divᵛhLims[1]),"0.0",numToString(divᵛhLims[2])]),
        ticklabelsize=18,
        ticklabelrotation=π/2
    )

    panel[1] += 1
    push!(glLocals, GridLayout(gl[fldmod1(panel[1], 4)...]))
    push!(axes, Axis(glLocals[end][1,1], aspect=DataAspect()))
    for i=1:I
        poly!(axes[end],
            cellPolygons[i],
            color=curlᶜh[i],
            colorrange=curlᶜhLims,
            colormap=:bam,
            strokewidth=1,
            strokecolor=(:black,0.25)
        )
    end
    Label(glLocals[end][1,1,Bottom()], L"-\mathrm{curl}^c",fontsize=36)
    Colorbar(glLocals[end][1,2],
        colorrange=curlᶜhLims,
        colormap=:bam,
        height=Relative(0.8),
        ticks=([curlᶜhLims[1],0.0,curlᶜhLims[2]], [numToString(curlᶜhLims[1]),"0.0",numToString(curlᶜhLims[2])]),
        ticklabelsize=18,
        ticklabelrotation=π/2
    )
    
    panel[1] += 1
    push!(glLocals, GridLayout(gl[fldmod1(panel[1], 4)...]))
    push!(axes, Axis(glLocals[end][1,1], aspect=DataAspect()))
    for k=1:Kⁱ
        poly!(axes[end],
            linkTriangles[k],
            color=codivᵛh[k],
            colorrange=codivᵛhLims,
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
    Label(glLocals[end][1,1,Bottom()], L"\mathrm{codiv}^v", fontsize=36)
    Colorbar(glLocals[end][1,2],
        limits=codivᵛhLims,
        colormap=:bam,
        height=Relative(0.8),
        ticks=([codivᵛhLims[1],0.0,codivᵛhLims[2]], [numToString(codivᵛhLims[1]),"0.0",numToString(codivᵛhLims[2])]),
        ticklabelsize=18,
        ticklabelrotation=π/2
    )
    
    panel[1] += 1
    push!(glLocals, GridLayout(gl[fldmod1(panel[1], 4)...]))
    push!(axes, Axis(glLocals[end][1,1], aspect=DataAspect()))
    for i=1:I
        poly!(axes[end],
            cellPolygons[i],
            color=-divᶜh[i],
            colorrange=divᶜhLims,
            colormap=:bam,
            strokewidth=1,
            strokecolor=(:black,0.25)
        )
    end
    Label(glLocals[end][1,1,Bottom()], L"-\mathrm{div}^c",fontsize=36)
    Colorbar(glLocals[end][1,2],
        colorrange=divᶜhLims,
        colormap=:bam,
        height=Relative(0.8),
        ticks=([divᶜhLims[1],0.0,divᶜhLims[2]], [numToString(divᶜhLims[1]),"0.0",numToString(divᶜhLims[2])]),
        ticklabelsize=18,
        ticklabelrotation=π/2
    )

    panel[1] += 1
    push!(glLocals, GridLayout(gl[fldmod1(panel[1], 4)...]))
    push!(axes, Axis(glLocals[end][1,1], aspect=DataAspect()))
    for k=1:Kⁱ
        poly!(axes[end],
            linkTriangles[k],
            color=cocurlᵛh[k],
            colorrange=cocurlᵛhLims,
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
    Label(glLocals[end][1,1,Bottom()], L"-\mathrm{cocurl}^v", fontsize=36)
    Colorbar(glLocals[end][1,2],
        limits=cocurlᵛhLims,
        colormap=:bam,
        height=Relative(0.8),
        ticks=([cocurlᵛhLims[1],0.0,cocurlᵛhLims[2]], [numToString(cocurlᵛhLims[1]),"0.0",numToString(cocurlᵛhLims[2])]),
        ticklabelsize=18,
        ticklabelrotation=π/2
    )
    
    panel[1] += 1
    push!(glLocals, GridLayout(gl[fldmod1(panel[1], 4)...]))
    push!(axes, Axis(glLocals[end][1,1], aspect=DataAspect()))
    for i=1:I
        poly!(axes[end],
            cellPolygons[i],
            color=codivᶜh[i],
            colorrange=codivᶜhLims,
            colormap=:bam,
            strokewidth=1,
            strokecolor=(:black,0.25)
        )
    end
    Label(glLocals[end][1,1,Bottom()], L"\mathrm{codiv}^c",fontsize=36)
    Colorbar(glLocals[end][1,2],
        colorrange=codivᶜhLims,
        colormap=:bam, 
        height=Relative(0.8),
        ticks=([codivᶜhLims[1],0.0,codivᶜhLims[2]], [numToString(codivᶜhLims[1]),"0.0",numToString(codivᶜhLims[2])]), 
        ticklabelsize=18, 
        ticklabelrotation=π/2
    )
    
    panel[1] += 1
    push!(glLocals, GridLayout(gl[fldmod1(panel[1], 4)...]))
    push!(axes, Axis(glLocals[end][1,1], aspect=DataAspect()))
    for k=1:Kⁱ
        poly!(axes[end],
            linkTriangles[k],
            color=curlᵛh[k],
            colorrange=curlᵛhLims,
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
    Label(glLocals[end][1,1,Bottom()], L"-\mathrm{curl}^v", fontsize=36)
    Colorbar(glLocals[end][1,2],
        limits=curlᵛhLims,
        colormap=:bam,
        height=Relative(0.8),
        ticks=([curlᵛhLims[1],0.0,curlᵛhLims[2]], [numToString(curlᵛhLims[1]),"0.0",numToString(curlᵛhLims[2])]),
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
save(plotsdir(inputDir, "Figure7Derivatives.png"), fig)
save(plotsdir(inputDir, "Figure7Derivatives.pdf"), fig)