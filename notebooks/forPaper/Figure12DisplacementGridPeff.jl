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
using FromFile
using InvertedIndices
using LaTeXStrings

@from "$(srcdir("AblateCells.jl"))" using AblateCells
@from "$(srcdir("Stresses.jl"))" using Stresses

inputSystems = ["Large1", "Large2", "Large3", "Large7"]#, "Large5", "Large6", "Large7"]
inputDir = "quadraticPotentialNoPressureMultiples"; isdir(plotsdir(inputDir)) ? nothing : mkdir(plotsdir(inputDir))
# inputDir = "quadraticPotentialWithPressure"; isdir(plotsdir(inputDir)) ? nothing : mkdir(plotsdir(inputDir))

fig = Figure(size=(2000,1500))
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"(%$l)" for l in individualLetters]

for inputSystem in inputSystems

    # Import system data
    fileName = datadir("referenceSystems", inputDir, "$(inputSystem)_testSystem.jld2")
    importedData = load(fileName)
    R = importedData["R"]
    A = importedData["A"]
    B = importedData["B"]
    F = importedData["F"]

    systemCOM = sum(R)./length(R)
    cellCentres = findCellCentresOfMass(R, A, B)
    centralCell = findmin(norm.([cellCentres[i].-systemCOM for i=1:size(B,1)]))[2]
  
    𝐡 = hNetwork(R, A, B, F)
    # Stress tensors 
    σᵢ = σ(R, A, B, 𝐡)
    # Deviatoric stress 
    σDᵢ = [σᵢ[i] .- 0.5*tr(σᵢ[i]) for i=1:size(B,1)]
    σDSᵢ = 0.5.*(σDᵢ .+ transpose.(σDᵢ))
    # ~\ref{eq:shearstressexact}
    ζᵢ = [sqrt(-det(σDSᵢ[i])) for i=1:size(B,1)]

    cocurlᶜh = cocurlᶜ(R, A, B, 𝐡)
    i_min = findmin(cocurlᶜh)[2]
    i_max = findmax(cocurlᶜh)[2]

    # @show ζᵢ[centralCell]
    # @show cocurlᶜh[centralCell]
    # @show ζᵢ[i_min]
    # @show cocurlᶜh[i_min]
    # @show ζᵢ[i_max]
    # @show cocurlᶜh[i_max]

    for i in [i_min, centralCell, i_max]
        if isfile(datadir("referenceSystems", inputDir, "$(inputSystem)Ablated$(i).jld2"))
            inFile = datadir("referenceSystems", inputDir, "$(inputSystem)Ablated$(i).jld2")
            importedData = load(inFile)
            @unpack Rablated, Aablated, Bablated, Fablated, ablationCOM = importedData
        else
            Rtmp, Atmp, Btmp = ablateCells(R, A, B, [i])
            integ2 = vertexModel(abstol = 1e-9,
                                reltol = 1e-9,
                                initialSystem="argument",
                                divisionToggle=0,
                                R_in=Rtmp,
                                A_in=Atmp,
                                B_in=Btmp,
                                pressureExternal=0.0,
                                nCycles=0.5,
                                outputToggle=0,
                                frameDataToggle=0,
                                frameImageToggle=0,
                                videoToggle=0,
                                printToggle=1,
                                energyModel="quadratic",
                            )
            #%%
            params2, matrices2 = integ2.p
            Rablated = reinterpret(SVector{2,Float64}, integ2.u)
            Aablated = matrices2.A
            Bablated = matrices2.B
            Fablated = matrices2.F
            C = findC(A, B)
            centralCellVertices = R[findall(x->x!=0, C[i, :])]
            ablationCOM = sum(centralCellVertices)./length(centralCellVertices) 
            jldsave(datadir("referenceSystems", inputDir, "$(inputSystem)Ablated$(i).jld2"); 
                Rablated,
                Aablated, 
                Bablated, 
                Fablated, 
                ablationCOM,
            )
        end
        
        cellCentresAblated = findCellCentresOfMass(Rablated, Aablated, Bablated)
        cellPolygonsAblated = findCellPolygons(Rablated, Aablated, Bablated)
        Δrᵢ = cellCentresAblated.-cellCentres[Not(i)]
        radiusVectorsAblated = [cellCentres[i].-ablationCOM for i=1:size(B,1)]
        directionsAblated = normalize.(Δrᵢ).⋅normalize.(radiusVectorsAblated[Not(i)])

        push!(axes, Axis(fig, aspect=DataAspect()))
        for i=1:size(Bablated,1)
            # poly!(axes[end], cellPolygons2[i], color=(:black, 0.1), strokewidth=1, strokecolor=(:black,0.1))
            poly!(axes[end], cellPolygonsAblated[i], color=directionsAblated[i], colormap=:managua, colorrange=(-1.0,1.0), strokewidth=1, strokecolor=(:black,0.1))
        end
        # arrows!(axes[end], Point{2,Float64}.(cellCentres2), Vec{2,Float64}.(Δrᵢ), color=directions, colormap=:bam, colorrange=climsDirection, lengthscale=20.0)
        scatter!(axes[end], Point{2,Float64}.([ablationCOM]), color=:red, markersize=10)
    end
end

hidedecorations!.(axes)
hidespines!.(axes)
counter = [0]
indices = [(row, column) for row=1:3, column=1:length(inputSystems)]
for (n, index) in enumerate(indices)
    fig[index[1], index[2]] = axes[n]
    Label(fig[index[1], index[2], Bottom()], subfigureLabels[n], fontsize=24) 
end

resize_to_layout!(fig)

display(fig)

save(plotsdir(inputDir, "Figure12DisplacementGrid.png"), fig)
save(plotsdir(inputDir, "Figure12DisplacementGrid.pdf"), fig)


# rowsize!(gl, 1, Relative(0.32))
# rowsize!(gl, 2, Relative(0.01))
# rowsize!(gl, 3, Relative(0.32))
# rowsize!(gl, 4, Relative(0.01))
# rowsize!(gl, 5, Relative(0.32))
# rowsize!(gl, 6, Relative(0.01))
# colsize!(gl, 1, Relative(1.0))
# push!(axes, Axis(fig[2,:]))
# hlines!(axes[end], [0.0], color=(:black,0.3), linewidth=5, linestyle=:solid)
# push!(axes, Axis(fig[4,:]))
# hlines!(axes[end], [0.0], color=(:black,0.3), linewidth=5, linestyle=:solid)
# push!(axes, Axis(fig[6,:]))
# hlines!(axes[end], [0.0], color=(:black,0.3), linewidth=5, linestyle=:solid)
# rowsize!(fig.layout, 1, Relative(0.24))
# rowsize!(fig.layout, 2, Relative(0.01))
# rowsize!(fig.layout, 3, Relative(0.24))
# rowsize!(fig.layout, 4, Relative(0.01))
# rowsize!(fig.layout, 5, Relative(0.24))
# rowsize!(fig.layout, 6, Relative(0.01))
# rowsize!(fig.layout, 7, Relative(0.24))
# colsize!(fig.layout, 1, Relative(0.25))
# colsize!(fig.layout, 2, Relative(0.25))
# colsize!(fig.layout, 3, Relative(0.25))
# colsize!(fig.layout, 4, Relative(0.25))