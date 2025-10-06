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

for (col,inputSystem) in enumerate(inputSystems)

    gl = GridLayout(fig[1,col])

    # Import system data
    fileName = datadir("referenceSystems", inputDir, "$(inputSystem)_testSystem.jld2")
    importedData = load(fileName)
    R = importedData["R"]
    A = importedData["A"]
    B = importedData["B"]
    F = importedData["F"]

    systemCOM = sum(R)./length(R)
    cellCentres1 = findCellCentresOfMass(R, A, B)
    centralCell = findmin(norm.([cellCentres1[i].-systemCOM for i=1:size(B,1)]))[2]
    neighbourMatrix = dropzeros(B*transpose(B))
    ablatedCells = [centralCell]

    if isfile(datadir("referenceSystems", inputDir, "$(inputSystem)Ablated_testSystem.jld2"))
        inFile = datadir("referenceSystems", inputDir, "$(inputSystem)Ablated_testSystem.jld2")
        importedData = load(inFile)
        @unpack R2, A2, B2, F2, systemCOM2 = importedData
    else
        Rtmp, Atmp, Btmp = ablateCells(R, A, B, ablatedCells)

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
        @show maximum(norm.(sum(F2, dims=2)))
        R2 = reinterpret(SVector{2,Float64}, integ2.u)
        A2 = matrices2.A
        B2 = matrices2.B
        F2 = matrices2.F

        C = findC(A, B)
        centralCellVertices = R[findall(x->x!=0, C[centralCell, :])]
        systemCOM2 = sum(centralCellVertices)./length(centralCellVertices)
        
        jldsave(datadir("referenceSystems", inputDir, "$(inputSystem)Ablated_testSystem.jld2"); 
            R2,
            A2, 
            B2, 
            F2, 
            systemCOM2,
        )
    end
    iᵖ2 = findPeripheralCells(B2).==1
    cellCentres1 = findCellCentresOfMass(R, A, B)
    cellCentres2 = findCellCentresOfMass(R2, A2, B2)
    cellPolygons2 = findCellPolygons(R2, A2, B2)

    cellradii2 = norm.([cellCentres2[i].-systemCOM2 for i=1:size(B2,1)])
    cellDummyDists2 = collect(maximum(cellradii2)/100:maximum(cellradii2)/100:maximum(cellradii2))
    Δrᵢ = cellCentres2.-cellCentres1[Not(ablatedCells)]
    directions = [normalize(Δrᵢ[i])⋅normalize(cellCentres2[i].-systemCOM2) for i=1:size(B2,1)]
    climsDirection = (-1.0,1.0)
    push!(axes, Axis(gl[3,1], aspect=DataAspect()))
    for i=1:size(B2,1)
        poly!(axes[end], cellPolygons2[i], color=(:black, 0.1), strokewidth=1, strokecolor=(:black,0.1))
    end
    arrows!(axes[end], Point{2,Float64}.(cellCentres2), Vec{2,Float64}.(Δrᵢ), color=directions, colormap=:bam, colorrange=climsDirection, lengthscale=1.0)
    scatter!(axes[end], Point{2,Float64}.([systemCOM2]), color=:red, markersize=10)

    𝐡 = hNetwork(R, A, B, F)

    # Stress tensors 
    σᵢ = σ(R, A, B, 𝐡)
    # Deviatoric stress 
    σDᵢ = [σᵢ[i] .- 0.5*tr(σᵢ[i]) for i=1:size(B,1)]
    σDSᵢ = 0.5.*(σDᵢ .+ transpose.(σDᵢ))
    # ~\ref{eq:shearstressexact}
    ζᵢ = [sqrt(-det(σDSᵢ[i])) for i=1:size(B,1)]

    cocurlᶜh = cocurlᶜ(R, A, B, 𝐡)
    C = findC(A, B)
    i_min = findmin(cocurlᶜh)[2]
    i_max = findmax(cocurlᶜh)[2]

    @show ζᵢ[centralCell]
    @show cocurlᶜh[centralCell]
    @show ζᵢ[i_min]
    @show cocurlᶜh[i_min]
    @show ζᵢ[i_max]
    @show cocurlᶜh[i_max]

    if isfile(datadir("referenceSystems", inputDir, "$(inputSystem)Ablated2_testSystem.jld2"))
        inFile = datadir("referenceSystems", inputDir, "$(inputSystem)Ablated2_testSystem.jld2")
        importedData = load(inFile)
        @unpack R2, A2, B2, F2, ablationCentre2, R3, A3, B3, F3, ablationCentre3 = importedData
    else
        Rtmp, Atmp, Btmp = ablateCells(R, A, B, [i_min])
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
        params2, matrices2 = integ2.p
        # @show maximum(norm.(sum(F2, dims=2)))
        R2 = reinterpret(SVector{2,Float64}, integ2.u)
        A2 = matrices2.A
        B2 = matrices2.B
        F2 = matrices2.F
        ablatedCellVertices = R[findall(x->x!=0, C[i_min, :])]
        ablationCentre2 = sum(ablatedCellVertices)./length(ablatedCellVertices)
        Rtmp, Atmp, Btmp = ablateCells(R, A, B, [i_max])
        integ3 = vertexModel(abstol = 1e-9,
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
        params3, matrices3 = integ3.p
        # @show maximum(norm.(sum(F2, dims=2)))
        R3 = reinterpret(SVector{2,Float64}, integ3.u)
        A3 = matrices3.A
        B3 = matrices3.B
        F3 = matrices3.F
        ablatedCellVertices = R[findall(x->x!=0, C[i_max, :])]
        ablationCentre3 = sum(ablatedCellVertices)./length(ablatedCellVertices)
        jldsave(datadir("referenceSystems", inputDir, "$(inputSystem)Ablated2_testSystem.jld2"); 
            R2,
            A2, 
            B2, 
            F2, 
            ablationCentre2,
            R3,
            A3, 
            B3, 
            F3,
            ablationCentre3,
        )
    end

    I = size(B2,1)
    J = size(B2,2)
    K = size(A2,2)

    cellCentres1 = findCellCentresOfMass(R, A, B)
    cellCentres2 = findCellCentresOfMass(R2, A2, B2)
    cellCentres3 = findCellCentresOfMass(R3, A3, B3)

    cellPolygons2 = findCellPolygons(R2, A2, B2)
    cellPolygons3 = findCellPolygons(R3, A3, B3)

    𝐜ⱼ1 = findEdgeMidpoints(R, A)
    𝐜ⱼ2 = findEdgeMidpoints(R2, A2)
    𝐜ⱼ3 = findEdgeMidpoints(R3, A3)


    Δrᵢ2 = cellCentres2.-cellCentres1[Not(i_min)]
    Δrᵢ3 = cellCentres3.-cellCentres1[Not(i_max)]
    
    directions2 = [normalize(Δrᵢ2[i])⋅normalize(cellCentres2[i].-ablationCentre2) for i=1:I]
    directions3 = [normalize(Δrᵢ3[i])⋅normalize(cellCentres3[i].-ablationCentre3) for i=1:I]
    clims = (-1.0,1.0)

    push!(axes, Axis(gl[1,1], aspect=DataAspect()))
    for i=1:I 
        poly!(axes[end], cellPolygons2[i], color=(:black, 0.1), strokewidth=1, strokecolor=(:black,0.1))
    end
    arrows!(axes[end], Point{2,Float64}.(cellCentres2), Vec{2,Float64}.(Δrᵢ2), color=directions2, colormap=:bam, colorrange=clims, lengthscale=1.0)
    scatter!(axes[end], Point{2,Float64}.([ablationCentre2]), color=:red, markersize=10)

    push!(axes, Axis(gl[5,1], aspect=DataAspect()))
    for i=1:I 
        poly!(axes[end], cellPolygons3[i], color=(:black, 0.1), strokewidth=1, strokecolor=(:black,0.1))
    end
    arrows!(axes[end], Point{2,Float64}.(cellCentres3), Vec{2,Float64}.(Δrᵢ3), color=directions3, colormap=:bam, colorrange=clims, lengthscale=1.0)
    scatter!(axes[end], Point{2,Float64}.([ablationCentre3]), color=:red, markersize=10)

    for row in [2,4,6]
        Label(gl[row,1], popfirst!(subfigureLabels), fontsize=24) 
    end

    rowsize!(gl, 1, Relative(0.32))
    rowsize!(gl, 2, Relative(0.01))
    rowsize!(gl, 3, Relative(0.32))
    rowsize!(gl, 4, Relative(0.01))
    rowsize!(gl, 5, Relative(0.32))
    rowsize!(gl, 6, Relative(0.01))
    colsize!(gl, 1, Relative(1.0))
    
end

# push!(axes, Axis(fig[2,:]))
# hlines!(axes[end], [0.0], color=(:black,0.3), linewidth=5, linestyle=:solid)
# push!(axes, Axis(fig[4,:]))
# hlines!(axes[end], [0.0], color=(:black,0.3), linewidth=5, linestyle=:solid)
# push!(axes, Axis(fig[6,:]))
# hlines!(axes[end], [0.0], color=(:black,0.3), linewidth=5, linestyle=:solid)

hidedecorations!.(axes)
hidespines!.(axes)

# rowsize!(fig.layout, 1, Relative(0.24))
# rowsize!(fig.layout, 2, Relative(0.01))
# rowsize!(fig.layout, 3, Relative(0.24))
# rowsize!(fig.layout, 4, Relative(0.01))
# rowsize!(fig.layout, 5, Relative(0.24))
# rowsize!(fig.layout, 6, Relative(0.01))
# rowsize!(fig.layout, 7, Relative(0.24))

colsize!(fig.layout, 1, Relative(0.25))
colsize!(fig.layout, 2, Relative(0.25))
colsize!(fig.layout, 3, Relative(0.25))
colsize!(fig.layout, 4, Relative(0.25))

resize_to_layout!(fig)

display(fig)

save(plotsdir(inputDir, "figure11DisplacementGrid.png"), fig)
save(plotsdir(inputDir, "figure11DisplacementGrid.pdf"), fig)