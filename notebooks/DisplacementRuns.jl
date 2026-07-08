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
using Dates 
using CircularArrays
using OrdinaryDiffEq
using Printf
using DiffEqCallbacks

@from "$(srcdir("AblateCells.jl"))" using AblateCells
@from "$(srcdir("Stresses.jl"))" using Stresses
@from "$(srcdir("DivideCell.jl"))" using DivideCell

deNovoSystem = false

γ_in = 0.2
L₀_in = 3.0

if deNovoSystem
    integ0 = vertexModel(
        nRows = 15,
        nCycles = 3,
        divisionToggle = 1,
        outputToggle = 0,
        frameDataToggle = 0,
        frameImageToggle = 0,
        energyModel = "quadratic",
        γ = γ_in,
        L₀ = L₀_in,
    )

    #%%
    (params0, matrices0) = integ0.p 
    # @unpack A, B = matrices 
    R0 = reinterpret(SVector{2,Float64}, integ0.u)

    integ1 = vertexModel(
        initialSystem = "argument",
        nCycles = 1.0,
        divisionToggle = 0,
        outputToggle = 0,
        frameDataToggle = 0,
        frameImageToggle = 0,
        energyModel = "quadratic",
        R_in = R0,
        A_in = matrices0.A, 
        B_in = matrices0.B,
        γ = params0.γ,
        L₀ = params0.L₀,
    )

    dateString = "$(Dates.format(Dates.now(),"yy-mm-dd-HH-MM-SS"))"
    !isdir(datadir("displacementFields", dateString)) ? mkpath(datadir("displacementFields", dateString)) : nothing 
    R = reinterpret(SVector{2,Float64}, integ1.u)
    params = integ1.p[1]
    matrices = integ1.p[2]
    jldsave(datadir("displacementFields", dateString, "$(dateString)_InitialSystem.jld2"); 
                R,
                params,
                matrices
            )
else
    # dateString = "26-06-05-15-08-01"
    # dateString = "26-06-19-08-32-27"
    dateString = "26-07-07-16-49-05"
    dataDict = load(datadir("displacementFields", dateString, "$(dateString)_InitialSystem.jld2");
                    typemap=Dict("VertexModel.../VertexModelContainers.jl.VertexModelContainers.MatricesContainer" => MatricesContainer, 
                                "VertexModel.../VertexModelContainers.jl.VertexModelContainers.ParametersContainer" => ParametersContainer
                    )
                )
    # Import system data
    @unpack R, params, matrices = dataDict
end

cellCentres = findCellCentresOfMass(R, matrices.A, matrices.B)

peripheralCells = findPeripheralCells(matrices.B)
systemCOM = sum(R)./size(matrices.B,1)
radii = norm.([r.-systemCOM for r in cellCentres])
orderedByRadius = sortperm(radii)
testCells = rand(findall(x->x==0, peripheralCells), 10) # Find a random set of non-peripheral cells 
# testCells = orderedByRadius[1:5:50] # Find a random set of non-peripheral cells 

for i in testCells
    if isfile(datadir("displacementFields", dateString, "$(dateString)_Ablated$(i).jld2"))
        # skip
    else
        println("Cell $i, ablated")
        Rtmp, Atmp, Btmp = ablateCells(R, matrices.A, matrices.B, [i])
        integAblated = vertexModel(abstol = 1e-9,
                            reltol = 1e-9,
                            initialSystem="argument",
                            divisionToggle=0,
                            R_in=Rtmp,
                            A_in=Atmp,
                            B_in=Btmp,                            
                            nCycles=1.0,
                            outputToggle=0,
                            energyModel="quadratic",
                            γ = params.γ,
                            L₀ = params.L₀,
                        )
                        
        #%%
        paramsAblated, matricesAblated = integAblated.p
        Rablated = reinterpret(SVector{2,Float64}, integAblated.u)
        Aablated = matricesAblated.A
        Bablated = matricesAblated.B
        Fablated = matricesAblated.F
        Fmax = maximum(norm.(sum(Fablated, dims=2)))
        @show Fmax
        # C = findC(A, B)
        # centralCellVertices = R[findall(x->x!=0, C[i, :])]
        ablationCOM = cellCentres[i] # sum(centralCellVertices)./length(centralCellVertices) 
        jldsave(datadir("displacementFields", dateString, "$(dateString)_Ablated$(i).jld2"); 
            # integAblated,
            Rablated,
            Aablated, 
            Bablated, 
            Fablated, 
            ablationCOM,
        )
    end
    
    if isfile(datadir("displacementFields", dateString, "$(dateString)_Divided$(i).jld2"))
        # skip
    else
        println("Cell $i, divided")
        Rtmp, Atmp, Btmp, shortVec = divideCell(R, params, matrices, i)
        integDivided = vertexModel(abstol = 1e-9,
                            reltol = 1e-9,
                            initialSystem="argument",
                            divisionToggle=0,
                            R_in=Rtmp,
                            A_in=Atmp,
                            B_in=Btmp,
                            nCycles=0.5,
                            outputToggle=0,
                            energyModel="quadratic",
                            γ = params.γ,
                            L₀ = params.L₀,
                        )
        #%%
        paramsDivided, matricesDivided = integDivided.p
        Rdivided = reinterpret(SVector{2,Float64}, integDivided.u)
        Adivided = matricesDivided.A
        Bdivided = matricesDivided.B
        Fdivided = matricesDivided.F
        Fmax = maximum(norm.(sum(Fdivided, dims=2)))
        @show Fmax
        # C = findC(A, B)
        # centralCellVertices = R[findall(x->x!=0, C[i, :])]
        divisionCOM = cellCentres[i]  # sum(centralCellVertices)./length(centralCellVertices) 
        jldsave(datadir("displacementFields", dateString, "$(dateString)_Divided$(i).jld2"); 
            Rdivided,
            Adivided, 
            Bdivided, 
            Fdivided, 
            divisionCOM,
        )
    end
end

