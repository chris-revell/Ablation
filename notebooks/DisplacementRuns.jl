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

@from "$(srcdir("AblateCells.jl"))" using AblateCells
@from "$(srcdir("Stresses.jl"))" using Stresses
@from "$(srcdir("DivideCell.jl"))" using DivideCell

deNovoSystem = true

γ_in = 0.2
L₀_in = 2.0

if deNovoSystem
    integ0 = vertexModel(
        nRows = 15,
        nCycles = 3,
        divisionToggle = 1,
        outputTotal = 1,
        outputToggle = 0,
        frameDataToggle = 0,
        frameImageToggle = 0,
        printToggle = 1,
        plotCells = 0,
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
        nCycles = 1,
        divisionToggle = 0,
        outputTotal = 1,
        outputToggle = 0,
        frameDataToggle = 0,
        frameImageToggle = 0,
        printToggle = 1,
        plotCells = 0,
        energyModel = "quadratic",
        R_in = R0,
        A_in = matrices0.A, 
        B_in = matrices0.B,
    )

    dateString = "$(Dates.format(Dates.now(),"yy-mm-dd-HH-MM-SS"))"
    !isdir(datadir("displacementFields", dateString)) ? mkpath(datadir("displacementFields", dateString)) : nothing 
    jldsave(datadir("displacementFields", dateString, "$(dateString)_InitialSystem.jld2"); 
                integ1,
            )
else
    dateString = "25-12-11-09-28-09"
    dataDict = load(datadir("displacementFields", dateString, "$(dateString)_InitialSystem.jld2"))
    # Import system data
    @unpack integ1 = dataDict
end

(params1, matrices1) = integ1.p 
R1 = reinterpret(SVector{2,Float64}, integ1.u)

cellCentres = findCellCentresOfMass(R1, matrices1.A, matrices1.B)

peripheralCells = findPeripheralCells(matrices1.B)
testCells = rand(findall(x->x==0, peripheralCells), 10) # Find a random set of non-peripheral cells 

for i in testCells
    if isfile(datadir("displacementFields", dateString, "$(dateString)_Ablated$(i).jld2"))
        # skip
    else
        println("Cell $i, ablated")
        Rtmp, Atmp, Btmp = ablateCells(R1, matrices1.A, matrices1.B, [i])
        integAblated = vertexModel(abstol = 1e-9,
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
                            printToggle=0,
                            energyModel="quadratic",
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
        Rtmp, Atmp, Btmp, shortVec = divideCell(R1, params1, matrices1, i)
        integDivided = vertexModel(abstol = 1e-9,
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
                            printToggle=0,
                            energyModel="quadratic",
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

