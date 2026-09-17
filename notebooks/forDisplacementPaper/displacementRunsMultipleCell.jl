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

    integ0 = vertexModel(abstol = 1e-8,
                    reltol = 1e-8,
                    nRows=5,
                    nCycles=6,
                    pressureExternal=0.0,
                    γ = γ_in,
                    L₀ = L₀_in,
                    energyModel = "quadratic",
                    divisionToggle=1,
                    printToggle=1,
                    outputToggle=1,
                    frameDataToggle=0,
                    frameImageToggle=0,
                    videoToggle=0,
                    # setRandomSeed=123,                                
                )

    R0 = reinterpret(SVector{2,Float64}, integ0.u) 
    (params0, matrices0) = integ0.p
    
    integ1 = vertexModel(abstol = 1e-8,
                        reltol = 1e-8,
                        sstol = 3e-5,
                        termSteadyState=true,                        
                        initialSystem="argument",
                        R_in=R0,
                        A_in=matrices0.A,
                        B_in=matrices0.B,
                        pressureExternal=0.0,
                        γ = params0.γ,
                        L₀ = params0.L₀,
                        energyModel = "quadratic",
                        divisionToggle=0,                        
                        printToggle=1,
                        outputToggle=1,
                        frameDataToggle=0,
                        frameImageToggle=0,
                        videoToggle=0,                       
                    )

    dateString = "$(Dates.format(Dates.now(),"yy-mm-dd-HH-MM-SS"))"
    !isdir(datadir("displacementFields", "MultiCell", dateString)) ? mkpath(datadir("displacementFields", "MultiCell", dateString)) : nothing 
    R = reinterpret(SVector{2,Float64}, integ1.u)
    params = integ1.p[1]
    matrices = integ1.p[2]
    jldsave(datadir("displacementFields", "MultiCell", dateString, "$(dateString)_InitialSystem.jld2"); 
                R,
                params,
                matrices
            )
else
    # dateString = "26-06-05-15-08-01"
    # dateString = "26-06-19-08-32-27"
    # dateString = "26-07-07-16-49-05"
    # dateString = "26-06-19-08-32-27"
    # dateString = "26-08-27-12-48-37"
    dateString = "26-08-27-12-48-37_2"
    dataDict = load(datadir("displacementFields", "MultiCell", dateString, "$(dateString)_InitialSystem.jld2");
                    typemap=Dict("VertexModel.../VertexModelContainers.jl.VertexModelContainers.MatricesContainer" => MatricesContainer, 
                                "VertexModel.../VertexModelContainers.jl.VertexModelContainers.ParametersContainer" => ParametersContainer
                    )
                )
    # Import system data
    @unpack R, params, matrices = dataDict
end

cellCentres = findCellCentresOfMass(R, matrices.A, matrices.B)

𝐡 = hNetwork(R, matrices.A, matrices.B, matrices.F)
# Stress tensors 
σᵢ = σ(R, matrices.A, matrices.B, 𝐡)
# Deviatoric stress 
σDᵢ = [σᵢ[i] .- 0.5*tr(σᵢ[i]) for i=1:size(matrices.B,1)]
σDSᵢ = 0.5.*(σDᵢ .+ transpose.(σDᵢ))
# ~\ref{eq:shearstressexact}
ζᵢ = [sqrt(-det(σDSᵢ[i])) for i=1:size(matrices.B,1)]
p_eff = -0.5.*cocurlᶜ(R, matrices.A, matrices.B, 𝐡)
systemCOM = sum(R)./size(matrices.B,1)
radii = norm.([r.-systemCOM for r in cellCentres])

peripheralCells = findPeripheralCells(matrices.B)
peripheralCellIndices = findall(x->x!=0, peripheralCells)
internalCellIndices = findall(x->x!=0, peripheralCells)
internalCells = peripheralCells.==0

orderedByRadius = sortperm(radii[internalCells])
orderedByPeff = sortperm(p_eff[internalCells]; rev=true)

randOrder = randperm(length(internalCellIndices))
pairs = [sort(randOrder[2*i-1:2*i]) for i=1:length(randOrder)÷2]

ablatedStrings = [f[28:end-5] for f in readdir(datadir("displacementFields", "MultiCell", dateString)) if occursin("Ablated",f)]
ablatedPairs = [parse.(Int64, split(f, "_")) for f in ablatedStrings]

neighbourMatrix = matrices.B*matrices.Bᵀ

# for p in pairs[1:12]
for p in ablatedPairs
    @show p
    # if isfile(datadir("displacementFields", "MultiCell", dateString, "$(dateString)_Ablated$(p[1])_$(p[2]).jld2"))
    #     # skip
    # elseif p[1]∈findall(x->x!=0, neighbourMatrix[p[2],:])
    #     # skip
    # else
    #     println("Cells $(p[1]), $(p[2]) ablated")
    #     Rtmp, Atmp, Btmp = ablateCells(R, matrices.A, matrices.B, p)

    #     integAblated = vertexModel(abstol = 1e-8,
    #                     reltol = 1e-8,
    #                     sstol = 3e-5,
    #                     termSteadyState=true,                        
    #                     initialSystem="argument",
    #                     # nCycles=0.5,
    #                     R_in=Rtmp,
    #                     A_in=Atmp,
    #                     B_in=Btmp,
    #                     pressureExternal=0.0,
    #                     γ = params.γ,
    #                     L₀ = params.L₀,
    #                     energyModel = "quadratic",
    #                     divisionToggle=0,                        
    #                     printToggle=1,
    #                     outputToggle=1,
    #                     frameDataToggle=0,
    #                     frameImageToggle=0,
    #                     videoToggle=0,
    #                 )
                        
    #     paramsAblated, matricesAblated = integAblated.p
    #     Rablated = reinterpret(SVector{2,Float64}, integAblated.u)
    #     Aablated = matricesAblated.A
    #     Bablated = matricesAblated.B
    #     Fablated = matricesAblated.F
    #     Fmax = maximum(norm.(sum(Fablated, dims=2)))
    #     @show Fmax
    #     ablationCOM1 = cellCentres[p[1]] 
    #     ablationCOM2 = cellCentres[p[2]] 
    #     jldsave(datadir("displacementFields", "MultiCell", dateString, "$(dateString)_Ablated$(p[1])_$(p[2]).jld2"); 
    #         Rablated,
    #         Aablated, 
    #         Bablated, 
    #         Fablated, 
    #         ablationCOM1,
    #         ablationCOM2,
    #     )
    # end
    
    if isfile(datadir("displacementFields", "MultiCell", dateString, "$(dateString)_Divided$(p[1])_$(p[2]).jld2"))
        # skip
    elseif p[1]∈findall(x->x!=0, neighbourMatrix[p[2],:])
        # skip
    else
        println("Cells $(p[1]), $(p[2]) divided")
        Rtmp0, Atmp0, Btmp0, shortVec0 = divideCell(R, matrices.A, matrices.B, p[1])
        Rtmp, Atmp, Btmp, shortVec1 = divideCell(Rtmp0, Atmp0, Btmp0, p[2])

        integDivided = vertexModel(abstol = 1e-8,
                        reltol = 1e-8,
                        sstol = 3e-5,
                        termSteadyState=true,                        
                        initialSystem="argument",
                        # nCycles=0.5,
                        R_in=Rtmp,
                        A_in=Atmp,
                        B_in=Btmp,
                        pressureExternal=0.0,
                        γ = params.γ,
                        L₀ = params.L₀,
                        energyModel = "quadratic",
                        divisionToggle=0,                        
                        printToggle=1,
                        outputToggle=1,
                        frameDataToggle=0,
                        frameImageToggle=0,
                        videoToggle=0,
                    )
        
        paramsDivided, matricesDivided = integDivided.p
        Rdivided = reinterpret(SVector{2,Float64}, integDivided.u)
        Adivided = matricesDivided.A
        Bdivided = matricesDivided.B
        Fdivided = matricesDivided.F
        Fmax = maximum(norm.(sum(Fdivided, dims=2)))
        @show Fmax
        divisionCOM1 = cellCentres[p[1]] 
        divisionCOM2 = cellCentres[p[2]] 
        jldsave(datadir("displacementFields", "MultiCell", dateString, "$(dateString)_Divided$(p[1])_$(p[2]).jld2"); 
            Rdivided,
            Adivided, 
            Bdivided, 
            Fdivided, 
            divisionCOM1,
            divisionCOM2,
        )
    end
end



# integ0 = vertexModel(abstol = 1e-9,
#         reltol = 1e-9,
#         nRows = 15,
#         nCycles = 2,
#         divisionToggle = 1,
#         outputToggle = 0,
#         # frameDataToggle = 0,
#         # frameImageToggle = 0,
#         energyModel = "quadratic",
#         γ = γ_in,
#         L₀ = L₀_in,
#     )

#  integ1 = vertexModel(abstol = 1e-9,
#         reltol = 1e-9,
#         initialSystem = "argument",
#         nCycles = 0.5,
#         pressureExternal=0.0,
#         divisionToggle = 0,
#         outputToggle = 0,
#         # frameDataToggle = 0,
#         # frameImageToggle = 0,
#         energyModel = "quadratic",
#         R_in = R0,
#         A_in = matrices0.A, 
#         B_in = matrices0.B,
#         γ = params0.γ,
#         L₀ = params0.L₀,
#     )

# integAblated = vertexModel(abstol = 1e-9,
        #                     reltol = 1e-9,
        #                     initialSystem="argument",
        #                     divisionToggle=0,
        #                     R_in=Rtmp,
        #                     A_in=Atmp,
        #                     B_in=Btmp,     
        #                     pressureExternal=0.0,                       
        #                     nCycles=0.5,
        #                     outputToggle=1,
        #                     energyModel="quadratic",
        #                     γ = params.γ,
        #                     L₀ = params.L₀,
        #                 )

        # integDivided = vertexModel(abstol = 1e-9,
        #                     reltol = 1e-9,
        #                     initialSystem="argument",
        #                     divisionToggle=0,
        #                     R_in=Rtmp,
        #                     A_in=Atmp,
        #                     B_in=Btmp,
        #                     nCycles=0.5,
        #                     outputToggle=1,
        #                     energyModel="quadratic",
        #                     γ = params.γ,
        #                     L₀ = params.L₀,
        #                 )