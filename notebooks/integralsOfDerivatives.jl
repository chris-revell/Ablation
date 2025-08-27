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
using InvertedIndices

inputSystems = ["NoHole", "SingleHole", "DoubleHole"]#, "Voronoi", "OldSystem"]

for inputSystem in inputSystems
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
        fileName = datadir("referenceSystems", "$(inputSystem)_testSystem.jld2")
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
    linkTriangleAreas = findCellLinkTriangleAreas(R, A, B)
    𝐡 = hNetwork(R, A, B, F)

    
    curlᶜh = curlᶜ(R, A, B, 𝐡)
    curlᵛh = curlᵛspokes(R, A, B, 𝐡)
    divᶜh = divᶜ(R, A, B, 𝐡)
    divᵛh = divᵛsuppress(R, A, B, 𝐡)
    cocurlᶜh = cocurlᶜ(R, A, B, 𝐡)
    cocurlᵛh = cocurlᵛspokes(R, A, B, 𝐡)
    codivᶜh = codivᶜ(R, A, B, 𝐡)
    codivᵛh = codivᵛsuppress(R, A, B, 𝐡)

    @show minimum(divᵛh)
    @show maximum(divᵛh)

    H = Diagonal(cellAreas)
    E = Diagonal(linkTriangleAreas)
    
    @show inputSystem
    @show sum(cocurlᶜh.*cellAreas)
    @show sum(-1.0.*divᶜh.*cellAreas)
    @show sum(-1.0.*divᵛh.*linkTriangleAreas)
    @show sum(cocurlᵛh.*linkTriangleAreas)
    @show sum(curlᶜh.*cellAreas)
    @show sum(codivᶜh.*cellAreas)
    @show sum(codivᵛh.*linkTriangleAreas)
    @show sum(curlᵛh.*linkTriangleAreas)

end
