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

inputSystem="NoHole"


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
linkTriangles = findCellLinkTriangles(R, A, B)
linkTriangleAreas = findCellLinkTriangleAreas(R, A, B)
cellPolygons = findCellPolygons(R, A, B)
𝐡 = hNetwork(R, A, B, F)

peripheralVertices = findPeripheralVertices(A, B).==1
peripheralCells = findPeripheralCells(B).==1

ϕ = rand(size(A,2))
Phi = rand(size(B,1))
u = rand(size(B,1))
U = rand(size(A,2))

L_𝒱 = geometricLv(R, A, B)
L_𝒞 = geometricLv(R, A, B)
L_ℱ = geometricLv(R, A, B)
L_𝒯 = geometricLv(R, A, B)

gradᵛϕ = gradᵛ(R, A, ϕ)
minusdivᵛgradᵛϕ = -divᵛsuppress(R, A, B, gradᵛϕ)
@show L_𝒱*ϕ == minusdivᵛgradᵛϕ
gradᶜPhi = gradᶜ(R, A, B, Phi)
minusdivᶜgradᶜPhi = -divᶜsuppress(R, A, B, gradᶜPhi)
@show L_𝒞*Phi == minusdivᶜgradᶜPhi
rotᶜu = rotᶜ(R, A, B, u)
curlᶜrotᶜu = curlᶜ(R, A, B, rotᶜu)
@show L_ℱ*u == curlᶜrotᶜu
rotᵛU = rotᵛ(R, A, B, U)
curlᵛrotᵛU = curlᵛ(R, A, B, rotᵛU)
@show L_𝒯*U == curlᵛrotᵛU

