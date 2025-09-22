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

I = size(B,1)
J = size(B,2)
K = size(A,2)
cellAreas = findCellAreas(R, A, B)
linkTriangles = findCellLinkTriangles(R, A, B)
linkTriangleAreas = findCellLinkTriangleAreas(R, A, B)
cellPolygons = findCellPolygons(R, A, B)
𝐡 = hNetwork(R, A, B, F)

peripheralEdges = findPeripheralEdges(B).==1
notPeripheralEdges = findPeripheralEdges(B).==0

curlᶜh = curlᶜ(R, A, B, 𝐡)
curlᵛh = curlᵛspokes(R, A, B, 𝐡)
divᶜh = divᶜ(R, A, B, 𝐡)
divᵛh = divᵛsuppress(R, A, B, 𝐡)
cocurlᶜh = cocurlᶜ(R, A, B, 𝐡)
cocurlᵛh = cocurlᵛspokes(R, A, B, 𝐡)
codivᶜh = codivᶜ(R, A, B, 𝐡)
codivᵛh = codivᵛsuppress(R, A, B, 𝐡)

Lv, Lvreindexing = geometricLvHatReduced(R, A, B)
Lf, Lfreindexing = geometricLfHatReduced(R, A, B)
Lc, Lcreindexing = geometricLcHatReduced(R, A, B)
Lt, Ltreindexing = geometricLtHatReduced(R, A, B)
H = Diagonal(cellAreas[Lcreindexing])
E = Diagonal(linkTriangleAreas[Lvreindexing])

ϕpar, ϕparspectrum = penrosePseudoInversion(Lv, -1.0.*divᵛh[Lvreindexing], E)
ϕpar2 = zeros(K)
ϕpar2[Lvreindexing] .= ϕpar
divᵛhDif = Lv*ϕpar2 .+ divᵛh

ϕperp, ϕperpspectrum = penrosePseudoInversion(Lv, -1.0.*codivᵛh[Lvreindexing], E)
ϕperp2 = zeros(K)
ϕperp2[Lvreindexing] .= ϕperp
codivᵛhDif = Lv*ϕperp2 .+ codivᵛh

upar, uparspectrum = penrosePseudoInversion(Lf, cocurlᶜh[Lfreindexing], H)
cocurlᶜhDif = Lf*upar .- cocurlᶜh

uperp, uperpspectrum = penrosePseudoInversion(Lf, curlᶜh[Lfreindexing], H)
curlᶜhDif = Lf*uperp .- curlᶜh

ϕCapitalpar, ϕCapitalparspectrum = penrosePseudoInversion(Lc, -1.0.*divᶜh[Lcreindexing], H)
divᶜhDif = Lc*ϕCapitalpar .+ divᶜh
ϕCapitalperp, ϕCapitalperpspectrum = penrosePseudoInversion(Lc, -1.0.*codivᶜh[Lcreindexing], H)
codivᶜhDif = Lc*ϕCapitalperp .+ codivᶜh

Upar, Uparspectrum = penrosePseudoInversion(Lt, cocurlᵛh[Ltreindexing], E)
Upar2 = zeros(K)
Upar2[Ltreindexing] .= Upar
cocurlᵛhDif = Lt*Upar2 .- cocurlᵛh
Uperp, Uperpspectrum = penrosePseudoInversion(Lt, curlᵛh[Ltreindexing], E)
Uperp2 = zeros(K)
Uperp2[Ltreindexing] .= Uperp
curlᵛhDif = Lt*Uperp2 .- curlᵛh

