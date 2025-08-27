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

@from "$(srcdir("PenrosePseudoInversion.jl"))" using PenrosePseudoInversion

inputSystems = ["NoHole"]#, "SingleHole", "DoubleHole"]#, "Voronoi", "OldSystem"]

suppressOrNot = "nosuppress"
spokesOrNot = "nospokes"

fig = Figure(size=(500, 1500), fontsize=24)
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"($l)" for l in individualLetters]


# inputSystem = "VoronoiPressurised"
inputSystem = "NoHole"
col=1

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
    # fileName = datadir("referenceSystems", "$(inputSystem).jld2")
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

# integ = vertexModel(abstol = 1e-8,
#                     reltol = 1e-8,
#                     nRows=9,
#                     nCycles=1,
#                     printToggle=1,
#                     frameDataToggle=0,
#                     frameImageToggle=0,
#                     videoToggle=0,
#                     setRandomSeed=123,
#                     divisionToggle=0,
#                     pressureExternal=0.,
#                 )
# R = reinterpret(SVector{2,Float64}, integ.u) 
# params, matrices = integ.p
# @unpack A, B, F = matrices 

I = size(B,1)
J = size(B,2)
K = size(A,2)
cellAreas = findCellAreas(R, A, B)
linkTriangles = findCellLinkTriangles(R, A, B)
linkTriangleAreas = findCellLinkTriangleAreas(R, A, B)
cellPolygons = findCellPolygons(R, A, B)
𝐡 = hNetwork(R, A, B, F)

boundaryVertices = findBoundaryVertices(A, B).==1
boundaryCells = findBoundaryCells(B).==1
boundaryEdges = findBoundaryEdges(B).==1
notBoundaryEdges = findBoundaryEdges(B).==0


𝐡 .= [𝐡[j]-𝐡[findfirst(x->x, boundaryEdges)] for j=1:J]

curlᶜh = curlᶜ(R, A, B, 𝐡)
curlᵛh = curlᵛspokes(R, A, B, 𝐡)
divᶜh = divᶜ(R, A, B, 𝐡)
divᵛh = divᵛsuppress(R, A, B, 𝐡)
cocurlᶜh = cocurlᶜ(R, A, B, 𝐡)
cocurlᵛh = cocurlᵛspokes(R, A, B, 𝐡)
codivᶜh = codivᶜ(R, A, B, 𝐡)
codivᵛh = codivᵛsuppress(R, A, B, 𝐡)

H = Diagonal(cellAreas)
E = Diagonal(linkTriangleAreas)
Lv = geometricLv(R, A, B)
Lf = geometricLf(R, A, B)
Lc = geometricLc(R, A, B)
Lt = geometricLt(R, A, B)
ϕpar, ϕparspectrum = penrosePseudoInversion(Lv, -1.0.*divᵛh, E)
ϕperp, ϕperpspectrum = penrosePseudoInversion(Lv, -1.0.*codivᵛh, E)
upar, uparspectrum = penrosePseudoInversion(Lf, cocurlᶜh, H)
uperp, uperpspectrum = penrosePseudoInversion(Lf, curlᶜh, H)
ϕCapitalpar, ϕCapitalparspectrum = penrosePseudoInversion(Lc, -1.0.*divᶜh, H)
ϕCapitalperp, ϕCapitalperpspectrum = penrosePseudoInversion(Lc, -1.0.*codivᶜh, H)
Upar, Uparspectrum = penrosePseudoInversion(Lt, cocurlᵛh, E)
Uperp, Uperpspectrum = penrosePseudoInversion(Lt, curlᵛh, E)

# 𝐯 = grad ϕ + rot u + x 
# 𝐕 = grad ϕCapital + rot U + x 
# grad ϕ = gradᵛ ϕpar + cogradᵛ ϕperp 
# grad ϕCapital = gradᶜ ϕCapitalpar + cogradᶜ ϕCapitalperp 
# rot u = rotᶜ uperp + corotᶜ upar 
# rot U = rotᵛ Uperp + corotᵛ Upar

𝐡_hh = gradᵛ(R, A, ϕpar) + cogradᵛ(R, A, B, ϕperp) + rotᶜ(R, A, B, uperp) + corotᶜ(R, A, B, upar) 
# 𝐡_hh .= [𝐡_hh[j].-𝐡_hh[1] for j=1:size(B,2)]
𝐇_hh = gradᶜ(R, A, B, ϕCapitalpar) + cogradᶜ(R, A, B, ϕCapitalperp) + rotᵛspokes(R, A, B, Uperp) + corotᵛspokes(R, A, B, Upar)
# 𝐇_hh .= [𝐇_hh[j].-𝐇_hh[1] for j=1:size(B,2)]

push!(axes, Axis(fig[1,col], aspect=DataAspect()))
for i=1:size(B,1)
    poly!(axes[end],cellPolygons[i],color=(:white, 1.0),strokewidth=1,strokecolor=(:black,1.0))
end
hidedecorations!(axes[end])
hidespines!(axes[end])


push!(axes, Axis(fig[2,col], aspect=DataAspect()))
# for i=1:I
#     orderedVerts, orderedEdges = orderAroundCell(A, B, i)
#     lines!(axes[end], Point{2,Float64}.(𝐡[orderedEdges[0:end]]), color=(:blue, 0.5))
#     # lines!(axes[end], Point{2,Float64}.(𝐡_hh[orderedEdges[0:end]]), color=(:red, 0.5))
#     lines!(axes[end], Point{2,Float64}.(𝐇_hh[orderedEdges[0:end]]), color=(:green, 0.5))
# end
for j=1:J
    if !boundaryEdges[j] 
        lines!(axes[end], Point{2,Float64}.([𝐡[j], 𝐇_hh[j]]), color=(:blue, 0.5))
    end
end
# scatter!(axes[end], Point{2,Float64}.(𝐡_hh[notBoundaryEdges]), color=(:red, 0.5))
# scatter!(axes[end], Point{2,Float64}.(𝐡_hh[boundaryEdges]), color=(:red, 0.5), marker=:cross)
scatter!(axes[end], Point{2,Float64}.(𝐇_hh[notBoundaryEdges]), color=(:green, 0.5))
# scatter!(axes[end], Point{2,Float64}.(𝐇_hh[boundaryEdges]), color=(:green, 0.5), marker=:cross)
scatter!(axes[end], Point{2,Float64}.(𝐡[notBoundaryEdges]), color=(:blue, 0.5))
# scatter!(axes[end], Point{2,Float64}.(𝐡[boundaryEdges]), color=(:blue, 0.5), marker=:cross)


xlims!(axes[end], (-0.1,0.1))
ylims!(axes[end], (-0.1,0.1))

Label(fig[2,col, Bottom()], "h")

push!(axes, Axis(fig[3,col], aspect=DataAspect()))

# scatter!(axes[end], Point{2,Float64}.(([𝐡_hh[j].-𝐡[j] for j=1:size(B,2)])[notBoundaryEdges)]), color=(:red, 0.5))
# scatter!(axes[end], Point{2,Float64}.(([𝐡_hh[j].-𝐡[j] for j=1:size(B,2)])[boundaryEdges]), color=(:red, 0.5), marker=:cross)
scatter!(axes[end], Point{2,Float64}.(([𝐇_hh[j].-𝐡[j] for j=1:size(B,2)])[notBoundaryEdges]), color=(:green, 0.5))
# scatter!(axes[end], Point{2,Float64}.(([𝐇_hh[j].-𝐡[j] for j=1:size(B,2)])[boundaryEdges]), color=(:green, 0.5), marker=:cross)

# xlims!(axes[end], (-0.2,0.2))
# ylims!(axes[end], (-0.2,0.2))
Label(fig[3,col, Bottom()], "x")

display(fig)
save(plotsdir("figureHelmholtz.png"), fig)