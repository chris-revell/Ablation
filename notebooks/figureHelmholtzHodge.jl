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

# @from "$(srcdir("TopologyFunctions.jl"))" using TopologyFunctions
# @from "$(srcdir("GeometryFunctions.jl"))" using GeometryFunctions
# @from "$(srcdir("OrderAroundCell.jl"))" using OrderAroundCell
@from "$(srcdir("PenrosePseudoInversion.jl"))" using PenrosePseudoInversion

# function hAroundCell!(ii, B, h, ϵ, F, currentNeighbourShell, traversedCells, traversedEdges, cellEdgeOrders, cellVertexOrders)
#     # Find an edge belonging to cell ii and shared by any other cell in traversedCells
#     if length(traversedCells)>0
#         startEdge = (findnz(B[traversedCells, :])[2] ∩ findnz(B[ii, :])[1])[1]
#     else 
#         startEdge = cellEdgeOrders[ii][1]
#     end
#     # Find the index of this edge within the ordering of edges around cell ii 
#     startInd = findall(x->x==startEdge, cellEdgeOrders[ii])[1]
#     # Remember in clockwise ordering, cellEdgeOrders[i][1] precedes cellVertexOrders[1]
#     for vertexInd = startInd:(startInd+length(cellVertexOrders[ii])-1)
#         h[cellEdgeOrders[ii][vertexInd+1]] = h[cellEdgeOrders[ii][vertexInd]] .+ ϵ*F[cellVertexOrders[ii][vertexInd], ii]
#         push!(traversedEdges, cellEdgeOrders[ii][vertexInd]) 
#     end
#     return nothing 
# end

# function hNetwork(R, A, B, F)
#     I = size(B,1)
#     J = size(B,2)
#     K = size(A,2)
#     peripheralEdges = findPeripheralEdges(B)
#     peripheralCells = findnz(B[:, peripheralEdges.==1])[1]
#     cellVertexOrders  = fill(CircularVector(Int64[]), I)
#     cellEdgeOrders    = fill(CircularVector(Int64[]), I)
#     for i = 1:I
#         cellVertexOrders[i], cellEdgeOrders[i] = orderAroundCell(A, B, i)
#     end
#     ϵ = SMatrix{2, 2, Float64}([
#             0.0 1.0
#             -1.0 0.0
#         ])
#     Ā = abs.(A)
#     B̄ = abs.(B)

#     # Ensure we don't start with a boundary cell
#     startCell = rand(collect(1:I)[Not(peripheralCells)])
#     # startCell = 1 
#     traversedCells = Int64[]
#     traversedEdges = Int64[]
#     cellNeighbourMatrix = B*transpose(B)
#     h = fill(SVector{2, Float64}(zeros(2)), J)

#     hAroundCell!(startCell, B, h, ϵ, F, Int64[], traversedCells, traversedEdges, cellEdgeOrders, cellVertexOrders)
#     push!(traversedCells, startCell)
#     while length(traversedCells) < I
#         currentNeighbourShell = setdiff(findnz(cellNeighbourMatrix[:, traversedCells])[1], traversedCells)
#         for ii in currentNeighbourShell
#             hAroundCell!(ii, B, h, ϵ, F, currentNeighbourShell, traversedCells, traversedEdges, cellEdgeOrders, cellVertexOrders)
#             push!(traversedCells, ii)
#         end
#     end

#     for j=2:J
#         h[j] = h[j] - h[1]
#     end
#     h[1] = @SVector zeros(2)

#     return h
# end




inputSystems = ["NoHole"]#, "SingleHole", "DoubleHole"]#, "Voronoi", "OldSystem"]

suppressOrNot = "nosuppress"
spokesOrNot = "nospokes"

fig = Figure(size=(500, 1500), fontsize=24)
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"($l)" for l in individualLetters]

for (col,inputSystem) in enumerate(inputSystems)
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
    linkTriangles = findCellLinkTriangles(R, A, B)
    linkTriangleAreas = findCellLinkTriangleAreas(R, A, B)
    cellPolygons = findCellPolygons(R, A, B)
    𝐡 = hNetwork(R, A, B, F)
    # 𝐡 = [SVector{2,Float64}(1.0,0.0) for _=1:J]

    curlᶜh = curlᶜ(R, A, B, 𝐡)
    curlᶜhMax = max(maximum(abs.(curlᶜh)), 0.0001)
    # curlᵛh = (spokesOrNot=="spokes" ? curlᵛspokes(R, A, B, 𝐡) : curlᵛ(R, A, B, 𝐡))
    curlᵛh = curlᵛspokes(R, A, B, 𝐡)
    curlᵛhMax = max(maximum(abs.(curlᵛh)), 0.0001)
    divᶜh = divᶜ(R, A, B, 𝐡)
    divᶜhMax = max(maximum(abs.(divᶜh)), 0.0001)
    divᵛh = divᵛsuppress(R, A, B, 𝐡)
    divᵛhMax = max(maximum(abs.(divᵛh)), 0.0001)
    cocurlᶜh = cocurlᶜ(R, A, B, 𝐡)
    cocurlᶜhMax = max(maximum(abs.(cocurlᶜh)), 0.0001)
    cocurlᵛh = cocurlᵛspokes(R, A, B, 𝐡)
    cocurlᵛhMax = max(maximum(abs.(cocurlᵛh)), 0.0001)
    codivᶜh = codivᶜ(R, A, B, 𝐡)
    codivᶜhMax = max(maximum(abs.(codivᶜh)), 0.0001)
    codivᵛh = codivᵛsuppress(R, A, B, 𝐡)
    codivᵛhMax = max(maximum(abs.(codivᵛh)), 0.0001)
    
    H = Diagonal(cellAreas)
    E = Diagonal(linkTriangleAreas)
    Lv = geometricLv(R, A, B)
    Lf = geometricLf(R, A, B)
    Lc = geometricLc(R, A, B)
    Lt = geometricLt(R, A, B)
    # ϕpar Lv -divᵛ
    ϕpar, ϕparspectrum = penrosePseudoInversion(Lv, -1.0.*divᵛh, E)
    # ϕperp Lv -codivᵛ
    ϕperp, ϕperpspectrum = penrosePseudoInversion(Lv, -1.0.*codivᵛh, E)
    # upar Lf cocurlᶜ
    upar, uparspectrum = penrosePseudoInversion(Lf, cocurlᶜh, H)
    # uperp Lf curlᶜ
    uperp, uperpspectrum = penrosePseudoInversion(Lf, curlᶜh, H)
    # ϕCapitalpar Lc -divᶜ
    ϕCapitalpar, ϕCapitalparspectrum = penrosePseudoInversion(Lc, -1.0.*divᶜh, H)
    # ϕCapitalperp Lc -codivᶜ
    ϕCapitalperp, ϕCapitalperpspectrum = penrosePseudoInversion(Lc, -1.0.*codivᶜh, H)
    # Upar Lt cocurlᵛ
    Upar, Uparspectrum = penrosePseudoInversion(Lt, cocurlᵛh, E)
    # Uperp Lt curlᵛ
    Uperp, Uperpspectrum = penrosePseudoInversion(Lt, curlᵛh, E)

    # 𝐯 = grad ϕ + rot u + x 
    # 𝐕 = grad ϕCapital + rot U + x 
    # grad ϕ = gradᵛ ϕpar + cogradᵛ ϕperp 
    # grad ϕCapital = gradᶜ ϕCapitalpar + cogradᶜ ϕCapitalperp 
    # rot u = rotᶜ uperp + corotᶜ upar 
    # rot U = rotᵛ Uperp + corotᵛ Upar

    # 𝐡_hh = gradᵛ(R, A, ϕpar) + cogradᵛ(R, A, B, ϕperp) + rotᶜ(R, A, B, uperp) + corotᶜ(R, A, B, upar) 
    # 𝐡_hh .= [𝐡_hh[j].-𝐡_hh[1] for j=1:size(B,2)]
    𝐇_hh = gradᶜ(R, A, B, ϕCapitalpar) + cogradᶜ(R, A, B, ϕCapitalperp) + rotᵛspokes(R, A, B, Uperp) + corotᵛspokes(R, A, B, Upar)
    𝐇_hh .= [𝐇_hh[j].-𝐇_hh[1] for j=1:size(B,2)]

    push!(axes, Axis(fig[1,col], aspect=DataAspect()))
    for i=1:size(B,1)
        poly!(axes[end],cellPolygons[i],color=(:white, 1.0),strokewidth=1,strokecolor=(:black,1.0))
    end
    hidedecorations!(axes[end])
    hidespines!(axes[end])
    
    peripheralEdges = findPeripheralEdges(B).==1

    push!(axes, Axis(fig[2,col], aspect=DataAspect()))
    # scatter!(axes[end], Point{2,Float64}.(𝐡_hh), color=(:red, 0.55))
    scatter!(axes[end], Point{2,Float64}.(𝐇_hh[Not(peripheralEdges)]), color=(:green, 0.2))
    scatter!(axes[end], Point{2,Float64}.(𝐇_hh[peripheralEdges]), color=(:green, 0.2), marker=:cross)

    for i=1:I, k=1:K
        k_js = findall(x->x!=0, A[:,k])
        i_js = findall(x->x!=0, B[i,:])
        js = k_js∩i_js
        lines!(axes[end], Point{2,Float64}.(𝐇_hh[js]))
    end


    # scatter!(axes[end], Point{2,Float64}.(𝐡[Not(peripheralEdges)]), color=(:blue, 0.2))
    # scatter!(axes[end], Point{2,Float64}.(𝐡[peripheralEdges]), color=(:blue, 0.2), marker=:cross)
    Label(fig[2,col, Bottom()], "h")

    push!(axes, Axis(fig[3,col], aspect=DataAspect()))
    # scatter!(axes[end], Point{2,Float64}.([𝐡_hh[j].-𝐡[j] for j=1:size(B,2)]), color=(:red, 0.55))
    scatter!(axes[end], Point{2,Float64}.(([𝐇_hh[j].-𝐡[j] for j=1:size(B,2)])[Not(peripheralEdges)]), color=(:green, 0.2))
    scatter!(axes[end], Point{2,Float64}.(([𝐇_hh[j].-𝐡[j] for j=1:size(B,2)])[peripheralEdges]), color=(:green, 0.2), marker=:cross)
    Label(fig[3,col, Bottom()], "x")
end

display(fig)
save(plotsdir("figureHelmholtz.png"), fig)