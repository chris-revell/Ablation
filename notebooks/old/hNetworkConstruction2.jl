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
using InvertedIndices

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
#     nCells = size(B,1)
#     nEdges = size(B,2)
#     nVerts = size(A,2)
#     boundaryEdges = findBoundaryEdges(B)
#     boundaryCells = findnz(B[:, boundaryEdges.==1])[1]
#     cellVertexOrders  = fill(CircularVector(Int64[]), nCells)
#     cellEdgeOrders    = fill(CircularVector(Int64[]), nCells)
#     for i = 1:length(cellVertexOrders)
#         cellVertexOrders[i], cellEdgeOrders[i] = orderAroundCell(A, B, i)
#     end
#     ϵ = SMatrix{2, 2, Float64}([
#             0.0 1.0
#             -1.0 0.0
#         ])
#     Ā = abs.(A)
#     B̄ = abs.(B)

#     # Ensure we don't start with a boundary cell
#     startCell = rand(collect(1:nCells)[Not(boundaryCells)])
#     traversedCells = Int64[]
#     traversedEdges = Int64[]
#     cellNeighbourMatrix = B*transpose(B)
#     h = fill(SVector{2, Float64}(zeros(2)), nEdges)

#     hAroundCell!(startCell, B, h, ϵ, F, Int64[], traversedCells, traversedEdges, cellEdgeOrders, cellVertexOrders)
#     push!(traversedCells, startCell)
#     while length(traversedCells) < nCells
#         currentNeighbourShell = setdiff(findnz(cellNeighbourMatrix[:, traversedCells])[1], traversedCells)
#         for ii in currentNeighbourShell
#             hAroundCell!(ii, B, h, ϵ, F, currentNeighbourShell, traversedCells, traversedEdges, cellEdgeOrders, cellVertexOrders)
#             push!(traversedCells, ii)
#         end
#     end
#     return h
# end


# # integ1 = vertexModel(nRows=19, nCycles=1.0, pressureExternal=0.0, frameDataToggle=0, frameImageToggle=0, printToggle=0, videoToggle=0)
# # R = reinterpret(SVector{2,Float64}, integ1.u) 
# # params, matrices = integ1.p
# # @unpack A, B = matrices

# # integ = vertexModel(nCycles=1.0, pressureExternal=0.0, frameDataToggle=0, frameImageToggle=0, printToggle=0, videoToggle=0, divisionToggle=0)

# # R = reinterpret(SVector{2,Float64}, integ.u) 
# # params, matrices = integ.p
# # @unpack A, B, cellTensions, cellPressures, F = matrices

# # fileName = datadir("sims", "main", "25-08-07-10-43-27_L₀=0.75_nCells=61_realTimetMax=86400.0_γ=0.2", "frameData", "systemData100.jld2")
# # importedData = load(fileName)
# # R = importedData["R"]
# # matrices = importedData["matrices"]
# # @unpack A, B, F = matrices

#%%

h = hNetwork(R, A, B, F)

fig = Figure(size=(500,500))
ax = Axis(fig[1,1])
scatter!(ax, Point{2,Float64}.(h))
display(fig)

