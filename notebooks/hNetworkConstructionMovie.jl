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

#%%

function hAroundCell!(ii, fig, ax1, ax2, cellPolygons, currentNeighbourShell, traversedCells, B, cellEdgeOrders, cellVertexOrders, h, ϵ, F, traversedEdges)
    # Clear everything from the axis 
    empty!(ax1)
    # Loop over all cells 
    for i=1:size(B,1)
        if i==ii
            poly!(ax1, cellPolygons[i], color=(:red, 0.5), strokecolor=(:black, 1.0), strokewidth=2)
        elseif i ∈ currentNeighbourShell && i ∈ traversedCells 
            poly!(ax1, cellPolygons[i], color=(:red, 0.5), strokecolor=(:black, 1.0), strokewidth=2)
        elseif i ∈ currentNeighbourShell
            poly!(ax1, cellPolygons[i], color=(:black, 0.5), strokecolor=(:black, 1.0), strokewidth=2)
        elseif i ∈ traversedCells
            poly!(ax1, cellPolygons[i], color=(:red, 0.5), strokecolor=(:black, 0.5), strokewidth=1)
        else   
            poly!(ax1, cellPolygons[i], color=(:black, 0.5), strokecolor=(:black, 0.5), strokewidth=1)
        end
    end 
    # Find an edge belonging to cell ii and shared by any other cell in traversedCells
    if length(traversedCells)>0
        startEdge = (findnz(B[traversedCells, :])[2] ∩ findnz(B[ii, :])[1])[1]
    else 
        startEdge = cellEdgeOrders[ii][1]
    end
    # Find the index of this edge within the ordering of edges around cell ii 
    startInd = findall(x->x==startEdge, cellEdgeOrders[ii])[1]
    # Remember in clockwise ordering, cellEdgeOrders[i][1] precedes cellVertexOrders[1]
    for vertexInd = startInd:(startInd+length(cellVertexOrders[ii])-1)
        h[cellEdgeOrders[ii][vertexInd+1]] = h[cellEdgeOrders[ii][vertexInd]] .+ ϵ*F[cellVertexOrders[ii][vertexInd], ii]
        arrows!(ax2, [Point{2,Float64}(h[cellEdgeOrders[ii][vertexInd]])], [Vec{2,Float64}(ϵ*F[cellVertexOrders[ii][vertexInd], ii])], color=:black)
        push!(traversedEdges, cellEdgeOrders[ii][vertexInd]) 
        reset_limits!(ax2)   
        recordframe!(mov)
    end
    return nothing 
end

#%%

# integ1 = vertexModel(nRows=19, nCycles=1.0, pressureExternal=0.0, frameDataToggle=0, frameImageToggle=0, printToggle=0, videoToggle=0)
# R = reinterpret(SVector{2,Float64}, integ1.u) 
# params, matrices = integ1.p
# @unpack A, B = matrices
# integ = vertexModel(initialSystem = "argument", R_in = R, A_in = matrices.A, B_in = matrices.B, nCycles=1.0, pressureExternal=0.0, frameDataToggle=0, frameImageToggle=0, printToggle=0, videoToggle=0, divisionToggle=0)

# integ = vertexModel(nRows=9, nCycles=1.0, pressureExternal=0.0, frameDataToggle=0, frameImageToggle=0, printToggle=0, videoToggle=0, divisionToggle=0)
# R = reinterpret(SVector{2,Float64}, integ.u) 
# params, matrices = integ.p
# @unpack A, B, cellTensions, cellPressures, F = matrices


fileName = datadir("referenceSystems", "smallerSystem.jld2")
importedData = load(fileName)
R = importedData["R"]
A = importedData["A"]
B = importedData["B"]
F = importedData["F"]

# Import system data
# fileName = datadir("DoubleHole_testSystem.jld2")
# importedData = load(fileName)
# R = importedData["ablatedGrownR"]
# A = importedData["ablatedGrownA"]
# B = importedData["ablatedGrownB"]
# F = importedData["ablatedGrownF"]
# cellTensions = importedData["ablatedGrownCellTensions"]
# cellPressures = importedData["ablatedGrownCellPressures"]
# cellPerimeters = importedData["ablatedGrownCellPerimeters"]
# cellAreas = importedData["ablatedGrownCellAreas"]
# cellEffectivePressures = cellPressures .+ cellTensions.*cellPerimeters./(2.0.*cellAreas)



#%%

matrices = (R, A, B)
nCells = size(B,1)
nEdges = size(B,2)
nVerts = size(A,2)
cellCentres = findCellCentresOfMass(R, A, B)
cellPolygons = findCellPolygons(R, A, B)
edgeQuadrilaterals = findEdgeQuadrilaterals(R, A, B)
edgeMidpoints = findEdgeMidpoints(R, A)
cellAreas = findCellAreas(R, A, B)
cellPerimeterLengths = findCellPerimeterLengths(R, A, B)
edgeTangents = findEdgeTangents(R, A)
edgeLengths = findEdgeLengths(R, A)
boundaryEdges = findPeripheralEdges(B)
boundaryCells = findnz(B[:, boundaryEdges.==1])[1]
cellVertexOrders  = fill(CircularVector(Int64[]), nCells)
cellEdgeOrders    = fill(CircularVector(Int64[]), nCells)
for i = 1:length(cellVertexOrders)
    cellVertexOrders[i], cellEdgeOrders[i] = orderAroundCell(A, B, i)
end
ϵ = SMatrix{2, 2, Float64}([
        0.0 1.0
        -1.0 0.0
    ])
Ā = abs.(A)
B̄ = abs.(B)

# F = spzeros(SVector{2,Float64}, nVerts, nCells)
# fill!(F, @SVector zeros(2))
# for k = 1:nVerts
#     for j in nzrange(A, k)
#         for i in nzrange(B, rowvals(A)[j])
#             # Force components from cell pressure perpendicular to edge tangents 
#             F[k, rowvals(B)[i]] += 0.5 * cellPressures[rowvals(B)[i]] * B[rowvals(B)[i], rowvals(A)[j]] * Ā[rowvals(A)[j], k] .* (ϵ * edgeTangents[rowvals(A)[j]])
#             # Force components from cell membrane tension parallel to edge tangents 
#             F[k, rowvals(B)[i]] -= cellTensions[rowvals(B)[i]] * B̄[rowvals(B)[i], rowvals(A)[j]] * A[rowvals(A)[j], k] .* edgeTangents[rowvals(A)[j]] ./ edgeLengths[rowvals(A)[j]]           
#         end
#     end
# end

#%%

fig = Figure(size=(500,1000))
ax1 = Axis(fig[1,1], aspect=DataAspect())
ax2 = Axis(fig[2,1], aspect=DataAspect())
for i=1:size(B,1)
    poly!(ax1, cellPolygons[i], color=(:black,0.5), strokecolor=(:black, 0.5), strokewidth=1)
end
reset_limits!(ax1)
hidedecorations!(ax1)
hidedecorations!(ax2)
hidespines!(ax1)
hidespines!(ax2)
mov = VideoStream(fig, framerate=10)
recordframe!(mov)

# Ensure we don't start with a boundary cell
startCell = rand(collect(1:nCells)[Not(boundaryCells)])
traversedCells = Int64[]
traversedEdges = Int64[]
neighbourMatrix = B*transpose(B)

h = fill(SVector{2, Float64}(zeros(2)), nEdges)

hAroundCell!(startCell, fig, ax1, ax2, cellPolygons, Int64[], traversedCells, B, cellEdgeOrders, cellVertexOrders, h, ϵ, F, traversedEdges)
push!(traversedCells, startCell)
while length(traversedCells) < nCells
    currentNeighbourShell = setdiff(findnz(neighbourMatrix[:, traversedCells])[1], traversedCells)
    for ii in currentNeighbourShell
        hAroundCell!(ii, fig, ax1, ax2, cellPolygons, currentNeighbourShell, traversedCells, B, cellEdgeOrders, cellVertexOrders, h, ϵ, F, traversedEdges)
        push!(traversedCells, ii)
    end
end
save(datadir("$(Dates.format(Dates.now(),"yy-mm-dd-HH-MM-SS"))_hNetwork.mp4"), mov)


empty!(ax1)       
# Loop over all cells 
for i=1:size(B,1)
    poly!(ax1, cellPolygons[i], color=(:red, 0.5), strokecolor=(:black, 0.5), strokewidth=1)
end
scatter!(ax1, Point{2,Float64}.(cellCentres), color=:red)
annotations!(ax1, string.(collect(1:nCells)), Point{2,Float64}.(cellCentres), color=:red)    
scatter!(ax1, Point{2,Float64}.(edgeMidpoints), color=:green)
# annotations!(ax1, string.(collect(1:nEdges)), Point{2,Float64}.(edgeMidpoints), color=:green)    
scatter!(ax1, Point{2,Float64}.(R), color=:blue)
annotations!(ax1, string.(collect(1:nVerts)), Point{2,Float64}.(R), color=:blue)   
# scatter!(ax2, Point{2,Float64}.(h), color=:green)
# annotations!(ax2, string.(collect(1:length(h))), Point{2,Float64}.(h), color=:green)   
save(datadir("$(Dates.format(Dates.now(),"yy-mm-dd-HH-MM-SS"))_hNetwork.png"), fig)



