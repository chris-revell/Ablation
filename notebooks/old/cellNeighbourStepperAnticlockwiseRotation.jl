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

# fileName = datadir("25-04-25-10-54-48_testSystem.jld2")
# importedData = load(fileName)
# @unpack R, A, B = importedData

integ = vertexModel(nRows=11, nCycles=1.0)
R = reinterpret(SVector{2,Float64}, integ.u) 
params, matrices = integ.p
@unpack A, B = matrices

#%%

matrices = (R, A, B)
nCells = size(B,1)
nEdges = size(B,2)
nVerts = size(A,2)

cellCentres = findCellCentresOfMass(matrices...)
cellPolygons = findCellPolygons(matrices...)
edgeQuadrilaterals = findEdgeQuadrilaterals(matrices...)
edgeMidpoints = findEdgeMidpoints(matrices[1:2]...)
cellAreas = findCellAreas(R, A, B)
cellPerimeterLengths = findCellPerimeterLengths(R, A, B)
edgeTangents = findEdgeTangents(R, A)
edgeLengths = findEdgeLengths(R, A)
cellTensions = 0.2*0.75 .* log.(cellPerimeterLengths ./ 0.75)
cellPressures = log.(cellAreas)
ϵ = SMatrix{2, 2, Float64}([
        0.0 1.0
        -1.0 0.0
    ])
Ā = abs.(A)
B̄ = abs.(B)

F = spzeros(SVector{2,Float64}, nVerts, nCells)
fill!(F, @SVector zeros(2))
for k = 1:nVerts
    for j in nzrange(A, k)
        for i in nzrange(B, rowvals(A)[j])
            # Force components from cell pressure perpendicular to edge tangents 
            F[k, rowvals(B)[i]] += 0.5 * cellPressures[rowvals(B)[i]] * B[rowvals(B)[i], rowvals(A)[j]] * Ā[rowvals(A)[j], k] .* (ϵ * edgeTangents[rowvals(A)[j]])
            # Force components from cell membrane tension parallel to edge tangents 
            F[k, rowvals(B)[i]] -= cellTensions[rowvals(B)[i]] * B̄[rowvals(B)[i], rowvals(A)[j]] * A[rowvals(A)[j], k] .* edgeTangents[rowvals(A)[j]] ./ edgeLengths[rowvals(A)[j]]           
        end
    end
end

#%%

fig = Figure(size=(500,500))
ax = Axis(fig[1,1], aspect=DataAspect())
for i=1:size(B,1)
    poly!(ax, cellPolygons[i], color=(:black,0.5), strokecolor=(:black, 0.25), strokewidth=1)
end
scatter!(ax, Point{2,Float64}.(R), color=:blue)
annotations!(ax, string.(collect(1:nVerts)), Point{2,Float64}.(R), color=:blue)
scatter!(ax, Point{2,Float64}.(edgeMidpoints), color=:green)
annotations!(ax, string.(collect(1:nEdges)), Point{2,Float64}.(edgeMidpoints), color=:green)
scatter!(ax, Point{2,Float64}.(cellCentres), color=:red)
annotations!(ax, string.(collect(1:nCells)), Point{2,Float64}.(cellCentres), color=:red)
hidedecorations!(ax)
hidespines!(ax)

# ax2 = Axis(fig[2,1])
# text!(ax2, 0, 0, text="test")
display(fig)

#%%
fig = Figure(size=(500,500))
ax = Axis(fig[1,1], aspect=DataAspect())
for i=1:size(B,1)
    poly!(ax, cellPolygons[i], color=(:black,0.5), strokecolor=(:black, 0.5), strokewidth=1)
end
reset_limits!(ax)
mov = VideoStream(fig, framerate=20)
recordframe!(mov)


boundaryEdges = findBoundaryEdges(B)
boundaryCells = findnz(B[:, boundaryEdges.==1])[1]
currentCell = [rand(boundaryCells)]

# Clear everything from the axis 
empty!(ax)
# Loop over all cells 
for i=1:size(B,1)
    if i==currentCell[1]
        poly!(ax, cellPolygons[i], color=(:red, 0.5), strokecolor=(:black, 0.5), strokewidth=1)
    else
        poly!(ax, cellPolygons[i], color=(:black, 0.5), strokecolor=(:black, 0.5), strokewidth=1)
    end
end
scatter!(ax, Point{2,Float64}.(cellCentres), color=:red)
annotations!(ax, string.(collect(1:nCells)), Point{2,Float64}.(cellCentres), color=:red)    
recordframe!(mov)

testedCells = Int64[]
currentCellNeighbours = cellNeighbourOrder(A, B, currentCell[1])
push!(testedCells, currentCell[1])
currentCell[1] = currentCellNeighbours[1]

empty!(ax)
# Loop over all cells 
for i=1:size(B,1)
    if i∈testedCells
        poly!(ax, cellPolygons[i], color=(:red, 0.5), strokecolor=(:black, 0.5), strokewidth=1)
    else
        poly!(ax, cellPolygons[i], color=(:black, 0.5), strokecolor=(:black, 0.5), strokewidth=1)
    end
end
scatter!(ax, Point{2,Float64}.(cellCentres), color=:red)
annotations!(ax, string.(collect(1:nCells)), Point{2,Float64}.(cellCentres), color=:red)
recordframe!(mov)

# Loop over all output timepoints 
while length(testedCells)<nCells-1
    currentCellNeighbours = cellNeighbourOrder(A, B, currentCell[1])
    lastCellIndex = findall(x->x==testedCells[end], currentCellNeighbours)[1]
    push!(testedCells, currentCell[1])
    # @show currentCellNeighbours[lastCellIndex+1:lastCellIndex+length(currentCellNeighbours)]
    nextCellIndex = findlast(x->x∉testedCells, currentCellNeighbours[lastCellIndex+1:lastCellIndex+length(currentCellNeighbours)])
    typeof(nextCellIndex) == Nothing ? break : nothing
    currentCell[1] = currentCellNeighbours[lastCellIndex+1:lastCellIndex+length(currentCellNeighbours)][nextCellIndex]
    
    # Clear everything from the axis 
    empty!(ax)

    # Loop over all cells 
    for i=1:size(B,1)
        if i∈testedCells
            poly!(ax, cellPolygons[i], color=(:red, 0.5), strokecolor=(:black, 0.5), strokewidth=1)
        else
            poly!(ax, cellPolygons[i], color=(:black, 0.5), strokecolor=(:black, 0.5), strokewidth=1)
        end
    end
    scatter!(ax, Point{2,Float64}.(cellCentres), color=:red)
    annotations!(ax, string.(collect(1:nCells)), Point{2,Float64}.(cellCentres), color=:red)
    
    recordframe!(mov)
end

empty!(ax)
# Loop over all cells 
for i=1:size(B,1)
    if i∈testedCells
        poly!(ax, cellPolygons[i], color=(:red, 0.5), strokecolor=(:black, 0.5), strokewidth=1)
    else
        poly!(ax, cellPolygons[i], color=(:black, 0.5), strokecolor=(:black, 0.5), strokewidth=1)
    end
end
scatter!(ax, Point{2,Float64}.(cellCentres), color=:red)
annotations!(ax, string.(collect(1:nCells)), Point{2,Float64}.(cellCentres), color=:red)

recordframe!(mov)
save(datadir("output.mp4"), mov)

#%%
