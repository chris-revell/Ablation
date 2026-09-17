#
#  AblateCells.jl
#  Ablation 
#
#

module DivideCell

# Julia packages
using InvertedIndices
using SparseArrays
using StaticArrays
using GeometryBasics
using LinearAlgebra
using DiscreteCalculus
using CircularArrays

# Local modules
# @from "$(srcdir("X.jl"))" using X

function divideCell(R, A, B, iDiv)

    I, J = size(B)
    K = size(A, 2)

    cellPerimeters = findCellPerimeterLengths(R, A, B)
    cellPositions = findCellCentresOfMass(R, A, B)
    cellVertexOrders = fill(CircularVector(Int64[]), I)
    cellEdgeOrders = fill(CircularVector(Int64[]), I)
    for i = 1:length(cellVertexOrders)
        (cellVertexOrders[i], cellEdgeOrders[i]) = orderAroundCell(A, B, i)
    end
    cellEdgeCount = size.(cellVertexOrders)
    cellShapeTensor = fill(SMatrix{2,2,Float64}(zeros(2,2)), I)
    # Find cell areas and shape tensors 
    for i = 1:I
        Rα = [R[kk].-cellPositions[i] for kk in cellVertexOrders[i]]
        cellShapeTensor[i] = sum(Rα.*transpose.(Rα))./cellEdgeCount[i]
    end
    jᵖ = findPeripheralEdges(B) # Peripheral edges

    
    # Long and short axis from eigenvectors of shapetensor
    # Put some sort of tolerance that if eigenvalues are approx equal we randomly choose a division orientation, eg circ >0.95
    eigenVals, eigenVecs = eigen(cellShapeTensor[iDiv]) # eigenvalues/vectors listed smallest to largest eigval.
    if eigenVecs[:,1][2] < 0.0 #make it so vector is pointing in positive y direction (to fit with existing code in assigning new edges)
        shortvec = -1.0*cellPerimeters[iDiv].*eigenVecs[:,1] # Multiplication by cell perimeter ensures this axis is long enough to completely cross the cell; eigenvector has unit length otherwise
    else
        shortvec = cellPerimeters[iDiv].*eigenVecs[:,1] # Multiplication by cell perimeter ensures this axis is long enough to completely cross the cell; eigenvector has unit length otherwise
    end
    shortAxisLine = Line(Point{2,Float64}(cellPositions[iDiv].+shortvec), Point{2,Float64}(cellPositions[iDiv].-shortvec))

    # Test cell edges for an intersection
    edgeLines = Line.(Point{2,Float64}.(R[cellVertexOrders[iDiv][0:end-1]]), Point{2,Float64}.(R[cellVertexOrders[iDiv][1:end]])) # Start and end with the same vertex by indexing circular array from 0 to end
    intersections = [intersects(line, shortAxisLine) for line in edgeLines] #find which edges intersect and where
    intersectedIndices = findall(x->x!=0, first.(intersections))

    intersectedEdges = cellEdgeOrders[iDiv][intersectedIndices]

    newCellVertices = cellVertexOrders[iDiv][intersectedIndices[1]:intersectedIndices[2]-1] # Not including new vertices to be added later
    oldCellVertices = setdiff(cellVertexOrders[iDiv], newCellVertices) # Not including new vertices to be added later
    newCellEdges = cellEdgeOrders[iDiv][intersectedIndices[1]+1:intersectedIndices[2]-1] # IntersectedEdges allocated to old cell, not including new edge to be added later
    oldCellEdges = setdiff(cellEdgeOrders[iDiv], newCellEdges) # Not including new edge to be added later

    # Labels of new edges, cell, and vertices 
    newEdges = J.+collect(1:3)
    newCell = I+1
    newVertices = K.+collect(1:2)

    # Add 1 new row and 3 new columns to B matrix for new cell and 3 new edges
    Btmp = spzeros(Int64,newCell,newEdges[end])
    Btmp[1:I ,1:J] .= B            

    # Add edges to new cell with clockwise orientations
    Btmp[newCell,newCellEdges] .= 1
    # Remove edges from existing cell that have been moved to new cell
    Btmp[iDiv,newCellEdges] .= 0
    # Add new edge dividing cells to existing cell with anticlockwise orientation
    Btmp[iDiv,newEdges[1]] = -1
    # Add all new edges to new cell with clockwise orientation
    Btmp[newCell,newEdges] .= 1

    # Find the neighbouring cells that share the intersected edges
    # Add new edges to these neighbour cells
    # These edges are clockwise in the new cell, so must be anticlockwise in these neighbour cells                      
    if jᵖ[intersectedEdges[1]] == 0
        neighbourCell = setdiff(findall(x->x!=0, @view B[:,intersectedEdges[1]]),[iDiv])[1]
        Btmp[neighbourCell,newEdges[2]] = -1
    end
    if jᵖ[intersectedEdges[2]] == 0
        neighbourCell = setdiff(findall(x->x!=0, @view B[:,intersectedEdges[2]]),[iDiv])[1]
        Btmp[neighbourCell,newEdges[3]] = -1
    end

    # Ensure orientations with respect to other cells of all edges in new cell are correct
    for edge in [newCellEdges...]
        if jᵖ[edge] == 0
            neighbourCell = setdiff(findall(x->x!=0, @view B[:,edge]),[iDiv])[1]
            Btmp[neighbourCell,edge] = -1
        end
    end

    # Add 3 new rows and 2 new columns to A matrix for new vertices and edges
    Atmp = spzeros(Int64,newEdges[end],newVertices[end])
    Atmp[1:J,1:K] .= A

    # First intersected old edge (which remains in the old cell) loses downstream vertex and gains the new vertex
    Atmp[intersectedEdges[1],cellVertexOrders[iDiv][intersectedIndices[1]]] = 0
    Atmp[intersectedEdges[1],newVertices[1]] = A[intersectedEdges[1],cellVertexOrders[iDiv][intersectedIndices[1]]]

    # Second intersected old edge (which remains in the old cell) loses upstream vertex and gains the new vertex             
    Atmp[intersectedEdges[2],cellVertexOrders[iDiv][intersectedIndices[2]-1]] = 0
    Atmp[intersectedEdges[2],newVertices[2]] = A[intersectedEdges[2],cellVertexOrders[iDiv][intersectedIndices[2]-1]]

    # First new edge gains both new vertices
    # Clockwise orientation of new edge with respect to new cell means the new edge leaves the second new vertex and enters the first new vertex
    Atmp[newEdges[1],newVertices[1]] = 1
    Atmp[newEdges[1],newVertices[2]] = -1

    # Second new edge gains first new vertex upstream (-1) and downstream vertex of old first intersected edge downstream (1)
    Atmp[newEdges[2],newVertices[1]] = -1
    Atmp[newEdges[2],cellVertexOrders[iDiv][intersectedIndices[1]]] = 1

    # Third new edge gains second new vertex downstream (1) and upstream vertex of second old intersected edge upstream (-1)
    Atmp[newEdges[3],cellVertexOrders[iDiv][intersectedIndices[2]-1]] = -1
    Atmp[newEdges[3],newVertices[2]] = 1

    # Check orientation of old vertices in new cell with respect to old edges allocated to new cell now that those old edges have been made clockwise with respect to new cell
    for (k,edge) in enumerate(newCellEdges)
        Atmp[edge, newCellVertices[k]] = -1
        Atmp[edge, newCellVertices[k+1]] = 1
    end

    dropzeros!(Atmp)
    dropzeros!(Btmp)

    Rtmp = copy(R)
    push!(Rtmp, SVector{2,Float64}(intersections[intersectedIndices[1]][2]))
    push!(Rtmp, SVector{2,Float64}(intersections[intersectedIndices[2]][2]))

    return Rtmp, Atmp, Btmp, shortvec 
end 

export divideCell

end #end module 
