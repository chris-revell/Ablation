# Julia packages
using DrWatson
using FromFile
using OrdinaryDiffEq
using LinearAlgebra
using JLD2
using SparseArrays
using StaticArrays
using CairoMakie
using Printf
using CairoMakie
using VertexModel
using GeometryBasics
using Random
using Colors
using InvertedIndices
using Dates
using DiscreteCalculus

@from "$(srcdir("AblateCells.jl"))" using AblateCells

function makeCellPolygons(R,params,matrices)
    cellPolygons = Vector{Point{2,Float64}}[]
    for i=1:params.nCells
        push!(cellPolygons,Point{2,Float64}.(R[matrices.cellVertexOrders[i]]))
    end
    return cellPolygons
end
# getRandomColor(seed) = RGB(rand(Xoshiro(seed),3)...)

function division(R, params, matrices, ind)
    # Long and short axis from eigenvectors of shapetensor
    # Put some sort of tolerance that if eigenvalues are approx equal we randomly choose a division orientation, eg circ >0.95
    eigenVals, eigenVecs = eigen(matrices.cellShapeTensor[ind]) # eigenvalues/vectors listed smallest to largest eigval.
    if eigenVecs[:,1][2] < 0.0 #make it so vector is pointing in positive y direction (to fit with existing code in assigning new edges)
        shortvec = -1.0*matrices.cellPerimeters[ind].*eigenVecs[:,1] # Multiplication by cell perimeter ensures this axis is long enough to completely cross the cell; eigenvector has unit length otherwise
    else
        shortvec = matrices.cellPerimeters[ind].*eigenVecs[:,1] # Multiplication by cell perimeter ensures this axis is long enough to completely cross the cell; eigenvector has unit length otherwise
    end
    shortAxisLine = Line(Point{2,Float64}(matrices.cellPositions[ind].+shortvec), Point{2,Float64}(matrices.cellPositions[ind].-shortvec))

    # Test cell edges for an intersection
    edgeLines = Line.(Point{2,Float64}.(R1[matrices.cellVertexOrders[ind][0:end-1]]), Point{2,Float64}.(R1[matrices.cellVertexOrders[ind][1:end]])) # Start and end with the same vertex by indexing circular array from 0 to end
    intersections = [intersects(line, shortAxisLine) for line in edgeLines] #find which edges intersect and where
    intersectedIndices = findall(x->x!=0, first.(intersections))

    intersectedEdges = matrices.cellEdgeOrders[ind][intersectedIndices]

    newCellVertices = matrices.cellVertexOrders[ind][intersectedIndices[1]:intersectedIndices[2]-1] # Not including new vertices to be added later
    oldCellVertices = setdiff(matrices.cellVertexOrders[ind], newCellVertices) # Not including new vertices to be added later
    newCellEdges = matrices.cellEdgeOrders[ind][intersectedIndices[1]+1:intersectedIndices[2]-1] # IntersectedEdges allocated to old cell, not including new edge to be added later
    oldCellEdges = setdiff(matrices.cellEdgeOrders[ind], newCellEdges) # Not including new edge to be added later

    # Labels of new edges, cell, and vertices 
    newEdges = params1.nEdges.+collect(1:3)
    newCell = params1.nCells +1
    newVertices = params1.nVerts.+collect(1:2)

    # Add 1 new row and 3 new columns to B matrix for new cell and 3 new edges
    Btmp = spzeros(Int64,newCell,newEdges[end])
    Btmp[1:params1.nCells ,1:params1.nEdges] .= matrices.B            

    # Add edges to new cell with clockwise orientations
    Btmp[newCell,newCellEdges] .= 1
    # Remove edges from existing cell that have been moved to new cell
    Btmp[ind,newCellEdges] .= 0
    # Add new edge dividing cells to existing cell with anticlockwise orientation
    Btmp[ind,newEdges[1]] = -1
    # Add all new edges to new cell with clockwise orientation
    Btmp[newCell,newEdges] .= 1

    # Find the neighbouring cells that share the intersected edges
    # Add new edges to these neighbour cells
    # These edges are clockwise in the new cell, so must be anticlockwise in these neighbour cells                      
    if matrices.boundaryEdges[intersectedEdges[1]] == 0
        neighbourCell = setdiff(findall(x->x!=0, @view matrices.B[:,intersectedEdges[1]]),[ind])[1]
        Btmp[neighbourCell,newEdges[2]] = -1
    end
    if matrices.boundaryEdges[intersectedEdges[2]] == 0
        neighbourCell = setdiff(findall(x->x!=0, @view matrices.B[:,intersectedEdges[2]]),[ind])[1]
        Btmp[neighbourCell,newEdges[3]] = -1
    end

    # Ensure orientations with respect to other cells of all edges in new cell are correct
    for edge in [newCellEdges...]
        if matrices.boundaryEdges[edge] == 0
            neighbourCell = setdiff(findall(x->x!=0, @view matrices.B[:,edge]),[ind])[1]
            Btmp[neighbourCell,edge] = -1
        end
    end

    # Add 3 new rows and 2 new columns to A matrix for new vertices and edges
    Atmp = spzeros(Int64,newEdges[end],newVertices[end])
    Atmp[1:params1.nEdges,1:params1.nVerts] .= matrices.A

    # First intersected old edge (which remains in the old cell) loses downstream vertex and gains the new vertex
    Atmp[intersectedEdges[1],matrices.cellVertexOrders[ind][intersectedIndices[1]]] = 0
    Atmp[intersectedEdges[1],newVertices[1]] = matrices.A[intersectedEdges[1],matrices.cellVertexOrders[ind][intersectedIndices[1]]]

    # Second intersected old edge (which remains in the old cell) loses upstream vertex and gains the new vertex             
    Atmp[intersectedEdges[2],matrices.cellVertexOrders[ind][intersectedIndices[2]-1]] = 0
    Atmp[intersectedEdges[2],newVertices[2]] = matrices.A[intersectedEdges[2],matrices.cellVertexOrders[ind][intersectedIndices[2]-1]]

    # First new edge gains both new vertices
    # Clockwise orientation of new edge with respect to new cell means the new edge leaves the second new vertex and enters the first new vertex
    Atmp[newEdges[1],newVertices[1]] = 1
    Atmp[newEdges[1],newVertices[2]] = -1

    # Second new edge gains first new vertex upstream (-1) and downstream vertex of old first intersected edge downstream (1)
    Atmp[newEdges[2],newVertices[1]] = -1
    Atmp[newEdges[2],matrices.cellVertexOrders[ind][intersectedIndices[1]]] = 1

    # Third new edge gains second new vertex downstream (1) and upstream vertex of second old intersected edge upstream (-1)
    Atmp[newEdges[3],matrices.cellVertexOrders[ind][intersectedIndices[2]-1]] = -1
    Atmp[newEdges[3],newVertices[2]] = 1

    # Check orientation of old vertices in new cell with respect to old edges allocated to new cell now that those old edges have been made clockwise with respect to new cell
    for (k,edge) in enumerate(newCellEdges)
        Atmp[edge, newCellVertices[k]] = -1
        Atmp[edge, newCellVertices[k+1]] = 1
    end

    Rtmp = copy(R)
    push!(Rtmp, SVector{2,Float64}(intersections[intersectedIndices[1]][2]))
    push!(Rtmp, SVector{2,Float64}(intersections[intersectedIndices[2]][2]))

    return Rtmp, Atmp, Btmp
end

# integ0 = vertexModel(
#     nRows = 15,
#     nCycles = 2,
#     divisionToggle = 1,
#     outputTotal = 1,
#     outputToggle = 0,
#     frameDataToggle = 0,
#     frameImageToggle = 0,
#     printToggle = 1,
#     plotCells = 0,
#     energyModel = "quadratic",
# )

# #%%
# (params0, matrices0) = integ0.p 
# # @unpack A, B = matrices 
# R0 = reinterpret(SVector{2,Float64}, integ0.u)

# integ1 = vertexModel(
#     initialSystem = "argument",
#     nCycles = 1,
#     divisionToggle = 0,
#     outputTotal = 1,
#     outputToggle = 0,
#     frameDataToggle = 0,
#     frameImageToggle = 0,
#     printToggle = 1,
#     plotCells = 0,
#     energyModel = "quadratic",
#     R_in = R0,
#     A_in = matrices0.A, 
#     B_in = matrices0.B,
# )
# (params1, matrices1) = integ1.p 
# # @unpack A, B = matrices1 
# R1 = reinterpret(SVector{2,Float64}, integ1.u)

# dateString = "$(Dates.format(Dates.now(),"yy-mm-dd-HH-MM-SS"))"
dateString = "25-12-02-16-58-52"
# !isdir(datadir("division", dateString)) ? mkpath(datadir("division", dateString)) : nothing 
# jldsave(datadir("division", dateString, "$(dateString)_InitialSystem.jld2"); 
#             integ1,
#         )

data = load("/Users/christopher/Postdoc/Code/Ablation/data/division/$(dateString)/$(dateString)_InitialSystem.jld2")
@unpack integ1 = data
(params1, matrices1) = integ1.p 
R1 = reinterpret(SVector{2,Float64}, integ1.u)

systemCOM1 = sum(R1)./length(R1)

nSamples = 20
# (pos, ind) = findmin(norm.([cell.-systemCOM1 for cell in matrices1.cellPositions]))
inds = rand(findall(x->x==0, findPeripheralCells(matrices1.B)), nSamples)

#%%

for ind in inds

    Rdiv, Adiv, Bdiv = division(R, params, matrices, ind)

    integ2 = vertexModel(
        initialSystem = "argument",
        nCycles = 0.1,
        divisionToggle = 0,
        outputTotal = 1,
        outputToggle = 0,
        frameDataToggle = 0,
        frameImageToggle = 0,
        printToggle = 1,
        plotCells = 0,
        energyModel = "quadratic",
        R_in = Rdiv,
        A_in = Adiv,
        B_in = Bdiv,
    )
    jldsave(datadir("division", dateString, "$(dateString)_Division_Cell$(ind).jld2"); 
                integ2,
            )
    (params2, matrices2) = integ2.p 
    # @unpack A, B = matrices2 
    R2 = reinterpret(SVector{2,Float64}, integ2.u)

    #%%

    ablatedR, ablatedA, ablatedB = ablateCells(R1, matrices1.A, matrices1.B, [ind])
    integ3 = vertexModel(
        initialSystem = "argument",
        nCycles = 0.1,
        divisionToggle = 0,
        outputTotal = 1,
        outputToggle = 0,
        frameDataToggle = 0,
        frameImageToggle = 0,
        printToggle = 1,
        plotCells = 0,
        energyModel = "quadratic",
        R_in = ablatedR,
        A_in = ablatedA,
        B_in = ablatedB,
    )
    jldsave(datadir("division", dateString, "$(dateString)_Ablation_Cell$(ind).jld2"); 
                integ3,
            )
    (params3, matrices3) = integ3.p 
    # @unpack A, B = matrices3 
    R3 = reinterpret(SVector{2,Float64}, integ3.u)

    radiusVectors = [cell.-matrices1.cellPositions[ind] for cell in matrices1.cellPositions]
    radii = norm.(radiusVectors)
    
    cellDisplacements_Division = [matrices2.cellPositions[i].-matrices1.cellPositions[i] for i=1:params1.nCells]
    cellDisplacementNorms_Division = norm.(cellDisplacements_Division)
    arrowColours_Division = normalize.(radiusVectors).⋅normalize.(cellDisplacements_Division)

    cellDisplacements_Ablation = matrices3.cellPositions.-matrices1.cellPositions[Not(ind)]
    cellDisplacementNorms_Ablation = norm.(cellDisplacements_Ablation)
    arrowColours_Ablation = normalize.(radiusVectors).⋅normalize.(cellDisplacements_Ablation)
    
    # Δr = f(r)(a+bcos(2θ-θ₀))
    # Δr̄ = (a+bcos(2θ-θ₀))


#%%







fig = Figure(size=(1500,1500))
axes = Axis[]
push!(axes, Axis(fig, aspect=DataAspect()))
cellPolygons1 = makeCellPolygons(R1, params1, matrices1)
for i in 1:params1.nCells
    if i!=ind
        poly!(axes[end], cellPolygons1[i], color=(:green, 0.25), strokecolor=(:black, 1.0), strokewidth=2)
    else 
        poly!(axes[end], cellPolygons1[i], color=(:black, 0.5), strokecolor=(:black, 1.0), strokewidth=2)
    end
end
hidespines!(axes[end])
hidedecorations!(axes[end])

push!(axes, Axis(fig, aspect=DataAspect()))
cellPolygons2 = makeCellPolygons(R2, params2, matrices2)
for i = 1:params2.nCells
    if i!=ind && i!= params2.nCells
        poly!(axes[end], cellPolygons2[i], color=(:green, 0.25), strokecolor=(:black, 1.0), strokewidth=2)
    else 
        poly!(axes[end], cellPolygons2[i], color=(:black, 0.5), strokecolor=(:black, 1.0), strokewidth=2)
    end
end
hidespines!(axes[end])
hidedecorations!(axes[end])

push!(axes, Axis(fig))
scatter!(axes[end], log10.(radii[Not(ind)]), log10.(cellDisplacementNorms[Not(ind)]), color=arrowColours[Not(ind)], colorrange=(-1.0,1.0), colormap=:bwr)
xs = log10.(norm.(radiusVectors[Not(ind)]))
ys = -1.5.-1.0.*xs
# lines!(axes[end], xs, ys, color=:blue)
ys = -1.25.-1.0.*xs
# lines!(axes[end], xs, ys, color=:red)

push!(axes, Axis(fig, aspect=DataAspect()))
for i = 1:params2.nCells
    poly!(axes[end], cellPolygons2[i], color=(:white,0.0), strokecolor=(:black, 0.1), strokewidth=1)
end
# arrows2d!(axes[end], Point{2,Float64}.(matrices1.cellPositions[Not(ind)]), Vec{2,Float64}.(cellDisplacements[Not(ind)]), color=arrowColours, lengthscale=20.0)
arrows!(axes[end], Point{2,Float64}.(matrices1.cellPositions[Not(ind)]), Vec{2,Float64}.(cellDisplacements[Not(ind)]), color=arrowColours[Not(ind)], colorrange=(-1.0,1.0), colormap=:bwr, lengthscale=20.0, linewidth=5)
scatter!(axes[end], Point{2,Float64}(systemCOM1))
lines!(axes[end], Point{2,Float64}.([matrices1.cellPositions[ind].-shortvec.*2, matrices1.cellPositions[ind].+shortvec.*2]), color=(:black,0.5), linewidth=4)
hidedecorations!(axes[end])
hidespines!(axes[end])


push!(axes, Axis(fig, aspect=DataAspect()))
cellPolygons3 = makeCellPolygons(R3, params3, matrices3)
for i = 1:params3.nCells
    poly!(axes[end], cellPolygons3[i], color=(:green, 0.25), strokecolor=(:black, 1.0), strokewidth=2)
end
hidespines!(axes[end])
hidedecorations!(axes[end])

arrowColours = normalize.(radiusVectors[Not(ind)]).⋅normalize.(cellDisplacements)
clims = (-1.0, 1.0)

push!(axes, Axis(fig))
scatter!(axes[end], log10.(radii[Not(ind)]), log10.(cellDisplacementNorms), color=arrowColours, colorrange=(-1.0,1.0), colormap=:bwr)
xs = log10.(norm.(radiusVectors[Not(ind)]))
ys = -1.5.-1.0.*xs
# lines!(axes[end], xs, ys, color=:blue)
ys = -1.25.-1.0.*xs
# lines!(axes[end], xs, ys, color=:red)

push!(axes, Axis(fig, aspect=DataAspect()))
for i = 1:params3.nCells
    poly!(axes[end], cellPolygons3[i], color=(:white,0.0), strokecolor=(:black, 0.1), strokewidth=1)
end
arrows!(axes[end], Point{2,Float64}.(matrices3.cellPositions), Vec{2,Float64}.(cellDisplacements), color=arrowColours, colorrange=(-1.0,1.0), colormap=:bwr, lengthscale=20.0, linewidth=5)
scatter!(axes[end], Point{2,Float64}(systemCOM1))
# lines!(axes[end], Point{2,Float64}.([matrices3.cellPositions[ind].-shortvec.*2, matrices3.cellPositions[ind].+shortvec.*2]), color=(:black,0.5), linewidth=4)
hidedecorations!(axes[end])
hidespines!(axes[end])


subfigureOrdering = CartesianIndex.([(1,1), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)])
for (n,i) in enumerate(subfigureOrdering)
    fig[i[1], i[2]] = axes[n]
end
# fig[1,1] = axes[1]
# fig[2,1] = axes[2]
# fig[2,2] = axes[3]
# fig[2,3] = axes[4]
# fig[3,1] = axes[5]
# fig[3,2] = axes[6]
# fig[3,3] = axes[7]

display(fig)

!isdir(plotsdir("division")) ? mkpath(plotsdir("division")) : nothing 
save(plotsdir("division", "$(Dates.format(Dates.now(),"yy-mm-dd-HH-MM-SS"))_division.png"), fig)
display(fig)