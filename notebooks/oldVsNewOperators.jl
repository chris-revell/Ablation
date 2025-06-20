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

# integ1 = vertexModel(nRows=11, nCycles=1.0, pressureExternal=0.0, frameDataToggle=0, frameImageToggle=0, printToggle=0, videoToggle=0)
# R = reinterpret(SVector{2,Float64}, integ1.u) 
# params, matrices = integ1.p
# # integ = vertexModel(initialSystem="argument", R_in=R, A_in=matrices.A, B_in=matrices.B, nCycles=0.1, pressureExternal=0.0, frameDataToggle=0, frameImageToggle=0, printToggle=0, videoToggle=0, divisionToggle=0)
# # R = reinterpret(SVector{2,Float64}, integ.u) 
# # params, matrices = integ.p
# @unpack A, B, F, cellPressures, cellTensions, cellPerimeters, cellAreas = matrices

# Import system data
conditionsDict    = load(datadir("oldPaper", "dataFinal.jld2"))
@unpack nVerts,nCells,nEdges,pressureExternal,γ,λ,viscousTimeScale,realTimetMax,tMax,dt,outputInterval,outputTotal,realCycleTime,t1Threshold = conditionsDict["params"]
matricesDict = load(datadir("oldPaper", "matricesFinal.jld2"))
@unpack A,B,C,R,F,edgeTangents,edgeMidpoints,cellPositions,ϵ,cellAreas,externalF,boundaryVertices,cellPressures,edgeLengths,cellTensions, cellPerimeters = matricesDict["matrices"]

dropzeros!(A)
dropzeros!(B)
dropzeros!(C)


# {divᶜb}ᵢ
# Calculate div on each cell
function calculateCellDivs(nCells, R, B, C, F, cellCentresOfMass, edgeMidpoints, edgeTangents, cellAreas, ϵᵢ)
    cellDivs = Float64[]
    for c = 1:nCells
        cellVertices = findall(x -> x != 0, C[c, :])
        vertexAngles = zeros(size(cellVertices))
        for (k, v) in enumerate(cellVertices)
            vertexAngles[k] = atan((R[v] .- cellCentresOfMass[c])...)
        end
        m = minimum(vertexAngles)
        vertexAngles .-= m
        cellVertices .= cellVertices[sortperm(vertexAngles)]
        cellEdges = findall(x -> x != 0, B[c, :])
        edgeAngles = zeros(size(cellEdges))
        for (k, e) in enumerate(cellEdges)
            edgeAngles[k] = atan((edgeMidpoints[e] .- cellCentresOfMass[c])...)
        end
        edgeAngles .+= (2π - m)
        edgeAngles .= edgeAngles .% (2π)
        cellEdges .= cellEdges[sortperm(edgeAngles)]
        h = @SVector [0.0, 0.0]
        divSum = 0
        for (i, e) in enumerate(cellEdges)
            h = h + ϵᵢ * F[cellVertices[i], c]
            divSum -= B[c, e] * (h ⋅ (ϵᵢ * edgeTangents[e])) / cellAreas[c]
        end
        # divSum *= (-0.5)
        push!(cellDivs, divSum)
    end
    return cellDivs
end

# {divᵛb}ₖ
# Calculate div at each vertex
function calculateVertexDivs(nVerts, R, C, cellCentresOfMass, F, ϵᵢ, q, linkTriangleAreas)
    vertexDivs = Float64[]
    for k = 1:nVerts
        divSum = 0
        vertexCells = findall(x -> x != 0, C[:, k])
        cellAngles = zeros(length(vertexCells))
        for i = 1:length(cellAngles)
            cellAngles[i] = atan((cellCentresOfMass[vertexCells[i]] .- R[k])...)
        end
        vertexCells .= vertexCells[sortperm(cellAngles, rev=true)]
        for i in vertexCells
            divSum += ((ϵᵢ * q[i, k]) ⋅ (ϵᵢ * F[k, i])) / linkTriangleAreas[k]
        end
        push!(vertexDivs, divSum)
    end
    return vertexDivs
end

# {CURLᵛb}ₖ
# Calculate curl at each vertex
function calculateVertexCurls(nVerts, R, C, cellCentresOfMass, F, ϵᵢ, q, linkTriangleAreas)
    vertexCurls = Float64[]
    # Working around a given vertex, an h force space point from a cell is mapped to the next edge anticlockwise from the cell
    for k = 1:nVerts
        curlSum = 0
        vertexCells = findall(x -> x != 0, C[:, k])
        cellAngles = zeros(length(vertexCells))
        for i = 1:length(cellAngles)
            cellAngles[i] = atan((cellCentresOfMass[vertexCells[i]] .- R[k])...)
        end
        vertexCells .= vertexCells[sortperm(cellAngles, rev=true)]
        for i in vertexCells
            curlSum += (q[i, k] ⋅ (ϵᵢ * F[k, i])) / linkTriangleAreas[k]
        end
        push!(vertexCurls, curlSum)
    end
    return vertexCurls
end

cellPolygons = findCellPolygons(R, A, B)
cellCentresOfMass = findCellCentresOfMass(R, A, B)
edgeMidpoints = findEdgeMidpoints(R, A)
edgeTangents = findEdgeTangents(R, A)
cellAreas = findCellAreas(R, A, B)
ϵᵢ = @SMatrix [  # Clockwise rotation matrix setting orientation of cell faces
               0.0 1.0
               -1.0 0.0
           ]
q = findSpokes(R, A, B)
linkTriangles = findCellLinkTriangles(R, A, B)
linkTriangleAreas = findCellLinkTriangleAreas(R, A, B)
𝐡 = hNetwork(R, A, B, F)

# {divᶜb}ᵢ
oldCellDivs = calculateCellDivs(nCells, R, B, C, F, cellCentresOfMass, edgeMidpoints, edgeTangents, cellAreas, ϵᵢ)
oldCellDivLims = (-maximum(abs.(oldCellDivs)),maximum(abs.(oldCellDivs)))
# -divᶜ => cocurlᶜ
cocurlᶜh = cocurlᶜ(R, A, B, 𝐡)
cocurlᶜhLims = (-maximum(abs.(cocurlᶜh)),maximum(abs.(cocurlᶜh)))
difCellDivs = (oldCellDivs .- (-1.0.*cocurlᶜh))
maxdifCellDivs = maximum(abs.(difCellDivs))
difCellDivLims = (-maxdifCellDivs, maxdifCellDivs)
# difCellDivLims = (-0.001, 0.001)# (-maxdifCellDivs, maxdifCellDivs)
@show maxdifCellDivs

# {divᵛb}ₖ
oldVertexDivs = calculateVertexDivs(nVerts, R, C, cellCentresOfMass, F, ϵᵢ, q, linkTriangleAreas)
oldVertexDivLims = (-maximum(abs.(oldVertexDivs)),maximum(abs.(oldVertexDivs)))
# -divᵛ => cocurlᵛ
cocurlᵛh = cocurlᵛ(R, A, B, 𝐡)
cocurlᵛhLims = (-maximum(abs.(cocurlᵛh)),maximum(abs.(cocurlᵛh)))
difVertexDivs = (oldVertexDivs .- (-1.0.*cocurlᵛh))
maxdifVertexDivs = maximum(abs.(difVertexDivs))
difVertexDivLims =(-maxdifVertexDivs, maxdifVertexDivs)
# difVertexDivLims = (-0.001, 0.001)#(-maxdifVertexDivs, maxdifVertexDivs)
@show maxdifVertexDivs

# {CURLᵛb}ₖ
oldVertexCurls = calculateVertexCurls(nVerts, R, C, cellCentresOfMass, F, ϵᵢ, q, linkTriangleAreas)
oldVertexCurlLims = (-maximum(abs.(oldVertexCurls)),maximum(abs.(oldVertexCurls)))
# CURLᵛ => -curlᵛ
curlᵛh = curlᵛ(R, A, B, 𝐡)
curlᵛhLims = (-maximum(abs.(curlᵛh)),maximum(abs.(curlᵛh)))
difVertexCurls = (oldVertexCurls .- (-1.0.*curlᵛh))
maxdifVertexCurls = maximum(abs.(difVertexCurls))
difVertexCurlLims = (-maxdifVertexCurls, maxdifVertexCurls)
# difVertexCurlLims = (-0.001, 0.001)# (-maxdifVertexCurls, maxdifVertexCurls)
@show maxdifVertexCurls


fig = Figure(size=(1000,1000))

ax11 = Axis(fig[1,1], aspect=DataAspect())
for i=1:nCells
    poly!(ax11,cellPolygons[i],color=oldCellDivs[i],colorrange=oldCellDivLims,colormap=:bwr,strokewidth=1,strokecolor=(:black,0.25))
end
hidedecorations!(ax11)
hidespines!(ax11)
Colorbar(fig[1,2], colorrange=oldCellDivLims, colormap=:bwr)
Label(fig[1,1,Bottom()],L"oldCellDivs=div^c h",fontsize = 24)
#ax12 
ax12 = Axis(fig[1,3], aspect=DataAspect())
for i=1:nCells
    poly!(ax12,cellPolygons[i],color=-cocurlᶜh[i],colorrange=cocurlᶜhLims,colormap=:bwr,strokewidth=1,strokecolor=(:black,0.25))
end
hidedecorations!(ax12)
hidespines!(ax12)
Colorbar(fig[1,4], colorrange=cocurlᶜhLims, colormap=:bwr)
Label(fig[1,3,Bottom()],L"-\{cocurl^c h\}_i",fontsize = 24)
#ax1
ax13 = Axis(fig[1,5], aspect=DataAspect())
for i=1:nCells
    poly!(ax13,cellPolygons[i],color=difCellDivs[i],colorrange=difCellDivLims,colormap=:bwr,strokewidth=1,strokecolor=(:black,0.25))
end
hidedecorations!(ax13)
hidespines!(ax13)
Colorbar(fig[1,6], colorrange=difCellDivLims, colormap=:bwr)
Label(fig[1,5,Bottom()],L"dif",fontsize = 24)

#ax21
ax21 = Axis(fig[2,1],aspect=DataAspect())
hidedecorations!(ax21)
hidespines!(ax21)
for k=1:nVerts
    poly!(ax21,linkTriangles[k],color=oldVertexDivs[k],colorrange=oldVertexDivLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
end
for i=1:nCells
    poly!(ax21,cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
end
Colorbar(fig[2,2],limits=oldVertexDivLims,colormap=:bwr)
Label(fig[2,1,Bottom()], L"oldVertexDivs=div^v h", fontsize = 24)
#ax22
ax22 = Axis(fig[2,3],aspect=DataAspect())
hidedecorations!(ax22)
hidespines!(ax22)
for k=1:nVerts
    poly!(ax22,linkTriangles[k],color=-cocurlᵛh[k],colorrange=cocurlᵛhLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
end
for i=1:nCells
    poly!(ax22,cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
end
Colorbar(fig[2,4], colorrange=cocurlᵛhLims, colormap=:bwr)
Label(fig[2,3,Bottom()],L"-\{cocurl^v h\}_k",fontsize = 24)
ax23 = Axis(fig[2,5],aspect=DataAspect())
hidedecorations!(ax23)
hidespines!(ax23)
for k=1:nVerts
    poly!(ax23,linkTriangles[k],color=difVertexDivs[k],colorrange=difVertexDivLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
end
for i=1:nCells
    poly!(ax23,cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
end
Colorbar(fig[2,6], colorrange=difVertexDivLims, colormap=:bwr)
Label(fig[2,5,Bottom()],L"dif",fontsize = 24)


#ax31
ax31 = Axis(fig[3,1],aspect=DataAspect())
hidedecorations!(ax31)
hidespines!(ax31)
for k=1:nVerts
    poly!(ax31,linkTriangles[k],color=oldVertexCurls[k],colorrange=oldVertexCurlLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
end
for i=1:nCells
    poly!(ax31,cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
end
Colorbar(fig[3,2],limits=oldVertexCurlLims,colormap=:bwr)
Label(fig[3,1,Bottom()], L"oldVertexCurls=CURLᵛ h", fontsize = 24)
#ax32
ax32 = Axis(fig[3,3],aspect=DataAspect())
hidedecorations!(ax32)
hidespines!(ax32)
for k=1:nVerts
    poly!(ax32,linkTriangles[k],color=-curlᵛh[k],colorrange=curlᵛhLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
end
for i=1:nCells
    poly!(ax32,cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
end
Colorbar(fig[3,4], colorrange=curlᵛhLims, colormap=:bwr)
Label(fig[3,3,Bottom()],L"-\{curl^v h\}_k",fontsize = 24)
#ax33
ax33 = Axis(fig[3,5],aspect=DataAspect())
hidedecorations!(ax33)
hidespines!(ax33)
for k=1:nVerts
    poly!(ax33,linkTriangles[k],color=difVertexCurls[k],colorrange=difVertexCurlLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
end
for i=1:nCells
    poly!(ax33,cellPolygons[i],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
end
Colorbar(fig[3,6], colorrange=difVertexCurlLims, colormap=:bwr)
Label(fig[3,5,Bottom()],L"dif",fontsize = 24)

display(fig)
save(datadir("oldVsNewDifferentialOperators.png"), fig)
#%%


# oldCellDiv => -cocurlᶜh
# oldVertexDiv => cocurlᵛh
# oldVertexCurl => curlᵛh

# So what we solved in the couple paper was actually
# Old: Lf ψᶜ = -oldCellDivs 
# New: Lf ψᶜ = cocurlᶜ 𝐡 

# Old: Lt ψ̆ᵛ = -oldVertexDivs
# New: Lt ψ̆ᵛ = -cocurlᵛ h̆

# New: Lt Ψ̆ᵛ = oldVertexCurls
# New: Lt Ψ̆ᵛ = -curlᵛ h̆