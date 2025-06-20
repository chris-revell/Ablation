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



@from "$(srcdir("AblateCells.jl"))" using AblateCells
getRandomColor(seed) = RGB(rand(Xoshiro(seed),3)...)

#%%
integ = vertexModel(nRows=11, nCycles=1.0, outputToggle=0, frameDataToggle=0, frameImageToggle=0, videoToggle=0, setRandomSeed=1234, divisionToggle=1)
R = reinterpret(SVector{2,Float64}, integ.u) 
params, matrices = integ.p
@unpack A, B, F = matrices 

cellPolygons = findCellPolygons(R, A, B)
cellCentres = findCellCentresOfMass(R, A, B)

keptCell = 6
keptEdges = findall(x->x!=0, B[keptCell,:])
keptVertices = unique(getindex.(findall(x->x!=0, A[keptEdges,:]), 2))
ablatedR = R[keptVertices]
ablatedA = A[keptEdges, keptVertices]
ablatedB = B[keptCell, keptEdges]
cellPolygonsAblated = cellPolygons[keptCell]


fig = Figure(size=(500,1500))
ax = Axis(fig[1,1], aspect=DataAspect())
poly!(ax, cellPolygons[keptCell], color=(:white, 0.0), strokecolor=(:black, 1.0), strokewidth=2)
arrows!(ax, Point{2,Float64}.(R[keptVertices]), Vec{2,Float64}.(F[keptVertices, keptCell]))
hidedecorations!(ax2)
hidespines!(ax2)

display(fig)



nCells = size(ablatedB,1)
nEdges = size(ablatedB,2)
nVerts = size(ablatedA,2)
linkTriangles = findCellLinkTriangles(R, A, B)
linkTriangleAreas = findCellLinkTriangleAreas(R, A, B)
cellPolygons = findCellPolygons(R, A, B)

R = ablatedR
A = ablatedA
B = sparse(reshape(ablatedB,(1,6)))
F = sparse(reshape(F[keptVertices, keptCell], (6,1)))

𝐡 = hNetwork(R, A, B, F)



curlᶜh = curlᶜ(R, A, B, 𝐡)
curlᶜhLims = (-max(maximum(abs.(curlᶜh)), 0.1), max(maximum(abs.(curlᶜh)), 0.1))
curlᵛh = curlᵛ(R, A, B, 𝐡)
curlᵛhLims = (-maximum(abs.(curlᵛh)), maximum(abs.(curlᵛh)))
divᶜh = divᶜ(R, A, B, 𝐡)
divᶜhLims = (-maximum(abs.(divᶜh)), maximum(abs.(divᶜh)))
divᵛh = divᵛ(R, A, B, 𝐡)
divᵛhLims = (-maximum(abs.(divᵛh)), maximum(abs.(divᵛh)))
cocurlᶜh = cocurlᶜ(R, A, B, 𝐡)
cocurlᶜhLims = (-maximum(abs.(cocurlᶜh)), maximum(abs.(cocurlᶜh)))
cocurlᵛh = cocurlᵛ(R, A, B, 𝐡)
cocurlᵛhLims = (-maximum(abs.(cocurlᵛh)), maximum(abs.(cocurlᵛh)))
codᶜh = codᶜ(R, A, B, 𝐡)
codᶜhLims = (-maximum(abs.(codᶜh)), maximum(abs.(codᶜh)))
codᵛh = codᵛ(R, A, B, 𝐡)
codᵛhLims = (-maximum(abs.(codᵛh)), maximum(abs.(codᵛh)))
# derivs = [curlᶜh, curlᵛh, divᶜh, divᵛh, cocurlᶜh, cocurlᵛh, codᶜh, codᵛh]

#%%

fig = Figure(size=(1000,2000))
axes = Axis[]

push!(axes, Axis(fig[1,1], aspect=DataAspect()))
poly!(axes[end],cellPolygons[keptCell],color=cocurlᶜh[1],colorrange=cocurlᶜhLims,colormap=:bwr,strokewidth=1,strokecolor=(:black,0.25))
Colorbar(fig[1,2], colorrange=cocurlᶜhLims, colormap=:bwr)
Label(fig[1,1,Bottom()],L"\{cocurl^c h\}_i",fontsize = 24)

push!(axes, Axis(fig[2,1], aspect=DataAspect()))
for (i,k) in enumerate(keptVertices)
    poly!(axes[end],linkTriangles[k],color=cocurlᵛh[i],colorrange=cocurlᵛhLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
end
poly!(axes[end],cellPolygons[keptCell],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
Colorbar(fig[2,2],limits=cocurlᵛhLims,colormap=:bwr)
Label(fig[2,1,Bottom()], L"\{cocurl^v \breve{h}\}_k", fontsize = 24)

push!(axes, Axis(fig[3,1], aspect=DataAspect()))
poly!(axes[end],cellPolygons[keptCell],color=curlᶜh[1],colorrange=curlᶜhLims,colormap=:bwr,strokewidth=1,strokecolor=(:black,0.25))
Colorbar(fig[3,2], colorrange=curlᶜhLims, colormap=:bwr)
Label(fig[3,1,Bottom()],L"\{curl^c h\}_i",fontsize = 24)

push!(axes, Axis(fig[4,1], aspect=DataAspect()))
for (i,k) in enumerate(keptVertices)
    poly!(axes[end],linkTriangles[k],color=curlᵛh[i],colorrange=curlᵛhLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
end
poly!(axes[end],cellPolygons[keptCell],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
Colorbar(fig[4,2],limits=curlᵛhLims,colormap=:bwr)
Label(fig[4,1,Bottom()], L"\{curl^v \breve{h}\}_k", fontsize = 24)



push!(axes, Axis(fig[1,3], aspect=DataAspect()))
poly!(axes[end],cellPolygons[keptCell],color=-divᶜh[1],colorrange=divᶜhLims,colormap=:bwr,strokewidth=1,strokecolor=(:black,0.25))
Colorbar(fig[1,4], colorrange=divᶜhLims, colormap=:bwr)
Label(fig[1,3,Bottom()],L"-\{div^c h\}_i",fontsize = 24)

push!(axes, Axis(fig[2,3], aspect=DataAspect()))
@show divᵛh
for (i,k) in enumerate(keptVertices)
    poly!(axes[end],linkTriangles[k],color=-divᵛh[i],colorrange=divᵛhLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
end
poly!(axes[end],cellPolygons[keptCell],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
Colorbar(fig[2,4],limits=divᵛhLims,colormap=:bwr)
Label(fig[2,3,Bottom()], L"-\{div^v \breve{h}\}_k", fontsize = 24)

push!(axes, Axis(fig[3,3], aspect=DataAspect()))
poly!(axes[end],cellPolygons[keptCell],color=codᶜh[1],colorrange=codᶜhLims,colormap=:bwr,strokewidth=1,strokecolor=(:black,0.25))
Colorbar(fig[3,4], colorrange=codᶜhLims, colormap=:bwr)
Label(fig[3,3,Bottom()],L"\{cod^c h\}_i",fontsize = 24)

push!(axes, Axis(fig[4,3], aspect=DataAspect()))
for (i,k) in enumerate(keptVertices)
    poly!(axes[end],linkTriangles[k],color=codᵛh[i],colorrange=codᵛhLims,colormap=:bwr,strokewidth=1,strokecolor=(:white,0.0))
end
poly!(axes[end],cellPolygons[keptCell],color=(:white,0.0),strokewidth=1,strokecolor=(:black,0.25))
Colorbar(fig[4,4],limits=codᵛhLims,colormap=:bwr)
Label(fig[4,3,Bottom()], L"\{cod^v \breve{h}\}_k", fontsize = 24)


hidedecorations!.(axes)
hidespines!.(axes)
display(fig)
save(datadir("Figure1$(inputSystem)Derivatives.png"), fig)
 