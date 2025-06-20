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

#%%

integ = vertexModel(nRows=11, nCycles=1.0)
R = reinterpret(SVector{2,Float64}, integ.u) 

params, matrices = integ.p
@unpack A, B = matrices


#%%

getRandomColor(seed) = RGB(rand(Xoshiro(seed),3)...)

function ablateCells(R, A, B, ablatedCellsList)
    newB = B[[ii for ii=1:size(B,1) if ii∉ablatedCellsList],[jj for jj=1:size(B,2)]]
    edgeCellPairs = [findall(x->x!=0,newB[:,j]) for j=1:size(newB,2)]
    orphanedEdges = findall(x->length(x)==0, edgeCellPairs)
    newB2 = newB[[ii for ii=1:size(newB,1)],[jj for jj=1:size(newB,2) if jj∉orphanedEdges]]
    newA = A[[jj for jj=1:size(A,1) if jj∉orphanedEdges],[kk for kk=1:size(A,2)]]
    edgeVertexPairs = [findall(x->x!=0,newA[:,k]) for k=1:size(newA,2)]
    orphanedVertices = findall(x->length(x)==0, edgeVertexPairs)
    newA2 = newA[[jj for jj=1:size(newA,1)],[kk for kk=1:size(newA,2) if kk∉orphanedVertices]]    
    senseCheck(newA2, newB2)
    newR = [R[k] for k=1:length(R) if k∉orphanedVertices]
    return newR, newA2, newB2
end 

function ablateEdge(A, B, j)
    j_is = first.(findnz(B[:,j]))
    newB = B[[ii for ii=1:size(B,1) if ii∉ablatedCellsList],[jj for jj=1:size(B,2)]]
    edgeCellPairs = first.([findnz(newB[:,j]) for j=1:size(newB,2)])
    orphanedEdges = findall(x->length(x)==0, edgeCellPairs)
    newB2 = newB[[ii for ii=1:size(newB,1)],[jj for jj=1:size(newB,2) if jj∉orphanedEdges]]
    newA = A[[jj for jj=1:size(A,1) if jj∉orphanedEdges],[kk for kk=1:size(A,2)]]
    senseCheck(newA, newB2)
    return newA, newB2
end 


#%%

cellPolygons = findCellPolygons(R, A, B)
cellCentres = findCellCentresOfMass(R, A, B)

fig = Figure(size=(500,1500))
ax = Axis(fig[1,1], aspect=DataAspect())
hidedecorations!(ax)
hidespines!(ax)
for i=1:size(B,1)
    poly!(ax, cellPolygons[i], color=(getRandomColor(i), 0.5), strokecolor=(:black, 1.0), strokewidth=2)
end
scatter!(ax, Point{2,Float64}.(cellCentres), color=(:black,1.0), markersize=10)
annotations!(ax, string.(collect(1:params.nCells)), Point{2,Float64}.(cellCentres), fontsize=12, color=(:black,1.0))
display(fig)

#%%

ablatedCells = [5, 6, 157, 134, 176, 25, 23, 152, 94, 21, 148, 116, 147, 8, 7, 118, 107, 142, 183]

#%%

ablatedR, ablatedA, ablatedB = ablateCells(R, A, B, ablatedCells)
cellPolygonsAblated = findCellPolygons(ablatedR, ablatedA, ablatedB)
cellCentresAblated = findCellCentresOfMass(ablatedR, ablatedA, ablatedB)

ax2 = Axis(fig[2,1], aspect=DataAspect())
for i=1:size(ablatedB,1)
    poly!(ax2, cellPolygonsAblated[i], color=(getRandomColor(i), 0.5), strokecolor=(:black, 1.0), strokewidth=2)
end
scatter!(ax2, Point{2,Float64}.(cellCentresAblated), color=(:black,1.0), markersize=10)
# scatter!(ax2, Point{2,Float64}.(ablatedR), color=(:black,1.0), markersize=10)
annotations!(ax2, string.(collect(1:size(ablatedB,1))), Point{2,Float64}.(cellCentresAblated), fontsize=12, color=(:black,1.0))
hidedecorations!(ax2)
hidespines!(ax2)
display(fig)


#%%

integ2 = vertexModel(initialSystem="argument", R_in=ablatedR, A_in=ablatedA, B_in=ablatedB, nCycles=1.0)
 
#%%
ablatedGrownR = reinterpret(SVector{2,Float64}, integ2.u) 
params2, matrices2 = integ2.p
ablatedGrownA = ablatedGrownA
ablatedGrownB = ablatedGrownB
ablatedGrownF = ablatedGrownB

cellPolygons2 = findCellPolygons(ablatedGrownR, ablatedGrownA, ablatedGrownB)
cellCentres2 = findCellCentresOfMass(ablatedGrownR, ablatedGrownA, ablatedGrownB)

ax3 = Axis(fig[3,1], aspect=DataAspect())
hidedecorations!(ax3)
hidespines!(ax3)
for i=1:size(ablatedGrownB,1)
    poly!(ax3, cellPolygons2[i], color=(getRandomColor(i), 0.5), strokecolor=(:black, 1.0), strokewidth=2)
end
scatter!(ax3, Point{2,Float64}.(cellCentres2), color=(:black,1.0), markersize=10)
annotations!(ax3, string.(collect(1:size(ablatedGrownB,1))), Point{2,Float64}.(cellCentres2), fontsize=12, color=(:black,1.0))
display(fig)

save("ablation.png", fig)

jldsave(datadir("$(Dates.format(Dates.now(),"yy-mm-dd-HH-MM-SS"))_testSystem.jld2"); R, A, B, ablatedR, ablatedA, ablatedB, ablatedGrownR, ablatedGrownA, ablatedGrownB, ablatedGrownF, ablatedCells)


# ℒₓ = ablatedA*transpose(ablatedA) + transpose(ablatedB)*ablatedB

# eigenvectors = (eigen(Matrix(ℒₓ))).vectors
# eigenvalues = (eigen(Matrix(ℒₓ))).values

# edgeTrapezia = findEdgeTrapezia(R, ablatedA, ablatedB)

# fig = Figure(size=(500,500))
# ax = Axis(fig[1,1], aspect=DataAspect())

# crange = (minimum(eigenvectors[:,1]), maximum(eigenvectors[:,1]))
# for i=1:size(ablatedA,1)
#     poly!(ax, edgeTrapezia[i], color=abs(eigenvectors[i,1]), colorrange=crange, colormap=:inferno)
# end
# hidedecorations!(ax)
# hidespines!(ax)
# Colorbar(fig[1,2], colorrange=crange, colormap=:inferno)
# # ax2 = Axis(fig[2,1])
# # text!(ax2, 0, 0, text="test")
# display(fig)












# #%%


# ḡ = ((onesVec'*H*cellDivs)/(onesVec'*H*ones(nCells))).*onesVec
# ğ = cellDivs.-ḡ
# ψ̆ = zeros(nCells)
# spectrum = Float64[]
# for k=2:nCells
#     numerator = eigenvectors[:,k]'*H*ğ
#     denominator = eigenvalues[k]*(eigenvectors[:,k]'*H*eigenvectors[:,k])
#     ψ̆ .-= (numerator/denominator).*eigenvectors[:,k]
#     push!(spectrum,(numerator/denominator))
# end
# return ψ̆, spectrum
