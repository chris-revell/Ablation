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
using FromFile

@from "$(srcdir("AblateCells.jl"))" using AblateCells

α = 2.0
β = 0.0
ϵᵢ = SMatrix{2, 2, Float64}([
                0.0 1.0
                -1.0 0.0
            ])
ϵₖ = SMatrix{2, 2, Float64}([
                0.0 -1.0
                1.0 0.0
            ])

inFile = datadir("referenceSystems", "Large_testSystem5.jld2")
importedData = load(inFile)
@unpack R, A, B, F = importedData

systemCOM = sum(R)./length(R)
cellCentres = findCellCentresOfMass(R, A, B)
centralCell = findmin(norm.([cellCentres[i].-systemCOM for i=1:size(B,1)]))[2]
centralCellCOM = cellCentres[centralCell]

R2, A2, B2 = ablateCells(R, A, B, [centralCell])

Lprimal = edgeLaplacianPrimal(R2, A2, B2)
Ldual = edgeLaplacianDual(R2, A2, B2)
eigenvectors_Lprimal = [col for col in  eachcol((eigen(Matrix(Lprimal))).vectors)]
eigenvalues_Lprimal = (eigen(Matrix(Lprimal))).values
eigenvectors_Ldual = [col for col in  eachcol((eigen(Matrix(Ldual))).vectors)]
eigenvalues_Ldual = (eigen(Matrix(Ldual))).values

𝐜ⱼ = findEdgeMidpoints(R2, A2)
𝐂ⱼ = findCellLinkMidpoints(R2, A2, B2)
boundaryEdges = findBoundaryEdges(B2)
primalBasisParallel = findEdgeTangents(R2, A2)./(findEdgeLengths(R2, A2).^2)
primalBasisPerp = [ϵᵢ*v for v in primalBasisParallel]
dualBasisParallel = findCellLinks(R2, A2, B2)./(findCellLinkLengths(R2, A2, B2).^2)
dualBasisPerp = [ϵₖ*v for v in dualBasisParallel]

#%%

fig = Figure(size=(1500,600), fontsize=24)
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"(%$l)" for l in individualLetters]

radii = norm.([𝐜ⱼ[j].-centralCellCOM for j=1:size(B2,2)])
dummyDists = collect(maximum(radii)/100:maximum(radii)/100:maximum(radii))

row = 1
col = 1
push!(axes, Axis(fig[1,1], aspect=DataAspect()))
cellPolygons = findCellPolygons(R2, A2, B2)
# getRandomColor(seed) = RGB(rand(MersenneTwister(seed),3)...)
# colors = getRandomColor.(collect(1:size(B,2)))
scatter!(axes[end], Point{2,Float64}(centralCellCOM), color=:red)
for i=1:size(B2,1)
    poly!(axes[end], cellPolygons[i], color=(:black, 0.25), strokecolor=(:black, 1.0), strokewidth=1)
end
hidedecorations!(axes[end])
hidespines!(axes[end])
Label(fig[row+1,col,Bottom()], popfirst!(subfigureLabels), fontsize=24) #"Single hole, 1st eigenmode, primal network, α=$(α), β=$(β)")

col+=1
# Primal network 
α = 1.0
β = 1.0
push!(axes, Axis(fig[row,col], yscale=log10, xscale=log10, aspect=1))
edgeVectors = eigenvectors_Lprimal[1].*(α.*primalBasisParallel .+ β.*primalBasisPerp)
scatter!(axes[end], radii, norm.(edgeVectors), color=(:blue,0.1))
lines!(axes[end], dummyDists, 1.0./dummyDists, color=(:black, 0.75), linestyle=:dash)
lines!(axes[end], dummyDists, 1.0./(dummyDists).^2, color=(:black, 0.75), linestyle=:dash)
lines!(axes[end], dummyDists, 1.0./(dummyDists).^3, color=(:black, 0.75), linestyle=:dash)
axes[end].xlabel = L"r"
axes[end].ylabel = L"\chi"
Label(fig[row+1,col,Bottom()], popfirst!(subfigureLabels), fontsize=24) #"Single hole, 1st eigenmode, primal network, α=$(α), β=$(β)")
# col += 1
# α = 0.0
# β = 2.0
# push!(axes, Axis(fig[row,col], yscale=log10, xscale=log10))
# edgeVectors = eigenvectors_Lprimal[1].*(α.*primalBasisParallel .+ β.*primalBasisPerp)
# scatter!(axes[end], radii, norm.(edgeVectors), color=(:blue,0.1))
# lines!(axes[end], dummyDists, 1.0./dummyDists, color=:black)
# lines!(axes[end], dummyDists, 1.0./(dummyDists).^2, color=:black)
# lines!(axes[end], dummyDists, 1.0./(dummyDists).^3, color=:black)
# axes[end].xlabel = L"r"
# axes[end].ylabel = L"\chi"
# Label(fig[row+1,col,Bottom()], popfirst!(subfigureLabels), fontsize=24) #"Single hole, 1st eigenmode, primal network, α=$(α), β=$(β)")

# Dual network 
col += 1
α = 1.0
β = 1.0
push!(axes, Axis(fig[row,col], yscale=log10, xscale=log10, aspect=1))
edgeVectors = eigenvectors_Ldual[1].*(α.*dualBasisParallel .+ β.*dualBasisPerp)
scatter!(axes[end], radii[boundaryEdges.==0], norm.(edgeVectors[boundaryEdges.==0]), color=(:blue,0.1))
scatter!(axes[end], radii[boundaryEdges.!=0], norm.(edgeVectors[boundaryEdges.!=0]), color=(:blue,0.1))
lines!(axes[end], dummyDists, 1.0./dummyDists, color=(:black, 0.75), linestyle=:dash)
lines!(axes[end], dummyDists, 1.0./(dummyDists).^2, color=(:black, 0.75), linestyle=:dash)
lines!(axes[end], dummyDists, 1.0./(dummyDists).^3, color=(:black, 0.75), linestyle=:dash)
axes[end].xlabel = L"r"
axes[end].ylabel = L"\chi"
Label(fig[row+1,col,Bottom()], popfirst!(subfigureLabels), fontsize=24) #"Single hole, 1st eigenmode, primal network, α=$(α), β=$(β)")

# col += 1
# α = 0.0
# β = 2.0
# push!(axes, Axis(fig[row,col], yscale=log10, xscale=log10))
# edgeVectors = eigenvectors_Ldual[1].*(α.*dualBasisParallel .+ β.*dualBasisPerp)
# scatter!(axes[end], radii[boundaryEdges.==0], norm.(edgeVectors[boundaryEdges.==0]), color=(:blue,0.1))
# scatter!(axes[end], radii[boundaryEdges.!=0], norm.(edgeVectors[boundaryEdges.!=0]), color=(:blue,0.1))
# lines!(axes[end], dummyDists, 1.0./dummyDists, color=:black)
# lines!(axes[end], dummyDists, 1.0./(dummyDists).^2, color=:black)
# lines!(axes[end], dummyDists, 1.0./(dummyDists).^3, color=:black)
# axes[end].xlabel = L"r"
# axes[end].ylabel = L"\chi"
# Label(fig[row+1,col,Bottom()], popfirst!(subfigureLabels), fontsize=24) #"Single hole, 1st eigenmode, primal network, α=$(α), β=$(β)")


rowsize!(fig.layout, 1, Relative(0.99))
rowsize!(fig.layout, 2, Relative(0.01))

display(fig)
save(plotsdir("edgeLaplacianHarmonicFieldCellRemoved.png"), fig)


# row = 1
# col = 1
# # Primal network 
# α = 2.0
# β = 0.0
# push!(axes, Axis(fig[row,col], aspect=DataAspect()))
# poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
# edgeVectors = eigenvectors_Lprimal[1].*(α.*primalBasisParallel .+ β.*primalBasisPerp)
# arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
# arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ), Vec{2,Float64}.(edgeVectors), color=arrowColours, linewidth=2)
# Label(fig[row,col,Bottom()], popfirst!(subfigureLabels), fontsize=36) #"Single hole, 1st eigenmode, primal network, α=$(α), β=$(β)")
# col += 1
# α = 0.0
# β = 2.0
# push!(axes, Axis(fig[row,col], aspect=DataAspect()))
# poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
# edgeVectors = eigenvectors_Lprimal[1].*(α.*primalBasisParallel .+ β.*primalBasisPerp)
# arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
# arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ), Vec{2,Float64}.(edgeVectors), color=arrowColours, linewidth=2)
# Label(fig[row,col,Bottom()], popfirst!(subfigureLabels), fontsize=36) #"Single hole, 1st eigenmode, primal network, α=$(α), β=$(β)")
# # Dual network 
# col += 1
# α = 2.0
# β = 0.0
# push!(axes, Axis(fig[row,col], aspect=DataAspect()))
# poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
# edgeVectors = eigenvectors_Ldual[1].*(α.*dualBasisParallel .+ β.*dualBasisPerp)
# arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
# arrows!(axes[end], Point{2,Float64}.(𝐂ⱼ[boundaryEdges.==0]), Vec{2,Float64}.(edgeVectors[boundaryEdges.==0]), color=arrowColours[boundaryEdges.==0], linewidth=2)
# arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ[boundaryEdges.!=0]), Vec{2,Float64}.(edgeVectors[boundaryEdges.!=0]), color=arrowColours[boundaryEdges.!=0], linewidth=2)
# Label(fig[row,col,Bottom()], popfirst!(subfigureLabels), fontsize=36) #"Single hole, 1st eigenmode, dual network, α=$(α), β=$(β)")
# col += 1
# α = 0.0
# β = 2.0
# push!(axes, Axis(fig[row,col], aspect=DataAspect()))
# poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
# edgeVectors = eigenvectors_Ldual[1].*(α.*dualBasisParallel .+ β.*dualBasisPerp)
# arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
# arrows!(axes[end], Point{2,Float64}.(𝐂ⱼ[boundaryEdges.==0]), Vec{2,Float64}.(edgeVectors[boundaryEdges.==0]), color=arrowColours[boundaryEdges.==0], linewidth=2)
# arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ[boundaryEdges.!=0]), Vec{2,Float64}.(edgeVectors[boundaryEdges.!=0]), color=arrowColours[boundaryEdges.!=0], linewidth=2)
# Label(fig[row,col,Bottom()], popfirst!(subfigureLabels), fontsize=36) #"Single hole, 1st eigenmode, dual network, α=$(α), β=$(β)")

# hidedecorations!.(axes)
# hidespines!.(axes)

#%%
