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

fig = Figure(size=(1000,1500))
axes = Axis[]

# Single hole 
inFile = datadir("referenceSystems", "SingleHole_testSystem.jld2")
importedData = load(inFile)
@unpack R, A, B, F = importedData
Lprimal = edgeLaplacianPrimal(R, A, B)
Ldual = edgeLaplacianDual(R, A, B)
eigenvectors_Lprimal = [col for col in  eachcol((eigen(Matrix(Lprimal))).vectors)]
eigenvalues_Lprimal = (eigen(Matrix(Lprimal))).values
eigenvectors_Ldual = [col for col in  eachcol((eigen(Matrix(Ldual))).vectors)]
eigenvalues_Ldual = (eigen(Matrix(Ldual))).values
cellPolygons = findCellPolygons(R, A, B)
𝐜ⱼ = findEdgeMidpoints(R, A)
primalBasisParallel = findEdgeTangents(R, A)./(findEdgeLengths(R, A).^2)
primalBasisPerp = [ϵᵢ*v for v in primalBasisParallel]
dualBasisParallel = findCellLinks(R, A, B)./(findCellLinkLengths(R, A, B).^2)
dualBasisPerp = [ϵₖ*v for v in dualBasisParallel]
# Primal network 
push!(axes, Axis(fig[1,1], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.5), strokewidth=1)
edgeVectors = eigenvectors_Lprimal[1].*(α.*primalBasisParallel .+ β.*primalBasisPerp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ), Vec{2,Float64}.(edgeVectors), color=arrowColours, linewidth=2)
Label(fig[1,1,Bottom()], "Single hole, 1st eigenmode, primal network")

# Dual network 
push!(axes, Axis(fig[1,2], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.5), strokewidth=1)
edgeVectors = eigenvectors_Ldual[1].*(α.*dualBasisParallel .+ β.*dualBasisPerp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ), Vec{2,Float64}.(edgeVectors), color=arrowColours, linewidth=2)
Label(fig[1,2,Bottom()], "Single hole, 1st eigenmode, dual network")

# Double hole 
inFile = datadir("referenceSystems", "DoubleHole_testSystem.jld2")
importedData = load(inFile)
@unpack R, A, B, F = importedData
Lprimal = edgeLaplacianPrimal(R, A, B)
Ldual = edgeLaplacianDual(R, A, B)
eigenvectors_Lprimal = [col for col in  eachcol((eigen(Matrix(Lprimal))).vectors)]
eigenvalues_Lprimal = (eigen(Matrix(Lprimal))).values
eigenvectors_Ldual = [col for col in  eachcol((eigen(Matrix(Ldual))).vectors)]
eigenvalues_Ldual = (eigen(Matrix(Ldual))).values
cellPolygons = findCellPolygons(R, A, B)
𝐜ⱼ = findEdgeMidpoints(R, A)
primalBasisParallel = findEdgeTangents(R, A)./(findEdgeLengths(R, A).^2)
primalBasisPerp = [ϵᵢ*v for v in primalBasisParallel]
dualBasisParallel = findCellLinks(R, A, B)./(findCellLinkLengths(R, A, B).^2)
dualBasisPerp = [ϵₖ*v for v in dualBasisParallel]
# First eigenmode 
# Primal network 
push!(axes, Axis(fig[2,1], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.5), strokewidth=1)
edgeVectors = eigenvectors_Lprimal[1].*(α.*primalBasisParallel .+ β.*primalBasisPerp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ), Vec{2,Float64}.(edgeVectors), color=arrowColours, linewidth=2)
Label(fig[2,1,Bottom()], "Double hole, 1st eigenmode, primal network")
# Dual network 
push!(axes, Axis(fig[2,2], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.5), strokewidth=1)
edgeVectors = eigenvectors_Ldual[1].*(α.*dualBasisParallel .+ β.*dualBasisPerp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ), Vec{2,Float64}.(edgeVectors), color=arrowColours, linewidth=2)
Label(fig[2,2,Bottom()], "Double hole, 1st eigenmode, dual network")
# Second eigenmode 
# Primal network 
push!(axes, Axis(fig[3,1], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.5), strokewidth=1)
edgeVectors = eigenvectors_Lprimal[2].*(α.*primalBasisParallel .+ β.*primalBasisPerp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ), Vec{2,Float64}.(edgeVectors), color=arrowColours, linewidth=2)
Label(fig[3,1,Bottom()], "Double hole, 2nd eigenmode, primal network")
# Dual network 
push!(axes, Axis(fig[3,2], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.5), strokewidth=1)
edgeVectors = eigenvectors_Ldual[2].*(α.*dualBasisParallel .+ β.*dualBasisPerp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ), Vec{2,Float64}.(edgeVectors), color=arrowColours, linewidth=2)
Label(fig[3,2,Bottom()], "Double hole, 2nd eigenmode, dual network")

hidedecorations!.(axes)
hidespines!.(axes)
display(fig)
save(datadir("edgeLaplacianEigenvectors_α=$(α)_β=$(β).png"), fig)



#%%









#%%

# divᵛb = divᵛ(allOutputs["SingleHole_R"], allOutputs["SingleHole_A"], allOutputs["SingleHole_B"], edgeVectorsSingleHolePrimal)
# @show maximum(abs.(divᵛb))
# curlᶜb = curlᶜ(allOutputs["SingleHole_R"], allOutputs["SingleHole_A"], allOutputs["SingleHole_B"], edgeVectorsSingleHolePrimal)
# @show maximum(abs.(curlᶜb))
# codivᵛb = codivᵛ(allOutputs["SingleHole_R"], allOutputs["SingleHole_A"], allOutputs["SingleHole_B"], edgeVectorsSingleHolePrimal)
# @show maximum(abs.(codivᵛb))
# cocurlᶜb = cocurlᶜ(allOutputs["SingleHole_R"], allOutputs["SingleHole_A"], allOutputs["SingleHole_B"], edgeVectorsSingleHolePrimal)
# @show maximum(abs.(cocurlᶜb))



# divᵛb = divᵛ(allOutputs["DoubleHole_R"], allOutputs["DoubleHole_A"], allOutputs["DoubleHole_B"], edgeVectorsDoubleHolePrimal1)
# @show maximum(abs.(divᵛb))
# curlᶜb = curlᶜ(allOutputs["DoubleHole_R"], allOutputs["DoubleHole_A"], allOutputs["DoubleHole_B"], edgeVectorsDoubleHolePrimal1)
# @show maximum(abs.(curlᶜb))
# codivᵛb = codivᵛ(allOutputs["DoubleHole_R"], allOutputs["DoubleHole_A"], allOutputs["DoubleHole_B"], edgeVectorsDoubleHolePrimal1)
# @show maximum(abs.(codivᵛb))
# cocurlᶜb = cocurlᶜ(allOutputs["DoubleHole_R"], allOutputs["DoubleHole_A"], allOutputs["DoubleHole_B"], edgeVectorsDoubleHolePrimal1)
# @show maximum(abs.(cocurlᶜb))


# fig = Figure(size=(1000,1000))
# ax = Axis(fig[1,1], aspect=DataAspect())
# clims = (-maximum(abs.(codivᵛb)), maximum(abs.(codivᵛb)))
# linkTriangles = findCellLinkTriangles(allOutputs["DoubleHole_R"], allOutputs["DoubleHole_A"], allOutputs["DoubleHole_B"])
# for k=1:size(allOutputs["DoubleHole_A"], 2)
#     poly!(ax, linkTriangles[k], color=codivᵛb[k], colorrange=clims, colormap=:bwr, strokecolor=(:black, 1.0), strokewidth=2)
# end
# for i=1:size(allOutputs["DoubleHole_B"], 1)
#     poly!(ax, allOutputs["DoubleHole_cellPolygons"][i], color=(:white, 0.0), strokecolor=(:black, 1.0), strokewidth=2)
# end
# display(fig)
    
    
    


# divᵛb = divᵛ(allOutputs["DoubleHole_R"], allOutputs["DoubleHole_A"], allOutputs["DoubleHole_B"], edgeVectorsDoubleHolePrimal2)
# @show maximum(abs.(divᵛb))
# curlᶜb = curlᶜ(allOutputs["DoubleHole_R"], allOutputs["DoubleHole_A"], allOutputs["DoubleHole_B"], edgeVectorsDoubleHolePrimal2)
# @show maximum(abs.(curlᶜb))

# codivᵛb = codivᵛ(allOutputs["SingleHole_R"], allOutputs["SingleHole_A"], allOutputs["SingleHole_B"], edgeVectorsDoubleHolePrimal2)
# @show maximum(abs.(codivᵛb))
# cocurlᶜb = cocurlᶜ(allOutputs["SingleHole_R"], allOutputs["SingleHole_A"], allOutputs["SingleHole_B"], edgeVectorsDoubleHolePrimal2)
# @show maximum(abs.(cocurlᶜb))