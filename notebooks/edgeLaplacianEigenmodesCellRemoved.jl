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

fig = Figure(size=(1500,1500*15/40), fontsize=36)
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"(%$l)" for l in individualLetters]

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
𝐂ⱼ = findCellLinkMidpoints(R, A, B)
boundaryEdges = findBoundaryEdges(B)
primalBasisParallel = findEdgeTangents(R, A)./(findEdgeLengths(R, A).^2)
primalBasisPerp = [ϵᵢ*v for v in primalBasisParallel]
dualBasisParallel = findCellLinks(R, A, B)./(findCellLinkLengths(R, A, B).^2)
dualBasisPerp = [ϵₖ*v for v in dualBasisParallel]
row = 1
col = 1
# Primal network 
α = 2.0
β = 0.0
push!(axes, Axis(fig[row,col], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
edgeVectors = eigenvectors_Lprimal[1].*(α.*primalBasisParallel .+ β.*primalBasisPerp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ), Vec{2,Float64}.(edgeVectors), color=arrowColours, linewidth=2)
Label(fig[row,col,Bottom()], popfirst!(subfigureLabels), fontsize=36) #"Single hole, 1st eigenmode, primal network, α=$(α), β=$(β)")
col += 1
α = 0.0
β = 2.0
push!(axes, Axis(fig[row,col], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
edgeVectors = eigenvectors_Lprimal[1].*(α.*primalBasisParallel .+ β.*primalBasisPerp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ), Vec{2,Float64}.(edgeVectors), color=arrowColours, linewidth=2)
Label(fig[row,col,Bottom()], popfirst!(subfigureLabels), fontsize=36) #"Single hole, 1st eigenmode, primal network, α=$(α), β=$(β)")
# Dual network 
col += 1
α = 2.0
β = 0.0
push!(axes, Axis(fig[row,col], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
edgeVectors = eigenvectors_Ldual[1].*(α.*dualBasisParallel .+ β.*dualBasisPerp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐂ⱼ[boundaryEdges.==0]), Vec{2,Float64}.(edgeVectors[boundaryEdges.==0]), color=arrowColours[boundaryEdges.==0], linewidth=2)
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ[boundaryEdges.!=0]), Vec{2,Float64}.(edgeVectors[boundaryEdges.!=0]), color=arrowColours[boundaryEdges.!=0], linewidth=2)
Label(fig[row,col,Bottom()], popfirst!(subfigureLabels), fontsize=36) #"Single hole, 1st eigenmode, dual network, α=$(α), β=$(β)")
col += 1
α = 0.0
β = 2.0
push!(axes, Axis(fig[row,col], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
edgeVectors = eigenvectors_Ldual[1].*(α.*dualBasisParallel .+ β.*dualBasisPerp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐂ⱼ[boundaryEdges.==0]), Vec{2,Float64}.(edgeVectors[boundaryEdges.==0]), color=arrowColours[boundaryEdges.==0], linewidth=2)
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ[boundaryEdges.!=0]), Vec{2,Float64}.(edgeVectors[boundaryEdges.!=0]), color=arrowColours[boundaryEdges.!=0], linewidth=2)
Label(fig[row,col,Bottom()], popfirst!(subfigureLabels), fontsize=36) #"Single hole, 1st eigenmode, dual network, α=$(α), β=$(β)")



R, A, B = ablateCells(R, A, B, [85])

Lprimal = edgeLaplacianPrimal(R, A, B)
Ldual = edgeLaplacianDual(R, A, B)
eigenvectors_Lprimal = [col for col in  eachcol((eigen(Matrix(Lprimal))).vectors)]
eigenvalues_Lprimal = (eigen(Matrix(Lprimal))).values
eigenvectors_Ldual = [col for col in  eachcol((eigen(Matrix(Ldual))).vectors)]
eigenvalues_Ldual = (eigen(Matrix(Ldual))).values
cellPolygons = findCellPolygons(R, A, B)
𝐜ⱼ = findEdgeMidpoints(R, A)
𝐂ⱼ = findCellLinkMidpoints(R, A, B)
boundaryEdges = findBoundaryEdges(B)
primalBasisParallel = findEdgeTangents(R, A)./(findEdgeLengths(R, A).^2)
primalBasisPerp = [ϵᵢ*v for v in primalBasisParallel]
dualBasisParallel = findCellLinks(R, A, B)./(findCellLinkLengths(R, A, B).^2)
dualBasisPerp = [ϵₖ*v for v in dualBasisParallel]
row = 2
col = 1
# Primal network 
α = 2.0
β = 0.0
push!(axes, Axis(fig[row,col], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
edgeVectors = eigenvectors_Lprimal[1].*(α.*primalBasisParallel .+ β.*primalBasisPerp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ), Vec{2,Float64}.(edgeVectors), color=arrowColours, linewidth=2)
Label(fig[row,col,Bottom()], popfirst!(subfigureLabels), fontsize=36) #"Single hole, 1st eigenmode, primal network, α=$(α), β=$(β)")
col += 1
α = 0.0
β = 2.0
push!(axes, Axis(fig[row,col], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
edgeVectors = eigenvectors_Lprimal[1].*(α.*primalBasisParallel .+ β.*primalBasisPerp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ), Vec{2,Float64}.(edgeVectors), color=arrowColours, linewidth=2)
Label(fig[row,col,Bottom()], popfirst!(subfigureLabels), fontsize=36) #"Single hole, 1st eigenmode, primal network, α=$(α), β=$(β)")
# Dual network 
col += 1
α = 2.0
β = 0.0
push!(axes, Axis(fig[row,col], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
edgeVectors = eigenvectors_Ldual[1].*(α.*dualBasisParallel .+ β.*dualBasisPerp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐂ⱼ[boundaryEdges.==0]), Vec{2,Float64}.(edgeVectors[boundaryEdges.==0]), color=arrowColours[boundaryEdges.==0], linewidth=2)
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ[boundaryEdges.!=0]), Vec{2,Float64}.(edgeVectors[boundaryEdges.!=0]), color=arrowColours[boundaryEdges.!=0], linewidth=2)
Label(fig[row,col,Bottom()], popfirst!(subfigureLabels), fontsize=36) #"Single hole, 1st eigenmode, dual network, α=$(α), β=$(β)")
col += 1
α = 0.0
β = 2.0
push!(axes, Axis(fig[row,col], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
edgeVectors = eigenvectors_Ldual[1].*(α.*dualBasisParallel .+ β.*dualBasisPerp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐂ⱼ[boundaryEdges.==0]), Vec{2,Float64}.(edgeVectors[boundaryEdges.==0]), color=arrowColours[boundaryEdges.==0], linewidth=2)
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ[boundaryEdges.!=0]), Vec{2,Float64}.(edgeVectors[boundaryEdges.!=0]), color=arrowColours[boundaryEdges.!=0], linewidth=2)
Label(fig[row,col,Bottom()], popfirst!(subfigureLabels), fontsize=36) #"Single hole, 1st eigenmode, dual network, α=$(α), β=$(β)")

hidedecorations!.(axes)
hidespines!.(axes)
display(fig)
save(datadir("edgeLaplacianEigenvectorsCellRemoved.png"), fig)
