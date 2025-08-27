using DrWatson
using DiscreteCalculus
using CairoMakie
using StaticArrays
# using VertexModel
using LinearAlgebra
using SparseArrays
using Random
using Colors 
using JLD2
using Dates

# (zpar, zperp) = (1,0)
ϵᵢ = SMatrix{2, 2, Float64}([
                0.0 1.0
                -1.0 0.0
            ])
ϵₖ = SMatrix{2, 2, Float64}([
                0.0 -1.0
                1.0 0.0
            ])

fig = Figure(size=(1500,1500*15/20), fontsize=36)
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"(%$l)" for l in individualLetters]

# Single hole 
inFile = datadir("referenceSystems", "quadraticPotentialNoPressure", "SingleHole_testSystem.jld2")
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


# Primal network 
(row,col) = (3,1)
(zpar, zperp) = (1,0)
push!(axes, Axis(fig[row,col], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
edgeVectors = eigenvectors_Lprimal[1].*(zpar.*primalBasisParallel .+ zperp.*primalBasisPerp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ), Vec{2,Float64}.(edgeVectors), color=arrowColours, linewidth=2, lengthscale=1.0)
# Label(fig[row,col,Bottom()], popfirst!(subfigureLabels), fontsize=36) #"Single hole, 1st eigenmode, primal network, zpar=$(zpar), zperp=$(zperp)")
# Label(fig[row,col,Bottom()], L"(%$(zpar), %$(zperp))", fontsize=36) #"Single hole, 1st eigenmode, primal network, zpar=$(zpar), zperp=$(zperp)")
(row,col) = (3,2)
(zpar, zperp) = (0,1)
push!(axes, Axis(fig[row,col], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
edgeVectors = eigenvectors_Lprimal[1].*(zpar.*primalBasisParallel .+ zperp.*primalBasisPerp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ), Vec{2,Float64}.(edgeVectors), color=arrowColours, linewidth=2, lengthscale=1.0)
# Label(fig[row,col,Bottom()], popfirst!(subfigureLabels), fontsize=36) #"Single hole, 1st eigenmode, primal network, zpar=$(zpar), zperp=$(zperp)")
# Label(fig[row,col,Bottom()], L"(%$(zpar), %$(zperp))", fontsize=36) #"Single hole, 1st eigenmode, primal network, zpar=$(zpar), zperp=$(zperp)")

# Dual network 
(row,col) = (3,3)
(zpar, zperp) = (1,0)
push!(axes, Axis(fig[row,col], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
edgeVectors = eigenvectors_Ldual[1].*(zpar.*dualBasisParallel .+ zperp.*dualBasisPerp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐂ⱼ[boundaryEdges.==0]), Vec{2,Float64}.(edgeVectors[boundaryEdges.==0]), color=arrowColours[boundaryEdges.==0], linewidth=2, lengthscale=1.0)
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ[boundaryEdges.!=0]), Vec{2,Float64}.(edgeVectors[boundaryEdges.!=0]), color=arrowColours[boundaryEdges.!=0], linewidth=2, lengthscale=1.0)
# Label(fig[row,col,Bottom()], popfirst!(subfigureLabels), fontsize=36) #"Single hole, 1st eigenmode, dual network, zpar=$(zpar), zperp=$(zperp)")
# Label(fig[row,col,Bottom()], L"(%$(zpar), %$(zperp))", fontsize=36) #"Single hole, 1st eigenmode, dual network, zpar=$(zpar), zperp=$(zperp)")
(row,col) = (3,4)
(zpar, zperp) = (0,1)
push!(axes, Axis(fig[row,col], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
edgeVectors = eigenvectors_Ldual[1].*(zpar.*dualBasisParallel .+ zperp.*dualBasisPerp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐂ⱼ[boundaryEdges.==0]), Vec{2,Float64}.(edgeVectors[boundaryEdges.==0]), color=arrowColours[boundaryEdges.==0], linewidth=2, lengthscale=1.0)
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ[boundaryEdges.!=0]), Vec{2,Float64}.(edgeVectors[boundaryEdges.!=0]), color=arrowColours[boundaryEdges.!=0], linewidth=2, lengthscale=1.0)
# Label(fig[row,col,Bottom()], popfirst!(subfigureLabels), fontsize=36) #"Single hole, 1st eigenmode, dual network, zpar=$(zpar), zperp=$(zperp)")
# Label(fig[row,col,Bottom()], L"(%$(zpar), %$(zperp))", fontsize=36) #"Single hole, 1st eigenmode, dual network, zpar=$(zpar), zperp=$(zperp)")

# Double hole 
inFile = datadir("referenceSystems", "quadraticPotentialNoPressure", "DoubleHole_testSystem.jld2")
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

# Second eigenmode 
# Primal network 
(row,col) = (4,1)
(zpar, zperp) = (1,0)
push!(axes, Axis(fig[row,col], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
edgeVectors = eigenvectors_Lprimal[2].*(zpar.*primalBasisParallel .+ zperp.*primalBasisPerp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ), Vec{2,Float64}.(edgeVectors), color=arrowColours, linewidth=2, lengthscale=1.0)
# Label(fig[row,col,Bottom()], popfirst!(subfigureLabels), fontsize=36) #"Double hole, 2nd eigenmode, primal network, zpar=$(zpar), zperp=$(zperp)")
# Label(fig[row,col,Bottom()], L"(%$(zpar), %$(zperp))", fontsize=36) #"Double hole, 2nd eigenmode, primal network, zpar=$(zpar), zperp=$(zperp)")
(row,col) = (4,2)
(zpar, zperp) = (0,1)
push!(axes, Axis(fig[row,col], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
edgeVectors = eigenvectors_Lprimal[2].*(zpar.*primalBasisParallel .+ zperp.*primalBasisPerp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ), Vec{2,Float64}.(edgeVectors), color=arrowColours, linewidth=2, lengthscale=1.0)
# Label(fig[row,col,Bottom()], popfirst!(subfigureLabels), fontsize=36) #"Double hole, 2nd eigenmode, primal network, zpar=$(zpar), zperp=$(zperp)")
# Label(fig[row,col,Bottom()], L"(%$(zpar), %$(zperp))", fontsize=36) #"Double hole, 2nd eigenmode, primal network, zpar=$(zpar), zperp=$(zperp)")

# First eigenmode 
# Dual network 
(row,col) = (4,3)
(zpar, zperp) = (1,0)
push!(axes, Axis(fig[row,col], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
edgeVectors = eigenvectors_Ldual[1].*(zpar.*dualBasisParallel .+ zperp.*dualBasisPerp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐂ⱼ[boundaryEdges.==0]), Vec{2,Float64}.(edgeVectors[boundaryEdges.==0]), color=arrowColours[boundaryEdges.==0], linewidth=2, lengthscale=1.0)
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ[boundaryEdges.!=0]), Vec{2,Float64}.(edgeVectors[boundaryEdges.!=0]), color=arrowColours[boundaryEdges.!=0], linewidth=2, lengthscale=1.0)
# Label(fig[row,col,Bottom()], popfirst!(subfigureLabels), fontsize=36) #"Double hole, 1st eigenmode, dual network, zpar=$(zpar), zperp=$(zperp)")
# Label(fig[row,col,Bottom()], L"(%$(zpar), %$(zperp))", fontsize=36) #"Double hole, 1st eigenmode, dual network, zpar=$(zpar), zperp=$(zperp)")
(row,col) = (4,4)
(zpar, zperp) = (0,1)
push!(axes, Axis(fig[row,col], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
edgeVectors = eigenvectors_Ldual[1].*(zpar.*dualBasisParallel .+ zperp.*dualBasisPerp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐂ⱼ[boundaryEdges.==0]), Vec{2,Float64}.(edgeVectors[boundaryEdges.==0]), color=arrowColours[boundaryEdges.==0], linewidth=2, lengthscale=1.0)
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ[boundaryEdges.!=0]), Vec{2,Float64}.(edgeVectors[boundaryEdges.!=0]), color=arrowColours[boundaryEdges.!=0], linewidth=2, lengthscale=1.0)
# Label(fig[row,col,Bottom()], popfirst!(subfigureLabels), fontsize=36) #"Double hole, 1st eigenmode, dual network, zpar=$(zpar), zperp=$(zperp)")
# Label(fig[row,col,Bottom()], L"(%$(zpar), %$(zperp))", fontsize=36) #"Double hole, 1st eigenmode, dual network, zpar=$(zpar), zperp=$(zperp)")

# First eigenmode 
# Primal network 
(row,col) = (5,1)
(zpar, zperp) = (1,0)
push!(axes, Axis(fig[row,col], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
edgeVectors = eigenvectors_Lprimal[1].*(zpar.*primalBasisParallel .+ zperp.*primalBasisPerp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ), Vec{2,Float64}.(edgeVectors), color=arrowColours, linewidth=2, lengthscale=1.0)
# Label(fig[row,col,Bottom()], popfirst!(subfigureLabels), fontsize=36) #"Double hole, 1st eigenmode, primal network, zpar=$(zpar), zperp=$(zperp)")
# Label(fig[row,col,Bottom()], L"(%$(zpar), %$(zperp))", fontsize=36) #"Double hole, 1st eigenmode, primal network, zpar=$(zpar), zperp=$(zperp)")
(row,col) = (5,2)
(zpar, zperp) = (0,1)
push!(axes, Axis(fig[row,col], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
edgeVectors = eigenvectors_Lprimal[1].*(zpar.*primalBasisParallel .+ zperp.*primalBasisPerp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ), Vec{2,Float64}.(edgeVectors), color=arrowColours, linewidth=2, lengthscale=1.0)
# Label(fig[row,col,Bottom()], popfirst!(subfigureLabels), fontsize=36) #"Double hole, 1st eigenmode, primal network, zpar=$(zpar), zperp=$(zperp)")
# Label(fig[row,col,Bottom()], L"(%$(zpar), %$(zperp))", fontsize=36) #"Double hole, 1st eigenmode, primal network, zpar=$(zpar), zperp=$(zperp)")

# Second eigenmode 
# Dual network 
(row,col) = (5,3)
(zpar, zperp) = (1,0)
push!(axes, Axis(fig[row,col], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
edgeVectors = eigenvectors_Ldual[2].*(zpar.*dualBasisParallel .+ zperp.*dualBasisPerp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐂ⱼ[boundaryEdges.==0]), Vec{2,Float64}.(edgeVectors[boundaryEdges.==0]), color=arrowColours[boundaryEdges.==0], linewidth=2, lengthscale=1.0)
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ[boundaryEdges.!=0]), Vec{2,Float64}.(edgeVectors[boundaryEdges.!=0]), color=arrowColours[boundaryEdges.!=0], linewidth=2, lengthscale=1.0)
# Label(fig[row,col,Bottom()], popfirst!(subfigureLabels), fontsize=36) 
# Label(fig[row,col,Bottom()], L"(%$(zpar), %$(zperp))", fontsize=36) #"Double hole, 2nd eigenmode, dual network, zpar=$(zpar), zperp=$(zperp)")
(row,col) = (5,4)
(zpar, zperp) = (0,1)
push!(axes, Axis(fig[row,col], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
edgeVectors = eigenvectors_Ldual[2].*(zpar.*dualBasisParallel .+ zperp.*dualBasisPerp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐂ⱼ[boundaryEdges.==0]), Vec{2,Float64}.(edgeVectors[boundaryEdges.==0]), color=arrowColours[boundaryEdges.==0], linewidth=2, lengthscale=1.0)
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ[boundaryEdges.!=0]), Vec{2,Float64}.(edgeVectors[boundaryEdges.!=0]), color=arrowColours[boundaryEdges.!=0], linewidth=2, lengthscale=1.0)
# Label(fig[row,col,Bottom()], popfirst!(subfigureLabels), fontsize=36) 
# Label(fig[row,col,Bottom()], L"(%$(zpar), %$(zperp))", fontsize=36) #"Double hole, 2nd eigenmode, dual network, zpar=$(zpar), zperp=$(zperp)")


#%%

# titleTex1 = TeXDocument(raw"""
#         \documentclass{standalone}
#         \usepackage{mathsfbf}
#         \begin{document}
#             $\mathsfbf{x}^{(m)}$
#         \end{document}
#     """
#     )
# titleTex2 = TeXDocument(raw"""
#         \documentclass{standalone}
#         \usepackage{mathsfbf}
#         \begin{document}
#             $\mathsfbf{x}^{(m)}$
#         \end{document}
#     """
#     )
# title1 = LTeX(fig[1, 1:2], titleTex1; tellheight = true, tellwidth = false)
# title1 = LTeX(fig[1, 3:4], titleTex2; tellheight = true, tellwidth = false)

# Label(fig[1,1:2], L"$\mathbf{x}^{(m)}$", fontsize=36)
# Label(fig[1,2:4], L"$\mathbf{X}^{(m)}$", fontsize=36)

Label(fig[2,1], L"(1,0)", fontsize=36)
Label(fig[2,2], L"(0,1)", fontsize=36)
Label(fig[2,3], L"(1,0)", fontsize=36)
Label(fig[2,4], L"(0,1)", fontsize=36)

Label(fig[2,0], L"(z^\parallel, z^\perp)", fontsize=36)
Label(fig[3,0], L"m=1", fontsize=36)
Label(fig[4,0], L"m=1", fontsize=36)
Label(fig[5,0], L"m=2", fontsize=36)

rowsize!(fig.layout, 1, Relative(0.04))
rowsize!(fig.layout, 2, Relative(0.04))
rowsize!(fig.layout, 3, Relative(0.31))
rowsize!(fig.layout, 4, Relative(0.32))
rowsize!(fig.layout, 5, Relative(0.31))

colsize!(fig.layout, 0, Relative(0.08))
colsize!(fig.layout, 1, Relative(0.23))
colsize!(fig.layout, 2, Relative(0.23))
colsize!(fig.layout, 3, Relative(0.23))
colsize!(fig.layout, 4, Relative(0.23))

hidedecorations!.(axes)
hidespines!.(axes)
display(fig)
save(plotsdir("edgeLaplacianHarmonicField.png"), fig)
save(plotsdir("edgeLaplacianHarmonicField.pdf"), fig)
