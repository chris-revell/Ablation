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

inputDir = "quadraticPotentialNoPressure"; isdir(plotsdir(inputDir)) ? nothing : mkdir(plotsdir(inputDir))
# inputDir = "quadraticPotentialWithPressure"; isdir(plotsdir(inputDir)) ? nothing : mkdir(plotsdir(inputDir))

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
inFile = datadir("referenceSystems", inputDir, "SingleHole_testSystem.jld2")
importedData = load(inFile)
@unpack R, A, B, F = importedData
jᵖ = findPeripheralEdges(B)
jⁱ = 1 .- jᵖ
𝐜ⱼ = findEdgeMidpoints(R, A)[jⁱ.==1]
𝐂ⱼ = findCellLinkMidpoints(R, A, B)[jⁱ.==1]
Lprimal = edgeLaplacianPrimalHat(R, A, B)
Ldual = edgeLaplacianDualHat(R, A, B)
μLprimal = [col for col in  eachcol((eigen(Matrix(Lprimal))).vectors)]
λLprimal = (eigen(Matrix(Lprimal))).values
μLdual = [col for col in  eachcol((eigen(Matrix(Ldual))).vectors)]
λLdual = (eigen(Matrix(Ldual))).values
cellPolygons = findCellPolygons(R, A, B)
𝐞Parallel = findEdgeTangents(R, A)[jⁱ.==1]./(findEdgeLengths(R, A)[jⁱ.==1].^2)
𝐞Perp = [ϵᵢ*v for v in 𝐞Parallel]
𝐄Parallel = findCellLinks(R, A, B)[jⁱ.==1]./(findCellLinkLengths(R, A, B)[jⁱ.==1].^2)
𝐄Perp = [ϵₖ*v for v in 𝐄Parallel]

# Primal network 
(row,col) = (2,1)
(zpar, zperp) = (1,0)
push!(axes, Axis(fig[row,col], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
edgeVectors = μLprimal[1].*(zpar.*𝐞Parallel .+ zperp.*𝐞Perp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ), Vec{2,Float64}.(edgeVectors), color=arrowColours, linewidth=2, lengthscale=1.0)
# Label(fig[row,col,Bottom()], popfirst!(subfigureLabels), fontsize=36) #"Single hole, 1st eigenmode, primal network, zpar=$(zpar), zperp=$(zperp)")
# Label(fig[row,col,Bottom()], L"(%$(zpar), %$(zperp))", fontsize=36) #"Single hole, 1st eigenmode, primal network, zpar=$(zpar), zperp=$(zperp)")
(row,col) = (2,2)
(zpar, zperp) = (0,1)
push!(axes, Axis(fig[row,col], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
edgeVectors = μLprimal[1].*(zpar.*𝐞Parallel .+ zperp.*𝐞Perp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ), Vec{2,Float64}.(edgeVectors), color=arrowColours, linewidth=2, lengthscale=1.0)
# Label(fig[row,col,Bottom()], popfirst!(subfigureLabels), fontsize=36) #"Single hole, 1st eigenmode, primal network, zpar=$(zpar), zperp=$(zperp)")
# Label(fig[row,col,Bottom()], L"(%$(zpar), %$(zperp))", fontsize=36) #"Single hole, 1st eigenmode, primal network, zpar=$(zpar), zperp=$(zperp)")

# Dual network 
(row,col) = (2,4)
(zpar, zperp) = (1,0)
push!(axes, Axis(fig[row,col], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
edgeVectors = μLdual[1].*(zpar.*𝐄Parallel .+ zperp.*𝐄Perp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐂ⱼ), Vec{2,Float64}.(edgeVectors), color=arrowColours, linewidth=2, lengthscale=1.0)
# arrows!(axes[end], Point{2,Float64}.(𝐂ⱼ[jᵖ.==0]), Vec{2,Float64}.(edgeVectors[jᵖ.==0]), color=arrowColours[jᵖ.==0], linewidth=2, lengthscale=1.0)
# arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ[jᵖ.!=0]), Vec{2,Float64}.(edgeVectors[jᵖ.!=0]), color=arrowColours[jᵖ.!=0], linewidth=2, lengthscale=1.0)
# Label(fig[row,col,Bottom()], popfirst!(subfigureLabels), fontsize=36) #"Single hole, 1st eigenmode, dual network, zpar=$(zpar), zperp=$(zperp)")
# Label(fig[row,col,Bottom()], L"(%$(zpar), %$(zperp))", fontsize=36) #"Single hole, 1st eigenmode, dual network, zpar=$(zpar), zperp=$(zperp)")
(row,col) = (2,5)
(zpar, zperp) = (0,1)
push!(axes, Axis(fig[row,col], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
edgeVectors = μLdual[1].*(zpar.*𝐄Parallel .+ zperp.*𝐄Perp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐂ⱼ), Vec{2,Float64}.(edgeVectors), color=arrowColours, linewidth=2, lengthscale=1.0)
# arrows!(axes[end], Point{2,Float64}.(𝐂ⱼ[jᵖ.==0]), Vec{2,Float64}.(edgeVectors[jᵖ.==0]), color=arrowColours[jᵖ.==0], linewidth=2, lengthscale=1.0)
# arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ[jᵖ.!=0]), Vec{2,Float64}.(edgeVectors[jᵖ.!=0]), color=arrowColours[jᵖ.!=0], linewidth=2, lengthscale=1.0)
# Label(fig[row,col,Bottom()], popfirst!(subfigureLabels), fontsize=36) #"Single hole, 1st eigenmode, dual network, zpar=$(zpar), zperp=$(zperp)")
# Label(fig[row,col,Bottom()], L"(%$(zpar), %$(zperp))", fontsize=36) #"Single hole, 1st eigenmode, dual network, zpar=$(zpar), zperp=$(zperp)")

# Double hole 
inFile = datadir("referenceSystems", inputDir, "DoubleHole_testSystem.jld2")
importedData = load(inFile)
@unpack R, A, B, F = importedData
jᵖ = findPeripheralEdges(B)
jⁱ = 1 .- jᵖ
𝐜ⱼ = findEdgeMidpoints(R, A)[jⁱ.==1]
𝐂ⱼ = findCellLinkMidpoints(R, A, B)[jⁱ.==1]
Lprimal = edgeLaplacianPrimalHat(R, A, B)
Ldual = edgeLaplacianDualHat(R, A, B)
μLprimal = [col for col in  eachcol((eigen(Matrix(Lprimal))).vectors)]
λLprimal = (eigen(Matrix(Lprimal))).values
μLdual = [col for col in  eachcol((eigen(Matrix(Ldual))).vectors)]
λLdual = (eigen(Matrix(Ldual))).values
cellPolygons = findCellPolygons(R, A, B)
𝐞Parallel = findEdgeTangents(R, A)[jⁱ.==1]./(findEdgeLengths(R, A)[jⁱ.==1].^2)
𝐞Perp = [ϵᵢ*v for v in 𝐞Parallel]
𝐄Parallel = findCellLinks(R, A, B)[jⁱ.==1]./(findCellLinkLengths(R, A, B)[jⁱ.==1].^2)
𝐄Perp = [ϵₖ*v for v in 𝐄Parallel]

# Second eigenmode 
# Primal network 
(row,col) = (3,1)
(zpar, zperp) = (1,0)
push!(axes, Axis(fig[row,col], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
edgeVectors = μLprimal[2].*(zpar.*𝐞Parallel .+ zperp.*𝐞Perp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ), Vec{2,Float64}.(edgeVectors), color=arrowColours, linewidth=2, lengthscale=1.0)
(row,col) = (3,2)
(zpar, zperp) = (0,1)
push!(axes, Axis(fig[row,col], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
edgeVectors = μLprimal[2].*(zpar.*𝐞Parallel .+ zperp.*𝐞Perp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ), Vec{2,Float64}.(edgeVectors), color=arrowColours, linewidth=2, lengthscale=1.0)
# First eigenmode 
# Dual network 
(row,col) = (3,4)
(zpar, zperp) = (1,0)
push!(axes, Axis(fig[row,col], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
edgeVectors = μLdual[1].*(zpar.*𝐄Parallel .+ zperp.*𝐄Perp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐂ⱼ), Vec{2,Float64}.(edgeVectors), color=arrowColours, linewidth=2, lengthscale=1.0)
(row,col) = (3,5)
(zpar, zperp) = (0,1)
push!(axes, Axis(fig[row,col], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
edgeVectors = μLdual[1].*(zpar.*𝐄Parallel .+ zperp.*𝐄Perp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐂ⱼ), Vec{2,Float64}.(edgeVectors), color=arrowColours, linewidth=2, lengthscale=1.0)
# First eigenmode 
# Primal network 
(row,col) = (4,1)
(zpar, zperp) = (1,0)
push!(axes, Axis(fig[row,col], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
edgeVectors = μLprimal[1].*(zpar.*𝐞Parallel .+ zperp.*𝐞Perp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ), Vec{2,Float64}.(edgeVectors), color=arrowColours, linewidth=2, lengthscale=1.0)
(row,col) = (4,2)
(zpar, zperp) = (0,1)
push!(axes, Axis(fig[row,col], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
edgeVectors = μLprimal[1].*(zpar.*𝐞Parallel .+ zperp.*𝐞Perp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ), Vec{2,Float64}.(edgeVectors), color=arrowColours, linewidth=2, lengthscale=1.0)
# Second eigenmode 
# Dual network 
(row,col) = (4,4)
(zpar, zperp) = (1,0)
push!(axes, Axis(fig[row,col], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
edgeVectors = μLdual[2].*(zpar.*𝐄Parallel .+ zperp.*𝐄Perp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐂ⱼ), Vec{2,Float64}.(edgeVectors), color=arrowColours, linewidth=2, lengthscale=1.0)
(row,col) = (4,5)
(zpar, zperp) = (0,1)
push!(axes, Axis(fig[row,col], aspect=DataAspect()))
poly!.(axes[end], cellPolygons, color=(:black,0.1), strokecolor=(:black, 0.2), strokewidth=1)
edgeVectors = μLdual[2].*(zpar.*𝐄Parallel .+ zperp.*𝐄Perp)
arrowColours = [(:blue, norm(v)/maximum(norm.(edgeVectors))) for v in edgeVectors]
arrows!(axes[end], Point{2,Float64}.(𝐂ⱼ), Vec{2,Float64}.(edgeVectors), color=arrowColours, linewidth=2, lengthscale=1.0)

push!(axes, Axis(fig[:,3]))
vlines!(axes[end], [0.0], color=(:black,0.3), linewidth=5, linestyle=:solid)


Label(fig[0,1:2], "Primal", fontsize=48)
Label(fig[0,4:5], "Dual", fontsize=48)

Label(fig[1,1], L"(1,0)", fontsize=36)
Label(fig[1,2], L"(0,1)", fontsize=36)
Label(fig[1,4], L"(1,0)", fontsize=36)
Label(fig[1,5], L"(0,1)", fontsize=36)

Label(fig[1,0], L"\{z^{\parallel(m)}, z^{\perp(m)}\}", fontsize=36)
Label(fig[2,0], L"m=1", fontsize=36)
Label(fig[3,0], L"m=1", fontsize=36)
Label(fig[4,0], L"m=2", fontsize=36)

rowsize!(fig.layout, 0, Relative(0.04))
rowsize!(fig.layout, 1, Relative(0.03))
rowsize!(fig.layout, 2, Relative(0.31))
rowsize!(fig.layout, 3, Relative(0.31))
rowsize!(fig.layout, 4, Relative(0.31))

colsize!(fig.layout, 0, Relative(0.115))
colsize!(fig.layout, 1, Relative(0.22))
colsize!(fig.layout, 2, Relative(0.22))
colsize!(fig.layout, 3, Relative(0.005))
colsize!(fig.layout, 4, Relative(0.22))
colsize!(fig.layout, 5, Relative(0.22))

hidedecorations!.(axes)
hidespines!.(axes)
display(fig)
save(plotsdir(inputDir, "edgeLaplacianHarmonicFieldVectors.png"), fig)
save(plotsdir(inputDir, "edgeLaplacianHarmonicFieldVectors.pdf"), fig)
