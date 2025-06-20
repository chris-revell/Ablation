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

inFile = datadir("referenceSystems", "SingleHole_testSystem.jld2")
importedData = load(inFile)
@unpack R, A, B, F = importedData

fig = Figure(size=(1000, 1500))
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"($l)" for l in individualLetters]

push!(axes, Axis(fig[1,1]))
λᵛ = (eigen(Matrix(geometricLf(R, A, B)))).values
@show minimum(λᵛ)
scatter!(axes[end], collect(1:length(λᵛ)), λᵛ)
Label(fig[1,1,Bottom()], L"L_tf, spectrum",fontsize=24)

push!(axes, Axis(fig[1,2]))
λᵛ = (eigen(Matrix(geometricLc(R, A, B)))).values
@show minimum(λᵛ)
scatter!(axes[end], collect(1:length(λᵛ)), λᵛ)
Label(fig[1,2,Bottom()], L"L_c\, spectrum",fontsize=24)

push!(axes, Axis(fig[2,1]))
λᵛ = (eigen(Matrix(geometricLv(R, A, B)))).values
@show minimum(λᵛ)
scatter!(axes[end], collect(1:length(λᵛ)), λᵛ)
Label(fig[2,1,Bottom()], L"L_v\, spectrum",fontsize=24)

push!(axes, Axis(fig[2,2]))
λᵛ = (eigen(Matrix(geometricLt(R, A, B)))).values
@show minimum(λᵛ)
scatter!(axes[end], collect(1:length(λᵛ)), λᵛ)
Label(fig[2,2,Bottom()], L"L_t\, spectrum",fontsize=24)

push!(axes, Axis(fig[3,1]))
λᵛ = (eigen(Matrix(edgeLaplacianPrimal(R, A, B)))).values
@show minimum(λᵛ)
scatter!(axes[end], collect(1:length(λᵛ)), λᵛ)
Label(fig[3,1,Bottom()], L"edgeLaplacianPrimal\, spectrum",fontsize=24)

push!(axes, Axis(fig[3,2]))
λᵛ = (eigen(Matrix(edgeLaplacianDual(R, A, B)))).values
@show minimum(λᵛ)
scatter!(axes[end], collect(1:length(λᵛ)), λᵛ)
Label(fig[3,2,Bottom()], L"edgeLaplacianDual\, spectrum",fontsize=24)


display(fig)

save("laplacianspectra.png", fig)