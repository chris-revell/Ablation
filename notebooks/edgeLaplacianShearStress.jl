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

@from "$(srcdir("Stresses.jl"))" using Stresses

inputDir = "quadraticPotentialNoPressure"; isdir(plotsdir(inputDir)) ? nothing : mkdir(plotsdir(inputDir))
# inputDir = "quadraticPotentialWithPressure"; isdir(plotsdir(inputDir)) ? nothing : mkdir(plotsdir(inputDir))

zpar = 0.0
zperp = 2.0
ϵᵢ = SMatrix{2, 2, Float64}([
                0.0 1.0
                -1.0 0.0
            ])
ϵₖ = SMatrix{2, 2, Float64}([
                0.0 -1.0
                1.0 0.0
            ])

fig = Figure(size=(1500,500), fontsize=36)
axes = Axis[]

# function ζ(R, A, B, m, zperp)
#     Lprimal = edgeLaplacianPrimal(R, A, B)
#     w = transpose((eigen(Matrix(Lprimal))).vectors)
#     𝐭̂ = normalize.(findEdgeTangents(R, A))
#     ζᵐᵢ = zeros(size(B,1))
#     for i=1:size(B,1)
#        tmp = sum([B[i,j].*w[m, j].*(𝐭̂[j]*𝐭̂[j]') for j=1:size(B,2)])
#        tmpDet = det(tmp)
#        ζᵐᵢ[i] = zperp*sqrt(-tmpDet)
#     end
#     return ζᵐᵢ
# end 
    

# Single hole 
inFile = datadir("referenceSystems", inputDir, "SingleHole_testSystem.jld2")
importedData = load(inFile)
@unpack R, A, B, F = importedData
cellPolygons = findCellPolygons(R, A, B)
# 1st eigenmode
ζᵢ = ζ(R, A, B, 1, zperp)
clims = (0.0, maximum(abs.(ζᵢ)))
push!(axes, Axis(fig[1,1], aspect=DataAspect()))
for i=1:size(B,1)
    poly!(axes[end], cellPolygons[i], color=ζᵢ[i], colorrange = clims, colormap = Reverse(:devon), strokecolor=(:black, 1.0), strokewidth=2)
end
Colorbar(fig[1,2], colorrange=clims, colormap=Reverse(:devon), height=Relative(0.8), label=L"\zeta_i^{(1)}")
# Label(fig[1,1,Bottom()], "1st eigenmode")
Label(fig[1,1,Bottom()], L"(a)", fontsize=36)

# Double hole 
inFile = datadir("referenceSystems", inputDir, "DoubleHole_testSystem.jld2")
importedData = load(inFile)
@unpack R, A, B, F = importedData
cellPolygons = findCellPolygons(R, A, B)
# 1st eigenmode
ζᵢ = ζ(R, A, B, 1, zperp)
clims = (0.0, maximum(abs.(ζᵢ)))
push!(axes, Axis(fig[1,3], aspect=DataAspect()))
for i=1:size(B,1)
    poly!(axes[end], cellPolygons[i], color=ζᵢ[i], colorrange = clims, colormap = Reverse(:devon), strokecolor=(:black, 1.0), strokewidth=2)
end
Colorbar(fig[1,4], colorrange=clims, colormap=Reverse(:devon), height=Relative(0.8), label=L"\zeta_i^{(1)}")
# Label(fig[1,3,Bottom()], "1st eigenmode")
Label(fig[1,3,Bottom()], L"(b)", fontsize=36)
# 2nd eigenmode
ζᵢ = ζ(R, A, B, 2, zperp)
clims = (0.0, maximum(abs.(ζᵢ)))
push!(axes, Axis(fig[1,5], aspect=DataAspect()))
for i=1:size(B,1)
    poly!(axes[end], cellPolygons[i], color=ζᵢ[i], colorrange = clims, colormap = Reverse(:devon), strokecolor=(:black, 1.0), strokewidth=2)
end
Colorbar(fig[1,6], colorrange=clims, colormap=Reverse(:devon), height=Relative(0.8), label=L"\zeta_i^{(2)}")
# Label(fig[1,5,Bottom()], "2nd eigenmode")
Label(fig[1,5,Bottom()], L"(c)", fontsize=36)


hidedecorations!.(axes)
hidespines!.(axes)
display(fig)
save(plotsdir(inputDir, "edgeLaplacianShearStress_zperp=$(zperp).png"), fig)
save(plotsdir(inputDir, "edgeLaplacianShearStress_zperp=$(zperp).pdf"), fig)



