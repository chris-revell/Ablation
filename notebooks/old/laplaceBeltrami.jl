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

@from "$(srcdir("PenrosePseudoInversion.jl"))" using PenrosePseudoInversion

inputSystems = ["NoHole", "SingleHole", "DoubleHole", "Voronoi"]#, "OldSystem"]

allOutputs = Dict()

for inputSystem in inputSystems

    inFile = datadir("referenceSystems", "$(inputSystem)_testSystem.jld2")
    # @show inFile
    importedData = load(inFile)
    @unpack R, A, B, F = importedData

    Lprimal = edgeLaplacianPrimal(R, A, B)
    Ldual = edgeLaplacianDual(R, A, B)

    eigenvectors_Lprimal = [col for col in  eachcol((eigen(Matrix(Lprimal))).vectors)]
    eigenvalues_Lprimal = (eigen(Matrix(Lprimal))).values
    eigenvectors_Ldual = [col for col in  eachcol((eigen(Matrix(Ldual))).vectors)]
    eigenvalues_Ldual = (eigen(Matrix(Ldual))).values

    allOutputs["$(inputSystem)_Lprimal"] = Lprimal
    allOutputs["$(inputSystem)_eigenvectors_Lprimal"] = eigenvectors_Lprimal
    allOutputs["$(inputSystem)_eigenvalues_Lprimal"] = eigenvalues_Lprimal
    allOutputs["$(inputSystem)_Ldual"] = Ldual
    allOutputs["$(inputSystem)_eigenvectors_Ldual"] = eigenvectors_Ldual
    allOutputs["$(inputSystem)_eigenvalues_Ldual"] = eigenvalues_Ldual
    allOutputs["$(inputSystem)_R"] = R
    allOutputs["$(inputSystem)_A"] = A
    allOutputs["$(inputSystem)_B"] = B
    allOutputs["$(inputSystem)_cellPolygons"] = findCellPolygons(R, A, B)
    allOutputs["$(inputSystem)_edgeQuadrilaterals"] = findEdgeQuadrilaterals(R, A, B)

    # @show inputSystem
    # innerProdsLprimal = [eigenvectors_Lprimal[i-1]'*Tₑ⁻¹*eigenvectors_Lprimal[i] for i=2:length(eigenvectors_Lprimal)]
    # @show findmax(innerProdsLprimal)
    # @show eigenvalues_Lprimal[1:2]
    # innerProdsLdual = [eigenvectors_Ldual[i-1]'*Tₗ⁻¹*eigenvectors_Ldual[i] for i=2:length(eigenvectors_Ldual)]
    # @show findmax(innerProdsLdual)
    # @show eigenvalues_Ldual[1:2]

    # x, xSpectrum = penrosePseudoInversion(Lprimal, zeros(size(Lprimal,1)), spdiagm(Fⱼ))
    # @show x
end


#%%

fig = Figure(size=(1000,1500))
axes = Axis[]

limsSingleHolePrimal1 = (-maximum(norm.(allOutputs["SingleHole_eigenvectors_Lprimal"][1])), maximum(norm.(allOutputs["SingleHole_eigenvectors_Lprimal"][1])))
limsSingleHoleDual1 = (-maximum(norm.(allOutputs["SingleHole_eigenvectors_Ldual"][1])), maximum(norm.(allOutputs["SingleHole_eigenvectors_Ldual"][1])))
limsDoubleHolePrimal1 = (-maximum(norm.(allOutputs["DoubleHole_eigenvectors_Lprimal"][1])), maximum(norm.(allOutputs["DoubleHole_eigenvectors_Lprimal"][1])))
limsDoubleHolePrimal2 = (-maximum(norm.(allOutputs["DoubleHole_eigenvectors_Lprimal"][2])), maximum(norm.(allOutputs["DoubleHole_eigenvectors_Lprimal"][2])))
limsDoubleHoleDual1 = (-maximum(norm.(allOutputs["DoubleHole_eigenvectors_Ldual"][1])), maximum(norm.(allOutputs["DoubleHole_eigenvectors_Ldual"][1])))
limsDoubleHoleDual2 = (-maximum(norm.(allOutputs["DoubleHole_eigenvectors_Ldual"][2])), maximum(norm.(allOutputs["DoubleHole_eigenvectors_Ldual"][2])))

push!(axes, Axis(fig[1,1], aspect=DataAspect()))
for j=1:size(allOutputs["SingleHole_B"],2)
    poly!(axes[end], allOutputs["SingleHole_edgeQuadrilaterals"][j], color=allOutputs["SingleHole_eigenvectors_Lprimal"][1][j], colormap=:bwr, colorrange=limsSingleHolePrimal1, strokecolor=(:black, 0.0), strokewidth=2)
end
for i=1:size(allOutputs["SingleHole_B"],1)
    poly!(axes[end], allOutputs["SingleHole_cellPolygons"][i], color=(:white,0.0), strokecolor=(:black, 1.0), strokewidth=2)
end
Label(fig[1,1,Bottom()], "Single hole, 1st eigenmode, primal network")
Colorbar(fig[1,2], colormap=:bwr, colorrange=limsSingleHolePrimal1)

push!(axes, Axis(fig[1,3], aspect=DataAspect()))
for j=1:size(allOutputs["SingleHole_B"],2)
    poly!(axes[end], allOutputs["SingleHole_edgeQuadrilaterals"][j], color=allOutputs["SingleHole_eigenvectors_Ldual"][1][j], colormap=:bwr, colorrange=limsSingleHoleDual1, strokecolor=(:black, 0.0), strokewidth=2)
end
for i=1:size(allOutputs["SingleHole_B"],1)
    poly!(axes[end], allOutputs["SingleHole_cellPolygons"][i], color=(:white,0.0), strokecolor=(:black, 1.0), strokewidth=2)
end
Label(fig[1,3,Bottom()], "Single hole, 1st eigenmode, dual network")
Colorbar(fig[1,4], colormap=:bwr, colorrange=limsSingleHoleDual1)

push!(axes, Axis(fig[2,1], aspect=DataAspect()))
for j=1:size(allOutputs["DoubleHole_B"],2)
    poly!(axes[end], allOutputs["DoubleHole_edgeQuadrilaterals"][j], color=allOutputs["DoubleHole_eigenvectors_Lprimal"][1][j], colormap=:bwr, colorrange=limsDoubleHolePrimal1, strokecolor=(:black, 0.0), strokewidth=2)
end
for i=1:size(allOutputs["DoubleHole_B"],1)
    poly!(axes[end], allOutputs["DoubleHole_cellPolygons"][i], color=(:white,0.0), strokecolor=(:black, 1.0), strokewidth=2)
end
Label(fig[2,1,Bottom()], "Double hole, 1st eigenmode, primary network")
Colorbar(fig[2,2], colormap=:bwr, colorrange=limsDoubleHolePrimal1)

push!(axes, Axis(fig[2,3], aspect=DataAspect()))
for j=1:size(allOutputs["DoubleHole_B"],2)
    poly!(axes[end], allOutputs["DoubleHole_edgeQuadrilaterals"][j], color=allOutputs["DoubleHole_eigenvectors_Ldual"][1][j], colormap=:bwr, colorrange=limsDoubleHoleDual1, strokecolor=(:black, 0.0), strokewidth=2)
end
for i=1:size(allOutputs["DoubleHole_B"],1)
    poly!(axes[end], allOutputs["DoubleHole_cellPolygons"][i], color=(:white,0.0), strokecolor=(:black, 1.0), strokewidth=2)
end
Label(fig[2,3,Bottom()], "Double hole, 1st eigenmode, dual network")
Colorbar(fig[2,4], colormap=:bwr, colorrange=limsDoubleHoleDual1)

push!(axes, Axis(fig[3,1], aspect=DataAspect()))
for j=1:size(allOutputs["DoubleHole_B"],2)
    poly!(axes[end], allOutputs["DoubleHole_edgeQuadrilaterals"][j], color=allOutputs["DoubleHole_eigenvectors_Lprimal"][2][j], colormap=:bwr, colorrange=limsDoubleHolePrimal2, strokecolor=(:black, 0.0), strokewidth=2)
end
for i=1:size(allOutputs["DoubleHole_B"],1)
    poly!(axes[end], allOutputs["DoubleHole_cellPolygons"][i], color=(:white,0.0), strokecolor=(:black, 1.0), strokewidth=2)
end
Label(fig[3,1,Bottom()], "Double hole, 2nd eigenmode, primary network")
Colorbar(fig[3,2], colormap=:bwr, colorrange=limsDoubleHolePrimal2)

push!(axes, Axis(fig[3,3], aspect=DataAspect()))
for j=1:size(allOutputs["DoubleHole_B"],2)
    poly!(axes[end], allOutputs["DoubleHole_edgeQuadrilaterals"][j], color=allOutputs["DoubleHole_eigenvectors_Ldual"][2][j], colormap=:bwr, colorrange=limsDoubleHoleDual2, strokecolor=(:black, 0.0), strokewidth=2)
end
for i=1:size(allOutputs["DoubleHole_B"],1)
    poly!(axes[end], allOutputs["DoubleHole_cellPolygons"][i], color=(:white,0.0), strokecolor=(:black, 1.0), strokewidth=2)
end
Label(fig[3,3,Bottom()], "Double hole, 2nd eigenmode, dual network")
Colorbar(fig[3,4], colormap=:bwr, colorrange=limsDoubleHoleDual2)

hidedecorations!.(axes)
hidespines!.(axes)

display(fig)

save(datadir("edgeLaplacianEigenvectors.png"), fig)



#%%

inVec = [[a, a] for a in allOutputs["SingleHole_eigenvectors_Ldual"][1]]

divᵛb = divᵛ(R, A, B, inVec)
@show maximum(abs.(divᵛb))
curlᶜb = curlᶜ(R, A, B, inVec)
@show maximum(abs.(curlᶜb))