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

@from "$(srcdir("SingularValueDecomposition.jl"))" using SingularValueDecomposition

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
    allOutputs["$(inputSystem)_edgeMidpoints"] = findEdgeMidpoints(R, A)
    allOutputs["$(inputSystem)_edgeTangents"] = findEdgeTangents(R, A)./(findEdgeLengths(R, A).^2)
end


#%%

α = 0.0
β = 10.0
ϵᵢ = SMatrix{2, 2, Float64}([
                0.0 1.0
                -1.0 0.0
            ])

fig = Figure(size=(1000,1500))
axes = Axis[]

limsSingleHolePrimal1 = (-maximum(norm.(allOutputs["SingleHole_eigenvectors_Lprimal"][1])), maximum(norm.(allOutputs["SingleHole_eigenvectors_Lprimal"][1])))
limsSingleHoleDual1 = (-maximum(norm.(allOutputs["SingleHole_eigenvectors_Ldual"][1])), maximum(norm.(allOutputs["SingleHole_eigenvectors_Ldual"][1])))
limsDoubleHolePrimal1 = (-maximum(norm.(allOutputs["DoubleHole_eigenvectors_Lprimal"][1])), maximum(norm.(allOutputs["DoubleHole_eigenvectors_Lprimal"][1])))
limsDoubleHolePrimal2 = (-maximum(norm.(allOutputs["DoubleHole_eigenvectors_Lprimal"][2])), maximum(norm.(allOutputs["DoubleHole_eigenvectors_Lprimal"][2])))
limsDoubleHoleDual1 = (-maximum(norm.(allOutputs["DoubleHole_eigenvectors_Ldual"][1])), maximum(norm.(allOutputs["DoubleHole_eigenvectors_Ldual"][1])))
limsDoubleHoleDual2 = (-maximum(norm.(allOutputs["DoubleHole_eigenvectors_Ldual"][2])), maximum(norm.(allOutputs["DoubleHole_eigenvectors_Ldual"][2])))

push!(axes, Axis(fig[1,1], aspect=DataAspect()))
for i=1:size(allOutputs["SingleHole_B"],1)
    poly!(axes[end], allOutputs["SingleHole_cellPolygons"][i], color=(:black,0.1), strokecolor=(:black, 0.5), strokewidth=1)
end
edgeVectorsSingleHolePrimal = α.*allOutputs["SingleHole_edgeTangents"] .+ β.*[ϵᵢ*v for v in allOutputs["SingleHole_edgeTangents"]]
edgeVectorsSingleHolePrimal .*= allOutputs["SingleHole_eigenvectors_Lprimal"][1]
arrows2d!(axes[end], Point{2,Float64}.(allOutputs["SingleHole_edgeMidpoints"]), Vec{2,Float64}.(edgeVectorsSingleHolePrimal), color=norm(edgeVectorsSingleHolePrimal), colormap=:inferno, linewidth=2)
Label(fig[1,1,Bottom()], "Single hole, 1st eigenmode, primal network")
Colorbar(fig[1,2], colormap=:bwr, colorrange=limsSingleHolePrimal1)

push!(axes, Axis(fig[1,3], aspect=DataAspect()))
for i=1:size(allOutputs["SingleHole_B"],1)
    poly!(axes[end], allOutputs["SingleHole_cellPolygons"][i], color=(:black,0.1), strokecolor=(:black, 0.5), strokewidth=1)
end
edgeVectorsSingleHoleDual = α.*allOutputs["SingleHole_edgeTangents"] .+ β.*[ϵᵢ*v for v in allOutputs["SingleHole_edgeTangents"]]
edgeVectorsSingleHoleDual .*= allOutputs["SingleHole_eigenvectors_Ldual"][1]
arrows2d!(axes[end], Point{2,Float64}.(allOutputs["SingleHole_edgeMidpoints"]), Vec{2,Float64}.(edgeVectorsSingleHoleDual), color=norm(edgeVectorsSingleHoleDual), colormap=:inferno, linewidth=2)
Label(fig[1,3,Bottom()], "Single hole, 1st eigenmode, dual network")
Colorbar(fig[1,4], colormap=:bwr, colorrange=limsSingleHoleDual1)

push!(axes, Axis(fig[2,1], aspect=DataAspect()))
for i=1:size(allOutputs["DoubleHole_B"],1)
    poly!(axes[end], allOutputs["DoubleHole_cellPolygons"][i], color=(:black,0.1), strokecolor=(:black, 0.5), strokewidth=1)
end
edgeVectorsDoubleHolePrimal1 = α.*allOutputs["DoubleHole_edgeTangents"] .+ β.*[ϵᵢ*v for v in allOutputs["DoubleHole_edgeTangents"]]
edgeVectorsDoubleHolePrimal1 .*= allOutputs["DoubleHole_eigenvectors_Lprimal"][1]
arrows2d!(axes[end], Point{2,Float64}.(allOutputs["DoubleHole_edgeMidpoints"]), Vec{2,Float64}.(edgeVectorsDoubleHolePrimal1), color=norm(edgeVectorsDoubleHolePrimal1), colormap=:inferno, linewidth=2)
Label(fig[2,1,Bottom()], "Double hole, 1st eigenmode, primary network")
Colorbar(fig[2,2], colormap=:bwr, colorrange=limsDoubleHolePrimal1)

push!(axes, Axis(fig[2,3], aspect=DataAspect()))
for i=1:size(allOutputs["DoubleHole_B"],1)
    poly!(axes[end], allOutputs["DoubleHole_cellPolygons"][i], color=(:black,0.1), strokecolor=(:black, 0.5), strokewidth=1)
end
edgeVectorsDoubleHoleDual1 = α.*allOutputs["DoubleHole_edgeTangents"] .+ β.*[ϵᵢ*v for v in allOutputs["DoubleHole_edgeTangents"]]
edgeVectorsDoubleHoleDual1 .*= allOutputs["DoubleHole_eigenvectors_Ldual"][1]
arrows2d!(axes[end], Point{2,Float64}.(allOutputs["DoubleHole_edgeMidpoints"]), Vec{2,Float64}.(edgeVectorsDoubleHoleDual1), color=norm(edgeVectorsDoubleHoleDual1), colormap=:inferno, linewidth=2)
Label(fig[2,3,Bottom()], "Double hole, 1st eigenmode, dual network")
Colorbar(fig[2,4], colormap=:bwr, colorrange=limsDoubleHoleDual1)

push!(axes, Axis(fig[3,1], aspect=DataAspect()))
for i=1:size(allOutputs["DoubleHole_B"],1)
    poly!(axes[end], allOutputs["DoubleHole_cellPolygons"][i], color=(:black,0.1), strokecolor=(:black, 0.5), strokewidth=1)
end
edgeVectorsDoubleHolePrimal2 = α.*allOutputs["DoubleHole_edgeTangents"] .+ β.*[ϵᵢ*v for v in allOutputs["DoubleHole_edgeTangents"]]
edgeVectorsDoubleHolePrimal2 .*= allOutputs["DoubleHole_eigenvectors_Lprimal"][2]
arrows2d!(axes[end], Point{2,Float64}.(allOutputs["DoubleHole_edgeMidpoints"]), Vec{2,Float64}.(edgeVectorsDoubleHolePrimal2), color=norm(edgeVectorsDoubleHolePrimal2), colormap=:inferno, linewidth=2)
Label(fig[3,1,Bottom()], "Double hole, 2nd eigenmode, primary network")
Colorbar(fig[3,2], colormap=:bwr, colorrange=limsDoubleHolePrimal2)

push!(axes, Axis(fig[3,3], aspect=DataAspect()))
for i=1:size(allOutputs["DoubleHole_B"],1)
    poly!(axes[end], allOutputs["DoubleHole_cellPolygons"][i], color=(:black,0.1), strokecolor=(:black, 0.5), strokewidth=1)
end
edgeVectorsDoubleHoleDual2 = α.*allOutputs["DoubleHole_edgeTangents"] .+ β.*[ϵᵢ*v for v in allOutputs["DoubleHole_edgeTangents"]]
edgeVectorsDoubleHoleDual2 .*= allOutputs["DoubleHole_eigenvectors_Ldual"][2]
arrows2d!(axes[end], Point{2,Float64}.(allOutputs["DoubleHole_edgeMidpoints"]), Vec{2,Float64}.(edgeVectorsDoubleHoleDual2), color=norm(edgeVectorsDoubleHoleDual2), colormap=:inferno, linewidth=2)
Label(fig[3,3,Bottom()], "Double hole, 2nd eigenmode, dual network")
Colorbar(fig[3,4], colormap=:bwr, colorrange=limsDoubleHoleDual2)

hidedecorations!.(axes)
hidespines!.(axes)

display(fig)

save(datadir("edgeLaplacianEigenvectors2.png"), fig)



#%%

# divᵛb = divᵛ(allOutputs["SingleHole_R"], allOutputs["SingleHole_A"], allOutputs["SingleHole_B"], edgeVectorsSingleHolePrimal)
# @show maximum(abs.(divᵛb))
# curlᶜb = curlᶜ(allOutputs["SingleHole_R"], allOutputs["SingleHole_A"], allOutputs["SingleHole_B"], edgeVectorsSingleHolePrimal)
# @show maximum(abs.(curlᶜb))
# codᵛb = codᵛ(allOutputs["SingleHole_R"], allOutputs["SingleHole_A"], allOutputs["SingleHole_B"], edgeVectorsSingleHolePrimal)
# @show maximum(abs.(codᵛb))
# cocurlᶜb = cocurlᶜ(allOutputs["SingleHole_R"], allOutputs["SingleHole_A"], allOutputs["SingleHole_B"], edgeVectorsSingleHolePrimal)
# @show maximum(abs.(cocurlᶜb))



# divᵛb = divᵛ(allOutputs["DoubleHole_R"], allOutputs["DoubleHole_A"], allOutputs["DoubleHole_B"], edgeVectorsDoubleHolePrimal1)
# @show maximum(abs.(divᵛb))
# curlᶜb = curlᶜ(allOutputs["DoubleHole_R"], allOutputs["DoubleHole_A"], allOutputs["DoubleHole_B"], edgeVectorsDoubleHolePrimal1)
# @show maximum(abs.(curlᶜb))
# codᵛb = codᵛ(allOutputs["DoubleHole_R"], allOutputs["DoubleHole_A"], allOutputs["DoubleHole_B"], edgeVectorsDoubleHolePrimal1)
# @show maximum(abs.(codᵛb))
# cocurlᶜb = cocurlᶜ(allOutputs["DoubleHole_R"], allOutputs["DoubleHole_A"], allOutputs["DoubleHole_B"], edgeVectorsDoubleHolePrimal1)
# @show maximum(abs.(cocurlᶜb))


# fig = Figure(size=(1000,1000))
# ax = Axis(fig[1,1], aspect=DataAspect())
# clims = (-maximum(abs.(codᵛb)), maximum(abs.(codᵛb)))
# linkTriangles = findCellLinkTriangles(allOutputs["DoubleHole_R"], allOutputs["DoubleHole_A"], allOutputs["DoubleHole_B"])
# for k=1:size(allOutputs["DoubleHole_A"], 2)
#     poly!(ax, linkTriangles[k], color=codᵛb[k], colorrange=clims, colormap=:bwr, strokecolor=(:black, 1.0), strokewidth=2)
# end
# for i=1:size(allOutputs["DoubleHole_B"], 1)
#     poly!(ax, allOutputs["DoubleHole_cellPolygons"][i], color=(:white, 0.0), strokecolor=(:black, 1.0), strokewidth=2)
# end
# display(fig)
    
    
    


# divᵛb = divᵛ(allOutputs["DoubleHole_R"], allOutputs["DoubleHole_A"], allOutputs["DoubleHole_B"], edgeVectorsDoubleHolePrimal2)
# @show maximum(abs.(divᵛb))
# curlᶜb = curlᶜ(allOutputs["DoubleHole_R"], allOutputs["DoubleHole_A"], allOutputs["DoubleHole_B"], edgeVectorsDoubleHolePrimal2)
# @show maximum(abs.(curlᶜb))

# codᵛb = codᵛ(allOutputs["SingleHole_R"], allOutputs["SingleHole_A"], allOutputs["SingleHole_B"], edgeVectorsDoubleHolePrimal2)
# @show maximum(abs.(codᵛb))
# cocurlᶜb = cocurlᶜ(allOutputs["SingleHole_R"], allOutputs["SingleHole_A"], allOutputs["SingleHole_B"], edgeVectorsDoubleHolePrimal2)
# @show maximum(abs.(cocurlᶜb))