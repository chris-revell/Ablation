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
using InvertedIndices
using LaTeXStrings
using Printf
using CircularArrays
using StatsBase
using LsqFit
@from "$(srcdir("AblateCells.jl"))" using AblateCells


function ϕ(θs, p)
    out = []
    for (i, θ) in enumerate(θs)
        tmp = p[1] + p[2]*cos(θ-p[3]) + p[4]*cos(2.0*(θ-p[5]))
        if abs(tmp) < 1.0
            push!(out, tmp)
        else
            push!(out, sign(tmp)*1.0)
        end
    end
    return out
end

# dateString = "25-12-02-16-58-52"
# dateString = "26-06-19-08-32-27"
# dateString = "26-06-19-08-32-27"
dateString = "26-07-07-16-49-05"

# Establish a set of cells for which ablated and divided results have been created
testFiles = filter(x->(occursin("Divided",x)||occursin("Ablated",x))&&occursin(".jld2",x), readdir(datadir("displacementFields", dateString)))
dividedCells = [i[(length(dateString)+1+length("Divided"))+1:end-5] for i in testFiles if occursin("Divided", i)]
ablatedCells = [i[(length(dateString)+1+length("Ablated"))+1:end-5] for i in testFiles if occursin("Ablated", i)]
testCells = parse.(Int64, dividedCells∩ablatedCells)

# fig = Figure(size=(600,1000))#, backgroundcolor = :tomato)
fig = Figure()#, backgroundcolor = :tomato)
axes = []
gridlayouts = []

nPanels = 18

# Import and process data
importedData = load(datadir("displacementFields", dateString, "$(dateString)_InitialSystem.jld2");
                typemap=Dict("VertexModel.../VertexModelContainers.jl.VertexModelContainers.MatricesContainer" => MatricesContainer, 
                            "VertexModel.../VertexModelContainers.jl.VertexModelContainers.ParametersContainer" => ParametersContainer
                )
            )
@unpack R, params, matrices = importedData
cellCentres = findCellCentresOfMass(R, matrices.A, matrices.B)

fittingParams = []
divθs = []

for (i, ablatedCell) in enumerate(testCells[1:min(nPanels, length(testCells))])
    
    @show ablatedCell

    push!(gridlayouts, GridLayout(parent=fig))
    push!(axes, Axis(gridlayouts[end][1,1], aspect=DataAspect()))
    hidedecorations!(axes[end]); hidespines!(axes[end])
    
    importedDataAblated = load(datadir("displacementFields", dateString, "$(dateString)_Ablated$(ablatedCell).jld2");
                    typemap=Dict("VertexModel.../VertexModelContainers.jl.VertexModelContainers.MatricesContainer" => MatricesContainer, 
                                "VertexModel.../VertexModelContainers.jl.VertexModelContainers.ParametersContainer" => ParametersContainer
                    )
                )
                
    @unpack Rablated, Aablated, Bablated, Fablated, ablationCOM = importedDataAblated
    cellCentresAblated = findCellCentresOfMass(Rablated, Aablated, Bablated)
    cellPolygonsAblated = findCellPolygons(Rablated, Aablated, Bablated)
    Δrᵢ = cellCentresAblated.-cellCentres[Not(ablatedCell)]
    radiusVectorsAblated = ([cellCentres[i].-ablationCOM for i=1:size(matrices.B,1)])[Not(ablatedCell)]
    directionsAblated = normalize.(Δrᵢ).⋅normalize.(radiusVectorsAblated)
    θs = atan.(getindex.(radiusVectorsAblated, 1), getindex.(radiusVectorsAblated, 2))
    centralcells = findall(x->norm(x)<0.33*maximum(norm.(radiusVectorsAblated)), radiusVectorsAblated)
    θsTruncated = θs[centralcells]
    θsSorted = CircularArray(sort(θsTruncated))
    radiusVectorsSorted = CircularArray(radiusVectorsAblated[sortperm(θsTruncated)])

    # Monolayer visualisation
    for i=1:size(Bablated,1)
        if i∈centralcells
            poly!(axes[end], cellPolygonsAblated[i], color=directionsAblated[i], colormap=:managua, colorrange=(-1.0,1.0), strokewidth=1, strokecolor=(:black,0.1))
        else
            poly!(axes[end], cellPolygonsAblated[i], color=(:black,0.1), strokewidth=1, strokecolor=(:black,0.1))
        end
    end

    # Resize axis panel
    axwidth, axheight = widths(axes[end].scene.viewport[])
    axes[end].width = axwidth
    axes[end].height = axheight

    push!(axes, Axis(gridlayouts[end][1,2]))
    ylims!(axes[end], (-1.0,1.0))

    # Raw data
    directionsTruncated = directionsAblated[centralcells]
    directionsSorted = CircularArray(directionsTruncated[sortperm(θsTruncated)])
    # signs = sign.(directionsSorted)
    # difs = signs.-signs[0:end-1]
    # intersectInds = findall(x->x!=0, difs)
    # intersects = Float64[]
    # for ind in intersectInds
    #     dy = directionsSorted[ind]-directionsSorted[ind-1]
    #     dx = θsSorted[ind] - θsSorted[ind-1]
    #     # 0 = directionsTruncated[ind-1] + xDif*dy/dx
    #     push!(intersects, (θsSorted[ind-1] - directionsSorted[ind-1]*dx/dy))
    # end
    # lines!(axes[end], θsSorted, directionsSorted, color=:red)
    scatter!(axes[end], θsSorted, directionsSorted, color=:red)
    # scatter!(axes[end], intersects, zeros(length(intersects)), color=:red)

    # # Smoothed data 
    # h1 = StatsBase.fit(Histogram, θsSorted, weights(directionsSorted), -π:π/20:π+π/50)
    # h2 = StatsBase.fit(Histogram, θsSorted, -π:π/20:π+π/50)
    # binEdges = collect(h1.edges[1])
    # deleteat!(binEdges, findall(x->x==0, h2.weights))
    # binMidpoints = binEdges[1:end-1] .+ (binEdges[2:end]-binEdges[1:end-1])/2.0
    # smoothedData = [h1.weights[i]/h2.weights[i] for i=1:length(h2.weights) if h2.weights[i]!=0]
    # lines!(axes[end], binMidpoints, smoothedData, linewidth=2, color=(:green,0.75))
    
    eigenVals, eigenVecs = eigen(matrices.cellShapeTensor[ablatedCell]) 
    divθ = atan(eigenVecs[1,1], eigenVecs[2,1])
    # abs(minimum(smoothedData)+1.0) > abs(maximum(smoothedData)-1.0) ? aOverb = minimum(smoothedData)+1.0 : aOverb = maximum(smoothedData)-1.0
    
    # p0 = [aOverb, divθ]
    # p0 = [0.0,0.0]
    # p0 = rand(2)
    p0 = rand(5)

    localFit = curve_fit(ϕ, θsSorted, directionsSorted, p0)

    bestFitDirections = ϕ(θsSorted, localFit.param)
    lines!(axes[end], θsSorted, bestFitDirections, linewidth=2, color=(:blue,0.75))

    colsize!(gridlayouts[end], 1, Relative(0.6))
    # colsize!(gridlayouts[end], 2, Aspect(1,1.0))

    push!(fittingParams, localFit.param)
    push!(divθs, divθ)

end

for i=1:min(nPanels, length(testCells))
    fig[fld1(i,2), mod1(i,2)] = gridlayouts[i]
end

resize_to_layout!(fig)
display(fig)
save(datadir("displacementFields", dateString, "tableauDirections.pdf"), fig)

#%%

fig2 = Figure()
ax2 = Axis(fig2[1,1])
scatter!(ax2, (divθs.+2*π).%2.0*π, (getindex.(fittingParams, 5).+2*π).%2.0*π, color=:red)
lines!(ax2, [0.0,2π], [0.0, 2π])
lines!(ax2, [0.0,2π], [0.0, 2π].-π/2)
lines!(ax2, [0.0,2π], [0.0, 2π].-π)
# scatter!(ax2, (divθs.+2*π).%2.0*π, (getindex.(fittingParams, 3).+2*π).%2.0*π, color=:green)
display(fig2)