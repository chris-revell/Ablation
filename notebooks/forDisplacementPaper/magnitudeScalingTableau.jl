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

dateString = "25-12-02-16-58-52"

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


for (i, ablatedCell) in enumerate(testCells[1:nPanels])
    
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
    scatter!(axes[end], log10.(norm.(radiusVectorsAblated)), log10.(norm.(Δrᵢ)), color=:red, markersize=3)
    lines!(axes[end], [0.0,1.0], [-2.0,-3.0], color=:black, linewidth=1)
    lines!(axes[end], [0.0,1.0], [-2.0,-4.0], color=:black, linewidth=1)
    

end

for i=1:nPanels
    fig[fld1(i,2), mod1(i,2)] = gridlayouts[i]
end

resize_to_layout!(fig)
display(fig)
save(datadir("displacementFields", dateString, "tableauMagnitudes.pdf"), fig)



# Makie.update_state_before_display!(f) # if you haven't displayed, yet, to update axis limits etc



# signsSmoothed = CircularArray(sign.(smoothedData))
    # difs = signsSmoothed.-signsSmoothed[0:end-1]
    # intersectInds = findall(x->x!=0, difs)
    # intersectsSmoothed = Float64[]
    # for ind in intersectInds
    #     dy = smoothedData[ind]-smoothedData[ind-1]
    #     dx = binMidpoints[ind]-binMidpoints[ind-1]
    #     push!(intersectsSmoothed, (binMidpoints[ind-1] - smoothedData[ind-1]*dx/dy))
    # end
    # scatter!(axes[2], intersectsSmoothed, zeros(length(intersectsSmoothed)), color=:green)
    
    # # Exponential fit 
    # prob = CurveFitProblem(Vector(θsSorted), Vector(directionsSorted))
    # sol = solve(prob, ExpSumFitAlgorithm(n=20, withconst=true))
    # xs = collect(-π:0.01:π)
    # ys = []
    # for i=1:length(xs)
    #     push!(ys, exponentialFit(xs[i], sol.u))
    # end
    # lines!(axes[2], xs, real.(ys))

    # # Polynomial fit 
    # prob = CurveFitProblem(Vector(θsSorted), Vector(directionsSorted))
    # sol = solve(prob, PolynomialFitAlgorithm(degree=20))
    # xs = collect(-π:0.01:π)
    # ys = []
    # for i=1:length(xs)
    #     push!(ys, polynomialFit(xs[i], sol.u))
    # end
    # lines!(axes[2], xs, real.(ys))