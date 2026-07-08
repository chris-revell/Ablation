using DrWatson
using DiscreteCalculus
using GLMakie
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
using REPL.TerminalMenus
using CircularArrays
using OrdinaryDiffEq

@from "$(srcdir("AblateCells.jl"))" using AblateCells
@from "$(srcdir("Stresses.jl"))" using Stresses

GLMakie.activate!()

options = ["yes", "no"]
# `pagesize` is the number of items to be displayed at a time.
#  The UI will scroll if the number of options is greater
#  than the `pagesize`
menu = RadioMenu(options, pagesize=length(options))

f₁(r) = 1/r 
f₂(r) = 1/(r^2) 
# ϕ(r, θ, θ₀, a, b) = f₁(r)*a + f₂(r)*b*cos(2*(θ-θ₀))
ϕ(r, θ, θ₀, a, b) = a/b + cos(2*(θ-θ₀))

dateString = "25-12-02-16-58-52"

testFiles = filter(x->(occursin("Divided",x)||occursin("Ablated",x))&&occursin(".jld2",x), readdir(datadir("displacementFields", dateString)))
dividedCells = [i[(length(dateString)+1+length("Divided"))+1:end-5] for i in testFiles if occursin("Divided", i)]
ablatedCells = [i[(length(dateString)+1+length("Ablated"))+1:end-5] for i in testFiles if occursin("Ablated", i)]
testCells = parse.(Int64, dividedCells∩ablatedCells)

acceptedCells = []
# cell = rand(testCells)

fig = Figure(size=(500,500))
ax1 = Axis(fig[1,1])
ax2 = Axis(fig[1,2], aspect=DataAspect())
hidedecorations!(ax2)
hidespines!(ax2)
ax3 = PolarAxis(fig[2, 1])
hidedecorations!(ax3)

display(fig)

for ablatedCell in testCells
    empty!(ax1)
    empty!(ax2)
    empty!(ax3)

    importedData = load(datadir("displacementFields", dateString, "$(dateString)_InitialSystem.jld2"))
    @unpack integ1 = importedData
    (params1, matrices1) = integ1.p 
    # @unpack A, B = matrices2 
    R1 = reinterpret(SVector{2,Float64}, integ1.u)
    cellCentres = findCellCentresOfMass(R1, matrices1.A, matrices1.B)

    importedDataAblated = load(datadir("displacementFields", dateString, "$(dateString)_Ablated$(ablatedCell).jld2"))
    @unpack Rablated, Aablated, Bablated, Fablated, ablationCOM = importedDataAblated

    cellCentresAblated = findCellCentresOfMass(Rablated, Aablated, Bablated)
    cellPolygonsAblated = findCellPolygons(Rablated, Aablated, Bablated)
    Δrᵢ = cellCentresAblated.-cellCentres[Not(ablatedCell)]
    radiusVectorsAblated = ([cellCentres[i].-ablationCOM for i=1:size(matrices1.B,1)])[Not(ablatedCell)]
    directionsAblated = normalize.(Δrᵢ).⋅normalize.(radiusVectorsAblated)
    θs = atan.(getindex.(radiusVectorsAblated, 1), getindex.(radiusVectorsAblated, 2))
    centralcells = findall(x->norm(x)<0.33*maximum(norm.(radiusVectorsAblated)), radiusVectorsAblated)
    θsTruncated = θs[centralcells]
    θsSorted = CircularArray(sort(θsTruncated))

    directionsTruncated = directionsAblated[centralcells]
    directionsSorted = CircularArray(directionsTruncated[sortperm(θsTruncated)])
    signs = sign.(directionsSorted)
    difs = signs.-signs[0:end-1]
    intersectInds = findall(x->x!=0, difs)
    intersects = Float64[]
    for ind in intersectInds
        dy = directionsSorted[ind]-directionsSorted[ind-1]
        dx = θsSorted[ind] - θsSorted[ind-1]
        # 0 = directionsTruncated[ind-1] + xDif*dy/dx
        push!(intersects, (θsSorted[ind-1] - directionsSorted[ind-1]*dx/dy))
    end

    scatter!(ax1, θsTruncated, directionsTruncated)
    lines!(ax1, θsSorted, directionsSorted)
    scatter!(ax1, intersects, zeros(length(intersects)), color="red")
    
    for i=1:size(Bablated,1)
        if i∈centralcells
            poly!(ax2, cellPolygonsAblated[i], color=directionsAblated[i], colormap=:managua, colorrange=(-1.0,1.0), strokewidth=1, strokecolor=(:black,0.1))
        else
            poly!(ax2, cellPolygonsAblated[i], color=(:black,0.1), strokewidth=1, strokecolor=(:black,0.1))
        end
    end

    eigenVals, eigenVecs = eigen(matrices1.cellShapeTensor[ablatedCell]) 
    # shortvec = eigenVecs[:,1]
    divθ = atan(eigenVecs[1,1], eigenVecs[2,1])
    # scatter!(ax2, [θsTruncated], directionsTruncated)
    # lines!(ax2, Point2{2,Float64}.(normalize.([-shortvec, shortvec])), color=:red)
    for int in intersects
        lines!(ax3, [int, int], [0,1], color=:blue)    
    end
    lines!(ax3, [divθ+π/2, divθ-π/2, divθ-π/2], [1.0,0.0,1.0], color=:red)


    # `request` displays the menu and returns the index after the
    #   user has selected a choice
    choice = request("Accept?", menu)
    if options[choice] == "yes"
        push!(acceptedCells, ablatedCell)
    end

end

@show acceptedCells



# eigenVals, eigenVecs = eigen(matrices.cellShapeTensor[i]) # eigenvalues/vectors listed smallest to largest eigval.

# # circ = abs(eigenVals[1]/eigenVals[2]) # circularity
# # #for very circular cells randomly choose division axis
# # if circ > 0.95 
# #     theta = rand()*π
# #     shortvec = [cos(theta), sin(theta)]
# # else