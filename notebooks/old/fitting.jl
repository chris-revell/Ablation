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

@from "$(srcdir("AblateCells.jl"))" using AblateCells
@from "$(srcdir("Stresses.jl"))" using Stresses

f₁(r) = 1/r 
f₂(r) = 1/(r^2) 
# ϕ(r, θ, θ₀, a, b) = f₁(r)*a + f₂(r)*b*cos(2*(θ-θ₀))
function ϕ(r, θ, θ₀1, θ₀2, a, b, c)
    # functionVal = a + b*cos(θ-θ₀1) + c*cos(2*(θ-θ₀2))
    functionVal = a + b*cos(2*(θ-θ₀2))
    if abs(functionVal) < 1.0
        return 
    else
        return sign(functionVal)
    end
end

dateString = "25-12-19-12-12-33"

testFiles = filter(x->(occursin("Divided",x)||occursin("Ablated",x))&&occursin(".jld2",x), readdir(datadir("displacementFields", dateString)))
dividedCells = [i[(length(dateString)+1+length("Divided"))+1:end-5] for i in testFiles if occursin("Divided", i)]
ablatedCells = [i[(length(dateString)+1+length("Ablated"))+1:end-5] for i in testFiles if occursin("Ablated", i)]
testCells = parse.(Int64, dividedCells∩ablatedCells)

cell = rand(testCells)

importedData = load(datadir("displacementFields", dateString, "$(dateString)_InitialSystem.jld2"))
@unpack integ1 = importedData
(params1, matrices1) = integ1.p 
# @unpack A, B = matrices2 
R1 = reinterpret(SVector{2,Float64}, integ1.u)
cellCentres = findCellCentresOfMass(R1, matrices1.A, matrices1.B)

importedDataAblated = load(datadir("displacementFields", dateString, "$(dateString)_Ablated$(cell).jld2"))
@unpack Rablated, Aablated, Bablated, Fablated, ablationCOM = importedDataAblated


cellCentresAblated = findCellCentresOfMass(Rablated, Aablated, Bablated)
cellPolygonsAblated = findCellPolygons(Rablated, Aablated, Bablated)
Δrᵢ = cellCentresAblated.-cellCentres[Not(cell)]
radiusVectorsAblated = ([cellCentres[i].-ablationCOM for i=1:size(matrices1.B,1)])[Not(cell)]
directionsAblated = normalize.(Δrᵢ).⋅normalize.(radiusVectorsAblated)
θs = atan.(getindex.(radiusVectorsAblated, 1), getindex.(radiusVectorsAblated, 2))

GLMakie.activate!()
fig = GLMakie.Figure(size=(1000,500))
ax1 = Axis(fig[1,1])
ax2 = Axis(fig[1,2], aspect=DataAspect())
ax2.xzoomlock = true; ax2.yzoomlock = true
hidedecorations!(ax2); hidespines!(ax2)
colsize!(fig.layout, 1, Relative(0.45))
colsize!(fig.layout, 2, Relative(0.45))


a = Observable(1.0)
b = Observable(1.0)
c = Observable(1.0)
θ₀1 = Observable(0.0)
θ₀2 = Observable(0.0)
# Set up parameter sliders
parameterSliders = SliderGrid(
    fig[2,1],
    (label="a" , range=-2.0:0.01:2.0, startvalue=1.0, format="{:.2f}"),
    (label="b" , range=-2.01:0.01:2.0, startvalue=1.0, format="{:.2f}"),
    (label="b" , range=-2.01:0.01:2.0, startvalue=1.0, format="{:.2f}"),
    (label="θ₀1" , range=0.0:0.01:(2*π), startvalue=0.0, format="{:.2f}"),
    (label="θ₀2" , range=0.0:0.01:(2*π), startvalue=0.0, format="{:.2f}"),
    width = 400,
)
sliderobservables = [s.value for s in parameterSliders.sliders]
connect!(a, sliderobservables[1])
connect!(b, sliderobservables[2])
connect!(b, sliderobservables[3])
connect!(θ₀1, sliderobservables[4])
connect!(θ₀2, sliderobservables[5])

directions1 = [@lift(ϕ(norm(radiusVectorsAblated), atan(radiusVectorsAblated[i][1], radiusVectorsAblated[i][2]), $(θ₀1), $(θ₀2), $(a), $(b), $(c))) for i=1:size(Bablated,1)]
order = sortperm(θs)
directions2 = @lift[(ϕ(norm(radiusVectorsAblated), atan(radiusVectorsAblated[i][1], radiusVectorsAblated[i][2]), $(θ₀1), $(θ₀2), $(a), $(b), $(c))) for i in order]

# extractVals(vecOfObs) = [o[] for o in vecOfObs]
# order = sortperm(θs)
# θsSorted = CircularArray(θs[order])
# directionsSorted = CircularArray(extractVals(directions))

for i=1:size(Bablated,1) 
    poly!(ax2, cellPolygonsAblated[i], color=directions1[i], colormap=:managua, colorrange=(-1.0,1.0), strokewidth=1, strokecolor=(:black,0.1))
    # poly!(ax2, cellPolygonsAblated[i], color=directions[i], colormap=:managua, colorrange=clims, strokewidth=1, strokecolor=(:black,0.1))
    # scatter!(ax1, θs[i], directions[i], color=:black)
end
# lines!(ax1, θs[order], directions2, color=:black)
cb = Colorbar(fig[1,3], colormap=:managua, colorrange=(-1.0,1.0))
colsize!(fig.layout, 3, Relative(0.1))

# onany() do _ 
#     lim = maximum(abs.(map(x->x.val, colours)))
#     clims[] = (-lim, lim)
#     clims[] = clims[]
#     cb.colorrange = clims[]
# end

# arrows!(ax2, Point{2,Float64}.(cellCentres2), Vec{2,Float64}.(Δrᵢ), color=directions, colormap=:bam, colorrange=climsDirection, lengthscale=20.0)
scatter!(ax2, Point{2,Float64}.([ablationCOM]), color=:red, markersize=10)

display(fig)

