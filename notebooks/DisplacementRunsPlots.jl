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

@from "$(srcdir("AblateCells.jl"))" using AblateCells
@from "$(srcdir("Stresses.jl"))" using Stresses

# dateString = "25-12-11-09-28-09"
dateString = "25-12-15-13-10-46"

fig = Figure(size=(2000,2000), fontsize=12)
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"(%$l)" for l in individualLetters]

# Import system data
importedData = load(datadir("displacementFields", dateString, "$(dateString)_InitialSystem.jld2"))
@unpack integ1 = importedData
(params1, matrices1) = integ1.p 
# @unpack A, B = matrices2 
R1 = reinterpret(SVector{2,Float64}, integ1.u)

testFiles = filter(x->(occursin("Divided",x)||occursin("Ablated",x))&&occursin(".jld2",x), readdir(datadir("displacementFields", dateString)))

dividedCells = [i[(length(dateString)+1+length("Divided"))+1:end-5] for i in testFiles if occursin("Divided", i)]
ablatedCells = [i[(length(dateString)+1+length("Ablated"))+1:end-5] for i in testFiles if occursin("Ablated", i)]
testCells = parse.(Int64, dividedCells∩ablatedCells)

cellCentres = findCellCentresOfMass(R1, matrices1.A, matrices1.B)

𝐡 = hNetwork(R1, matrices1.A, matrices1.B, matrices1.F)
# Stress tensors 
σᵢ = σ(R1, matrices1.A, matrices1.B, 𝐡)
# Deviatoric stress 
σDᵢ = [σᵢ[i] .- 0.5*tr(σᵢ[i]) for i=1:size(matrices1.B,1)]
σDSᵢ = 0.5.*(σDᵢ .+ transpose.(σDᵢ))
# ~\ref{eq:shearstressexact}
ζᵢ = [sqrt(-det(σDSᵢ[i])) for i=1:size(matrices1.B,1)]
p_eff = -0.5.*cocurlᶜ(R1, matrices1.A, matrices1.B, 𝐡)

cellOrder = sortperm(p_eff[testCells])
orderedCells = testCells[cellOrder]

maxPanels = 25
nPanels = min(maxPanels,length(testCells))

for cell in orderedCells[1:nPanels]

    importedDataAblated = load(datadir("displacementFields", dateString, "$(dateString)_Ablated$(cell).jld2"))
    @unpack Rablated, Aablated, Bablated, Fablated, ablationCOM = importedDataAblated
    
    cellCentresAblated = findCellCentresOfMass(Rablated, Aablated, Bablated)
    cellPolygonsAblated = findCellPolygons(Rablated, Aablated, Bablated)
    Δrᵢ = cellCentresAblated.-cellCentres[Not(cell)]
    radiusVectorsAblated = [cellCentres[i].-ablationCOM for i=1:size(matrices1.B,1)]
    directionsAblated = normalize.(Δrᵢ).⋅normalize.(radiusVectorsAblated[Not(cell)])

    p_effString = @sprintf("%.3f", p_eff[cell])

    push!(axes, Axis(fig, aspect=DataAspect(), title=L"Ablation:\ i=%$(cell),\ P_{eff}=%$(p_effString)", titlealign = :center))
    for i=1:size(Bablated,1)
        poly!(axes[end], cellPolygonsAblated[i], color=directionsAblated[i], colormap=:managua, colorrange=(-1.0,1.0), strokewidth=1, strokecolor=(:black,0.1))
    end
    # arrows!(axes[end], Point{2,Float64}.(cellCentres2), Vec{2,Float64}.(Δrᵢ), color=directions, colormap=:bam, colorrange=climsDirection, lengthscale=20.0)
    scatter!(axes[end], Point{2,Float64}.([ablationCOM]), color=:red, markersize=10)


    importedDataDivided = load(datadir("displacementFields", dateString, "$(dateString)_Divided$(cell).jld2"))
    @unpack Rdivided, Adivided, Bdivided, Fdivided, divisionCOM = importedDataDivided
    
    cellCentresDivided = findCellCentresOfMass(Rdivided, Adivided, Bdivided)
    cellPolygonsDivided = findCellPolygons(Rdivided, Adivided, Bdivided)
    Δrᵢ = cellCentresDivided[1:end-1].-cellCentres
    radiusVectorsDivided = [cellCentres[i].-ablationCOM for i=1:size(matrices1.B,1)]
    directionsDivided = normalize.(Δrᵢ).⋅normalize.(radiusVectorsDivided)

    push!(axes, Axis(fig, aspect=DataAspect(), title=L"Division:\ i=%$(cell),\ P_{eff}=%$(p_effString)", titlealign = :center))
    for i=1:size(Bdivided,1)
        if i==cell || i==size(Bdivided,1)
            poly!(axes[end], cellPolygonsDivided[i], color=(:black,1.0), strokewidth=1, strokecolor=(:black,1.0))
        else
            poly!(axes[end], cellPolygonsDivided[i], color=directionsDivided[i], colormap=:managua, colorrange=(-1.0,1.0), strokewidth=1, strokecolor=(:black,0.1))
        end
    end
    # arrows!(axes[end], Point{2,Float64}.(cellCentres2), Vec{2,Float64}.(Δrᵢ), color=directions, colormap=:bam, colorrange=climsDirection, lengthscale=20.0)
    scatter!(axes[end], Point{2,Float64}.([divisionCOM]), color=:red, markersize=10)

end

hidedecorations!.(axes)
hidespines!.(axes)

#%%

nCols = 5 
# for (ind, val) in enumerate(orderAxes[1:min(26,length(testCells))])
#     gl = GridLayout(fig[fld1(ind,nCols), mod1(ind,nCols)])
#     gl[1,1] = axes[2*val-1]
#     gl[1,2] = axes[2*val]
#     Label(gl[1,1:2,Bottom()], subfigureLabels[ind], fontsize=24) 
#     Box(gl[1,1:2], color = (:white, 0.0), strokecolor=:black, cornerradius=20)
#     colsize!(gl, 1, Relative(0.5))
#     colsize!(gl, 2, Relative(0.5))
#     # rowsize!(gl, 1, Relative(1.0))
# end
for ind=1:nPanels
    gl = GridLayout(fig[fld1(ind,nCols), mod1(ind,nCols)])
    gl[1,1] = axes[2*ind-1]
    gl[1,2] = axes[2*ind]
    Label(gl[1,1:2,Bottom()], subfigureLabels[ind], fontsize=24) 
    Box(gl[1,1:2], color = (:white, 0.0), strokecolor=:black, cornerradius=20)
    colsize!(gl, 1, Relative(0.5))
    colsize!(gl, 2, Relative(0.5))
    # rowsize!(gl, 1, Relative(1.0))
end

for col=1:nCols
    colsize!(fig.layout, col, Relative(1.0/nCols))
end
nRows = fld1(nPanels, nCols)
for row=1:nRows
    rowsize!(fig.layout, row, Aspect(1, 0.55 ))
end
resize_to_layout!(fig)
display(fig)

#%%

!isdir(mkpath(plotsdir("displacementFields", dateString))) ? mkpath(plotsdir("displacementFields", dateString)) : nothing
save(plotsdir("displacementFields", dateString, "DisplacementGridMultiCell.png"), fig)
save(plotsdir("displacementFields", dateString, "DisplacementGridMultiCell.pdf"), fig)
