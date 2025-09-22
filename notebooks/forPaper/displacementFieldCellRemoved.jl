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

@from "$(srcdir("AblateCells.jl"))" using AblateCells
@from "$(srcdir("Stresses.jl"))" using Stresses

inputSystem = "Large2"
inputDir = "quadraticPotentialNoPressure"; isdir(plotsdir(inputDir)) ? nothing : mkdir(plotsdir(inputDir))
# inputDir = "quadraticPotentialWithPressure"; isdir(plotsdir(inputDir)) ? nothing : mkdir(plotsdir(inputDir))

inFile = datadir("referenceSystems", inputDir, "$(inputSystem)_testSystem.jld2")
importedData = load(inFile)
@unpack R, A, B, F = importedData

systemCOM = sum(R)./length(R)
cellCentres1 = findCellCentresOfMass(R, A, B)
centralCell = findmin(norm.([cellCentres1[i].-systemCOM for i=1:size(B,1)]))[2]

neighbourMatrix = dropzeros(B*transpose(B))
ablatedCells = [centralCell]

if isfile(datadir("referenceSystems", inputDir, "$(inputSystem)Ablated_testSystem.jld2"))
    inFile = datadir("referenceSystems", inputDir, "$(inputSystem)Ablated_testSystem.jld2")
    importedData = load(inFile)
    @unpack R2, A2, B2, F2, systemCOM2 = importedData
else
    Rtmp, Atmp, Btmp = ablateCells(R, A, B, ablatedCells)

    integ2 = vertexModel(abstol = 1e-9,
                        reltol = 1e-9,
                        initialSystem="argument",
                        divisionToggle=0,
                        R_in=Rtmp,
                        A_in=Atmp,
                        B_in=Btmp,
                        pressureExternal=0.0,
                        nCycles=0.5,
                        outputToggle=0,
                        frameDataToggle=0,
                        frameImageToggle=0,
                        videoToggle=0,
                        printToggle=1,
                        energyModel="quadratic",
                    )
    #%%
    R2 = reinterpret(SVector{2,Float64}, integ2.u) 
    params2, matrices2 = integ2.p
    A2 = matrices2.A
    B2 = matrices2.B
    F2 = matrices2.F
    @show maximum(norm.(sum(F2, dims=2)))

    C = findC(A, B)
    centralCellVertices = R[findall(x->x!=0, C[centralCell, :])]
    systemCOM2 = sum(centralCellVertices)./length(centralCellVertices)
    
    jldsave(datadir("referenceSystems", inputDir, "$(inputSystem)Ablated_testSystem.jld2"); R2,
        A2, 
        B2, 
        F2, 
        systemCOM2,
    )
end

I = size(B2,1)
J = size(B2,2)
K = size(A2,2)

cellCentres1 = findCellCentresOfMass(R, A, B)
cellCentres2 = findCellCentresOfMass(R2, A2, B2)

𝐜ⱼ1 = findEdgeMidpoints(R, A)
𝐜ⱼ2 = findEdgeMidpoints(R2, A2)

cellradii2 = norm.([cellCentres2[i].-systemCOM2 for i=1:size(B2,1)])
cellDummyDists2 = collect(maximum(cellradii2)/100:maximum(cellradii2)/100:maximum(cellradii2))
vertexradii2 = norm.([R2[k].-systemCOM2 for k=1:size(A2,2)])
vertexDummyDists2 = collect(maximum(vertexradii2)/100:maximum(vertexradii2)/100:maximum(vertexradii2))
edgeradii2 = norm.([𝐜ⱼ2[j].-systemCOM2 for j=1:size(A2,1)])
edgeDummyDists2 = collect(maximum(edgeradii2)/100:maximum(edgeradii2)/100:maximum(edgeradii2))
ΔRₖ = R2.-R 
Δrᵢ = cellCentres2.-cellCentres1[Not(ablatedCells)]
Δrⱼ = 𝐜ⱼ2.-𝐜ⱼ1

#%%

fig = Figure(size=(1000,500), fontsize=24)
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"(%$l)" for l in individualLetters]

push!(axes, Axis(fig[1,1], aspect=DataAspect(), alignmode=Inside()))
# scatter!(axes[end], Point{2,Float64}.(R), color=(:red,0.5), markersize=5)
# scatter!(axes[end], Point{2,Float64}.(R2), color=(:green,0.5), markersize=5)
arrowColours = [(:green, norm(Δrⱼ[j])/maximum(norm.(Δrⱼ))) for j=1:J]
# for j=1:J
#     lines!(axes[end], Point{2,Float64}.([𝐜ⱼ1[j], 𝐜ⱼ2[j]]), color=:black, linewidth=5)
# end
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ1), Vec{2,Float64}.(Δrⱼ), color=arrowColours, lengthscale=1.0)
hidedecorations!(axes[end])
hidespines!(axes[end])
Label(fig[2,1], popfirst!(subfigureLabels), fontsize=24) 

push!(axes, Axis(fig[1,2], aspect=AxisAspect(1)))
scatter!(axes[end], log10.(edgeradii2), log10.(norm.(Δrⱼ)), color=(:blue,0.3))
lines!(axes[end], log10.(edgeDummyDists2), log10.(0.02./edgeDummyDists2), color=(:black, 0.75), linestyle=:dash)
# lines!(axes[end], log10.(edgeDummyDists2), log10.(0.02./(edgeDummyDists2).^2), color=(:black, 0.75), linestyle=:dash)
axes[end].xlabel = L"\log_{10}\left(r_j\right)"
axes[end].ylabel = L"\log_{10}\left(\Delta r_j\right)"
ylims!(axes[end], (-4.2, -1.0))
xlims!(axes[end], (-0.8, 1.0))
axes[end].xticks = (-0.8:0.8:0.8, string.(-0.8:0.8:0.8))
Label(fig[2,2], popfirst!(subfigureLabels), fontsize=24) 


# push!(axes, Axis(fig[1,1], aspect=DataAspect(), alignmode=Inside()))
# # scatter!(axes[end], Point{2,Float64}.(R), color=(:red,0.5), markersize=5)
# # scatter!(axes[end], Point{2,Float64}.(R2), color=(:green,0.5), markersize=5)
# arrowColours = [(:green, log10(norm(ΔRₖ[k]))/log10(maximum(norm.(ΔRₖ)))) for k=1:K]
# arrows!(axes[end], Point{2,Float64}.(R), Vec{2,Float64}.(ΔRₖ), color=arrowColours, lengthscale=2.0)
# hidedecorations!(axes[end])
# hidespines!(axes[end])
# Label(fig[2,1], popfirst!(subfigureLabels), fontsize=24) 

# push!(axes, Axis(fig[1,2], aspect=AxisAspect(1)))
# scatter!(axes[end], log10.(vertexradii2), log10.(norm.(ΔRₖ)), color=(:blue,0.3))
# lines!(axes[end], log10.(vertexDummyDists2), log10.(0.02./vertexDummyDists2), color=(:black, 0.75), linestyle=:dash)
# lines!(axes[end], log10.(vertexDummyDists2), log10.(0.02./(vertexDummyDists2).^2), color=(:black, 0.75), linestyle=:dash)
# axes[end].xlabel = L"\log_{10}\left(r_k\right)"
# axes[end].ylabel = L"\log_{10}\left(\Delta r_k\right)"
# ylims!(axes[end], (-4.2, -1.0))
# xlims!(axes[end], (-0.8, 1.0))
# axes[end].xticks = (-0.8:0.8:0.8, string.(-0.8:0.8:0.8))
# Label(fig[2,2], popfirst!(subfigureLabels), fontsize=24) 


# push!(axes, Axis(fig[3,1], aspect=DataAspect(), alignmode=Inside()))
# # scatter!(axes[end], Point{2,Float64}.(cellCentres1), color=(:red,0.5), markersize=5)
# # scatter!(axes[end], Point{2,Float64}.(cellDummyDists2), color=(:green,0.5), markersize=5)
# # arrowColours = [(:green, norm(Δrᵢ[i])/maximum(norm.(Δrᵢ))) for i=1:I]
# arrowColours = [(:green, log10(norm(Δrᵢ[i]))/log10(maximum(norm.(Δrᵢ)))) for i=1:I]
# arrows!(axes[end], Point{2,Float64}.(cellCentres2), Vec{2,Float64}.(Δrᵢ), color=arrowColours, lengthscale=2.0)
# hidedecorations!(axes[end])
# hidespines!(axes[end])
# Label(fig[4,1], popfirst!(subfigureLabels), fontsize=24) 

# push!(axes, Axis(fig[3,2], aspect=AxisAspect(1)))
# scatter!(axes[end], log10.(cellradii2), log10.(norm.(Δrᵢ)), color=(:blue,0.3))
# lines!(axes[end], log10.(cellDummyDists2), log10.(0.02./cellDummyDists2), color=(:black, 0.75), linestyle=:dash)
# lines!(axes[end], log10.(cellDummyDists2), log10.(0.02./(cellDummyDists2).^2), color=(:black, 0.75), linestyle=:dash)
# axes[end].xlabel = L"\log_{10}\left(R_i\right)"
# axes[end].ylabel = L"\log_{10}\left(\Delta R_i\right)"
# ylims!(axes[end], (-4.2, -1.0))
# xlims!(axes[end], (-0.8, 1.0))
# axes[end].xticks = (-0.8:0.8:0.8, string.(-0.8:0.8:0.8))
# Label(fig[4,2], popfirst!(subfigureLabels), fontsize=24) 

rowsize!(fig.layout, 1, Relative(0.9))
rowsize!(fig.layout, 2, Relative(0.1))
# rowsize!(fig.layout, 3, Relative(0.4))
# rowsize!(fig.layout, 4, Relative(0.1))
colsize!(fig.layout, 1, Relative(0.5))
colsize!(fig.layout, 2, Relative(0.5))
# colsize!(fig.layout, 3, Relative(0.4))

# colgap!(fig.layout, 1, Relative(-0.05))
# rowgap!(fig.layout, 1, Relative(-0.01))
# rowgap!(fig.layout, 3, Relative(-0.01))

# resize_to_layout!(fig)

display(fig)

save(plotsdir(inputDir, "displacement.png"), fig)
save(plotsdir(inputDir, "displacement.pdf"), fig)


