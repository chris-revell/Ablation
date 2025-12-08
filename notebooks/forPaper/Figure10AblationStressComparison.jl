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
    params2, matrices2 = integ2.p
    @show maximum(norm.(sum(F2, dims=2)))
    R2 = reinterpret(SVector{2,Float64}, integ2.u)
    A2 = matrices2.A
    B2 = matrices2.B
    F2 = matrices2.F

    C = findC(A, B)
    centralCellVertices = R[findall(x->x!=0, C[centralCell, :])]
    systemCOM2 = sum(centralCellVertices)./length(centralCellVertices)
    
    jldsave(datadir("referenceSystems", inputDir, "$(inputSystem)Ablated_testSystem.jld2"); 
        R2,
        A2, 
        B2, 
        F2, 
        systemCOM2,
    )
end

#%%
    
cellCentres2 = findCellCentresOfMass(R2, A2, B2)

aᵢ1 = findCellAreas(R, A, B)
aᵢ2 = findCellAreas(R2, A2, B2)

Rᵢ2 = norm.([cellCentres2[i].-systemCOM2 for i=1:size(B2,1)])
Rᵢ2dummy = collect(maximum(Rᵢ2)/100:maximum(Rᵢ2)/100:maximum(Rᵢ2))

cellPolygons2 = findCellPolygons(R2, A2, B2)
edgeQuadrilaterals2 = findEdgeQuadrilaterals(R2, A2, B2)
iᵖ2 = findPeripheralCells(B2).==1

𝐡1 = hNetwork(R, A, B, F)
𝐡2 = hNetwork(R2, A2, B2, F2)

# Isotropic stress 
# pEff1 = pEff(R, A, B, 0.2, 0.75; forceModel="quadratic")
pEff1 = 0.5.*cocurlᶜ(R, A, B, 𝐡1)
# pEff2 = pEff(R2, A2, B2, 0.2, 0.75; forceModel="quadratic")
pEff2 = 0.5.*cocurlᶜ(R2, A2, B2, 𝐡2)

# Stress tensors 
σᵢ1 = σ(R, A, B, 𝐡1)
σᵢ2 = σ(R2, A2, B2, 𝐡2)

cocurlᶜh = cocurlᶜ(R, A, B, 𝐡1)

#Validate cocurlᶜh==tr(σ)
printstyled("Validate cocurlᶜh==tr(σ)\n", color= (maximum(abs.(cocurlᶜh.-tr.(σᵢ1)))<0.0000001 ? :green : :red))
@show maximum(abs.(cocurlᶜh.-tr.(σᵢ1)))
#Validate cocurlᶜh==tr(σ)
printstyled("Validate cocurlᶜh==2Peff\n", color= (maximum(abs.(cocurlᶜh.-2.0.*pEff1))<0.0000001 ? :green : :red))
@show maximum(abs.(cocurlᶜh.-2.0.*pEff1))
#Validate ∑aᵢtr(σᵢ)=0
printstyled("Validate ∑aᵢtr(σᵢ)=0\n", color= (sum(aᵢ1.*tr.(σᵢ1))<0.0000001 ? :green : :red))
@show sum(aᵢ1.*tr.(σᵢ1))

# Deviatoric stress 
σDᵢ1 = [σᵢ1[i] .- 0.5*tr(σᵢ1[i]) for i=1:size(B,1)]
# @show maximum(abs.(tr.(σDᵢ1))) # Validation 
σDᵢ2 = [σᵢ2[i] .- 0.5*tr(σᵢ2[i]) for i=1:size(B2,1)]
# @show maximum(abs.(tr.(σDᵢ2))) # Validation 
σDSᵢ1 = 0.5.*(σDᵢ1 .+ transpose.(σDᵢ1))
σDSᵢ2 = 0.5.*(σDᵢ2 .+ transpose.(σDᵢ2))

# ~\ref{eq:shearstressexact}
ζᵢ1 = [sqrt(-det(σDSᵢ1[i])) for i=1:size(B,1)]
ζᵢ2 = [sqrt(-det(σDSᵢ2[i])) for i=1:size(B2,1)]
ζᵢ1lims = (minimum(log10.(ζᵢ1)), maximum(log10.(ζᵢ1)))
ζᵢ2lims = (minimum(log10.(ζᵢ2)), maximum(log10.(ζᵢ2)))


#%%

Δζ = abs.(ζᵢ2.-ζᵢ1[Not(ablatedCells)])
ΔζLims = (log10(minimum(Δζ)), log10(maximum(Δζ)))
ΔpEff = abs.(pEff2.-pEff1[Not(ablatedCells)])
ΔpEffLims = (log10(minimum(ΔpEff)), log10(maximum(ΔpEff)))
clims = (log10(min(minimum(Δζ), minimum(ΔpEff))), log10(max(maximum(Δζ), maximum(ΔpEff))))

fig = Figure(size=(1500,800), fontsize=24)
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"(%$l)" for l in individualLetters]

gl1 = GridLayout(fig[1,1])
push!(axes, Axis(gl1[1,1], aspect=DataAspect()))
hidedecorations!(axes[end])
hidespines!(axes[end])
for i=1:size(B2,1)
    poly!(axes[end], cellPolygons2[i], color=log10(Δζ[i]), colorrange = clims, colormap = :bam, strokecolor=(:black, 0.2), strokewidth=0.5)
end
Colorbar(gl1[1,2], colorrange=clims, colormap=:bam, height=Relative(0.6), label=L"\log_{10}\left(|\Delta\zeta_i|\right)")
Label(gl1[2,1:2], popfirst!(subfigureLabels), fontsize=24) 
colsize!(gl1, 1, Relative(0.9))
colsize!(gl1, 2, Relative(0.1))
rowsize!(gl1, 1, Relative(0.95))
rowsize!(gl1, 2, Relative(0.05))

gl1b = GridLayout(fig[2,1])
push!(axes, Axis(gl1b[1,1]))#, aspect=AxisAspect(1.25)))
scatter!(axes[end], log10.(Rᵢ2[Not(iᵖ2)]), log10.(abs.(Δζ[Not(iᵖ2)])), color=(:blue, 0.25))
scatter!(axes[end], log10.(Rᵢ2[iᵖ2]), log10.(abs.(Δζ[iᵖ2])), color=(:red, 0.25))
# lines!(axes[end], log10.(Rᵢ2dummy), log10.(0.015*1.0./Rᵢ2dummy), color=(:black, 0.75), linestyle=:dash)
lines!(axes[end], log10.(Rᵢ2dummy), log10.(0.016*1.0./Rᵢ2dummy.^2), color=(:black, 0.75), linestyle=:dash)
axes[end].ylabel = L"\log_{10}\left(|\Delta\zeta_i|\right)"
axes[end].xlabel = L"\log_{10}\left(R_i\right)"
ylims!(axes[end], clims)
xlims!(axes[end], (-0.6, 1.0))
axes[end].xticks = (-0.6:0.6:0.6, string.(-0.6:0.6:0.6))
Label(gl1b[2,1], popfirst!(subfigureLabels), fontsize=24) 
colsize!(gl1b, 1, Relative(1.0))
rowsize!(gl1b, 1, Relative(0.99))
rowsize!(gl1b, 2, Relative(0.01))

gl2 = GridLayout(fig[1,2])
push!(axes, Axis(gl2[1,1], aspect=DataAspect()))
hidedecorations!(axes[end])
hidespines!(axes[end])
for i=1:size(B2,1)
    poly!(axes[end], cellPolygons2[i], color=log10(ΔpEff[i]), colorrange = clims, colormap = :bam, strokecolor=(:black, 0.2), strokewidth=0.5)
end
Colorbar(gl2[1,2], colorrange=clims, colormap=:bam, height=Relative(0.6), label=L"\log_{10}\left(|\Delta P_{eff}|\right)")
Label(gl2[2,1:2], popfirst!(subfigureLabels), fontsize=24) 
colsize!(gl2, 1, Relative(0.9))
colsize!(gl2, 2, Relative(0.1))
rowsize!(gl2, 1, Relative(0.95))
rowsize!(gl2, 2, Relative(0.05))

gl2b = GridLayout(fig[2,2])
push!(axes, Axis(gl2b[1,1]))#, aspect=AxisAspect(1.25)))
scatter!(axes[end], log10.(Rᵢ2[Not(iᵖ2)]), log10.(abs.(ΔpEff[Not(iᵖ2)])), color=(:blue, 0.25))
scatter!(axes[end], log10.(Rᵢ2[iᵖ2]), log10.(abs.(ΔpEff[iᵖ2])), color=(:red, 0.25))
# lines!(axes[end], log10.(Rᵢ2dummy), log10.(0.005*1.0./Rᵢ2dummy), color=(:black, 0.75), linestyle=:dash)
lines!(axes[end], log10.(Rᵢ2dummy), log10.(0.008*1.0./(Rᵢ2dummy).^2), color=(:black, 0.75), linestyle=:dash)
axes[end].ylabel = L"\log_{10}\left(|\Delta P_{eff}|\right)"
axes[end].xlabel = L"\log_{10}\left(R_i\right)"
ylims!(axes[end], clims)
xlims!(axes[end], (-0.6, 1.0))
axes[end].xticks = (-0.6:0.6:0.6, string.(-0.6:0.6:0.6))
Label(gl2b[2,1], popfirst!(subfigureLabels), fontsize=24) 
colsize!(gl2b, 1, Relative(1.0))
rowsize!(gl2b, 1, Relative(0.99))
rowsize!(gl2b, 2, Relative(0.01))




cellCentres1 = findCellCentresOfMass(R, A, B)
cellCentres2 = findCellCentresOfMass(R2, A2, B2)

cellradii2 = norm.([cellCentres2[i].-systemCOM2 for i=1:size(B2,1)])
cellDummyDists2 = collect(maximum(cellradii2)/100:maximum(cellradii2)/100:maximum(cellradii2))
vertexradii2 = norm.([R2[k].-systemCOM2 for k=1:size(A2,2)])
vertexDummyDists2 = collect(maximum(vertexradii2)/100:maximum(vertexradii2)/100:maximum(vertexradii2))
# edgeradii2 = norm.([𝐜ⱼ2[j].-systemCOM2 for j=1:size(A2,1)])
# edgeDummyDists2 = collect(maximum(edgeradii2)/100:maximum(edgeradii2)/100:maximum(edgeradii2))
# ΔRₖ = R2.-R 
Δrᵢ = cellCentres2.-cellCentres1[Not(ablatedCells)]
# Δrⱼ = 𝐜ⱼ2.-𝐜ⱼ1

#%%

directions = [normalize(Δrᵢ[i])⋅normalize(cellCentres2[i].-systemCOM2) for i=1:size(B2,1)]
# maxDirections = maximum(abs.(directions))
# directionsNormalised = 0.5.+directions./2.0
# colours = [(:blue, directionsNormalised[i]) for i=1:I]
climsDirection = (-1.0,1.0)

gl3 = GridLayout(fig[1,3])
push!(axes, Axis(gl3[1,1], aspect=DataAspect()))
for i=1:size(B2,1)
    poly!(axes[end], cellPolygons2[i], color=(:white, 0.0), strokewidth=1, strokecolor=(:black,0.1))
end
# arrowColours = [(:green, 0.2+0.8*norm(Δrᵢ[i])/maximum(norm.(Δrᵢ))) for i=1:I]
# arrowColours = [(:green, norm(Δrᵢ[i])/maximum(norm.(Δrᵢ))) for i=1:I]
# arrowColours = [(:green, 1.0) for i=1:I]
arrows!(axes[end], Point{2,Float64}.(cellCentres2), Vec{2,Float64}.(Δrᵢ), color=directions, colormap=:bam, colorrange=climsDirection, lengthscale=20.0, align=:head)
hidedecorations!(axes[end])
hidespines!(axes[end])
Colorbar(gl3[1,2], colormap=:bam, colorrange=climsDirection, height=Relative(0.6), label=L"\cos\left(\theta\right)")
Label(gl3[2,1:2], popfirst!(subfigureLabels), fontsize=24) 
colsize!(gl3, 1, Relative(0.9))
colsize!(gl3, 2, Relative(0.1))
rowsize!(gl3, 1, Relative(0.95))
rowsize!(gl3, 2, Relative(0.05))

gl3b = GridLayout(fig[2,3])
push!(axes, Axis(gl3b[1,1]))#, aspect=AxisAspect(1.25)))
scatter!(axes[end], log10.(cellradii2), log10.(norm.(Δrᵢ)), color=directions, colormap=:bam, colorrange=climsDirection)
lines!(axes[end], log10.(cellDummyDists2), log10.(0.02./cellDummyDists2), color=(:black, 0.75), linestyle=:dash)
axes[end].xlabel = L"\log_{10}\left(R_i\right)"
axes[end].ylabel = L"\log_{10}\left(|\Delta \mathbf{R}_i|\right)"
ylims!(axes[end], (-4.2, -1.0))
xlims!(axes[end], (-0.6, 1.0))
axes[end].xticks = (-0.6:0.6:0.6, string.(-0.6:0.6:0.6))
Label(gl3b[2,1], popfirst!(subfigureLabels), fontsize=24) 
colsize!(gl3b, 1, Relative(1.0))
rowsize!(gl3b, 1, Relative(0.99))
rowsize!(gl3b, 2, Relative(0.01))

rowsize!(fig.layout, 1, Relative(0.5))
# rowsize!(fig.layout, 2, Relative(0.01))
rowsize!(fig.layout, 2, Relative(0.5))
# rowsize!(fig.layout, 4, Relative(0.01))
# rowsize!(fig.layout, 3, Relative(0.32))
# rowsize!(fig.layout, 6, Relative(0.01))

colsize!(fig.layout, 1, Relative(0.32))
colsize!(fig.layout, 2, Relative(0.32))
colsize!(fig.layout, 3, Relative(0.32))

# rowgap!(fig.layout, 1, Relative(-0.1))
# rowgap!(fig.layout, 2, Relative(-0.1))
# rowgap!(fig.layout, 3, Relative(-0.01))

resize_to_layout!(fig)

display(fig)

save(plotsdir(inputDir, "Figure10AblationStressComparison.png"), fig)
save(plotsdir(inputDir, "Figure10AblationStressComparison.pdf"), fig)