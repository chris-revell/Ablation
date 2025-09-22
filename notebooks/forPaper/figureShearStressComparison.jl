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

inputSystem = "Large5"
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
    integ2 = vertexModel(abstol = 1e-10,
                        reltol = 1e-10,
                        initialSystem="argument",
                        divisionToggle=0,
                        R_in=Rtmp,
                        A_in=Atmp,
                        B_in=Btmp,
                        pressureExternal=0.0,
                        nCycles=1.0,
                        outputToggle=0,
                        frameDataToggle=0,
                        frameImageToggle=0,
                        videoToggle=0,
                        printToggle=1,
                        energyModel="quadratic",
                    )
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

#%%
    
cellCentres2 = findCellCentresOfMass(R2, A2, B2)

aᵢ1 = findCellAreas(R, A, B)
aᵢ2 = findCellAreas(R2, A2, B2)

rᵢ2 = norm.([cellCentres2[i].-systemCOM2 for i=1:size(B2,1)])
rᵢ2dummy = collect(maximum(rᵢ2)/100:maximum(rᵢ2)/100:maximum(rᵢ2))

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

fig = Figure(size=(1000,1000), fontsize=24)
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"(%$l)" for l in individualLetters]

push!(axes, Axis(fig[1,1], aspect=DataAspect()))
hidedecorations!(axes[end])
hidespines!(axes[end])
for i=1:size(B2,1)
    poly!(axes[end], cellPolygons2[i], color=log10(Δζ[i]), colorrange = clims, colormap = :bam, strokecolor=(:black, 0.2), strokewidth=0.5)
end
Colorbar(fig[1,2], colorrange=clims, colormap=:bam, height=Relative(0.8), label=L"\log_{10}\left(|\Delta\zeta_i|\right)")
Label(fig[2,1], popfirst!(subfigureLabels), fontsize=24) 

push!(axes, Axis(fig[1,3], aspect=AxisAspect(1)))#, xscale=log10, yscale=log10))
# push!(axes, Axis(fig[1,3], aspect=AxisAspect(1.5)))#, aspect=AxisAspect(1)))#, xscale=log10, yscale=log10))
scatter!(axes[end], log10.(rᵢ2[Not(iᵖ2)]), log10.(abs.(Δζ[Not(iᵖ2)])), color=(:blue, 0.25))
scatter!(axes[end], log10.(rᵢ2[iᵖ2]), log10.(abs.(Δζ[iᵖ2])), color=(:red, 0.25))
lines!(axes[end], log10.(rᵢ2dummy), log10.(0.01*1.0./rᵢ2dummy), color=(:black, 0.75), linestyle=:dash)
lines!(axes[end], log10.(rᵢ2dummy), log10.(0.01*1.0./rᵢ2dummy.^2), color=(:black, 0.75), linestyle=:dash)
axes[end].ylabel = L"\log_{10}\left(|\Delta\zeta_i|\right)"
axes[end].xlabel = L"\log_{10}\left(r_i\right)"
xlims!(axes[end], (minimum(log10.(rᵢ2)), maximum(log10.(rᵢ2))))
# xlims!(axes[end], (0.8,1.0))
ylims!(axes[end], clims)
Label(fig[2,3], popfirst!(subfigureLabels), fontsize=24) 
# ylims!(axes[end], (minimum(abs.(Δζ)),1.0))


push!(axes, Axis(fig[3,1], aspect=DataAspect()))
hidedecorations!(axes[end])
hidespines!(axes[end])
for i=1:size(B2,1)
    poly!(axes[end], cellPolygons2[i], color=log10(ΔpEff[i]), colorrange = clims, colormap = :bam, strokecolor=(:black, 0.2), strokewidth=0.5)
end
Colorbar(fig[3,2], colorrange=clims, colormap=:bam, height=Relative(0.8), label=L"\log_{10}\left(|\Delta P_{eff}|\right)")
Label(fig[4,1], popfirst!(subfigureLabels), fontsize=24) 

push!(axes, Axis(fig[3,3], aspect=AxisAspect(1)))#, xscale=log10, yscale=log10))
# push!(axes, Axis(fig[3,3], aspect=AxisAspect(1.5)))#, aspect=AxisAspect(1)))#, xscale=log10, yscale=log10))
scatter!(axes[end], log10.(rᵢ2[Not(iᵖ2)]), log10.(abs.(ΔpEff[Not(iᵖ2)])), color=(:blue, 0.25))
scatter!(axes[end], log10.(rᵢ2[iᵖ2]), log10.(abs.(ΔpEff[iᵖ2])), color=(:red, 0.25))
lines!(axes[end], log10.(rᵢ2dummy), log10.(0.01*1.0./rᵢ2dummy), color=(:black, 0.75), linestyle=:dash)
lines!(axes[end], log10.(rᵢ2dummy), log10.(0.01*1.0./(rᵢ2dummy).^2), color=(:black, 0.75), linestyle=:dash)
axes[end].ylabel = L"\log_{10}\left(|\Delta P_{eff}|\right)"
axes[end].xlabel = L"\log_{10}\left(r_i\right)"
xlims!(axes[end], (minimum(log10.(rᵢ2)), maximum(log10.(rᵢ2))))
# xlims!(axes[end], (0.8,1.0))
ylims!(axes[end], clims)
Label(fig[4,3], popfirst!(subfigureLabels), fontsize=24) 
# ylims!(axes[end], (minimum(abs.(ΔpEff)),1.0))

# resize_to_layout!(fig)

rowsize!(fig.layout, 1, Relative(0.49))
rowsize!(fig.layout, 2, Relative(0.01))
rowsize!(fig.layout, 3, Relative(0.49))
rowsize!(fig.layout, 4, Relative(0.01))

colsize!(fig.layout, 1, Relative(0.5))
colsize!(fig.layout, 2, Relative(0.1))
colsize!(fig.layout, 3, Relative(0.4))

rowgap!(fig.layout, 1, Relative(-0.0001))
rowgap!(fig.layout, 3, Relative(-0.0001))
colgap!(fig.layout, 1, Relative(-0.01))
resize_to_layout!(fig)

display(fig)

save(plotsdir(inputDir, "$(inputSystem)_comparison.png"), fig)