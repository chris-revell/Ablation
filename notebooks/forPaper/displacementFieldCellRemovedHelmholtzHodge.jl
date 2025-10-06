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
@from "$(srcdir("PenrosePseudoInversion.jl"))" using PenrosePseudoInversion
@from "$(srcdir("HelmholtzHodge.jl"))" using HelmholtzHodge

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

I = size(B2,1)
J = size(B2,2)
K = size(A2,2)

cellCentres1 = findCellCentresOfMass(R, A, B)
cellCentres2 = findCellCentresOfMass(R2, A2, B2)

cellPolygons2 = findCellPolygons(R, A, B)

𝐜ⱼ1 = findEdgeMidpoints(R, A)
𝐜ⱼ2 = findEdgeMidpoints(R2, A2)
𝐂ⱼ1 = findCellLinkMidpoints(R, A, B)
𝐂ⱼ2 = findCellLinkMidpoints(R2, A2, B2)

cellradii2 = norm.([cellCentres2[i].-systemCOM2 for i=1:size(B2,1)])
cellDummyDists2 = collect(maximum(cellradii2)/100:maximum(cellradii2)/100:maximum(cellradii2))
vertexradii2 = norm.([R2[k].-systemCOM2 for k=1:size(A2,2)])
vertexDummyDists2 = collect(maximum(vertexradii2)/100:maximum(vertexradii2)/100:maximum(vertexradii2))
edgeradii2 = norm.([𝐜ⱼ2[j].-systemCOM2 for j=1:size(A2,1)])
edgeDummyDists2 = collect(maximum(edgeradii2)/100:maximum(edgeradii2)/100:maximum(edgeradii2))
linkradii2 = norm.([𝐂ⱼ2[j].-systemCOM2 for j=1:size(A2,1)])
linkDummyDists2 = collect(maximum(linkradii2)/100:maximum(linkradii2)/100:maximum(linkradii2))
ΔRₖ = R2.-R 
Δrᵢ = cellCentres2.-cellCentres1[Not(ablatedCells)]
Δrⱼ = 𝐜ⱼ2.-𝐜ⱼ1
# ΔrⱼDual = 𝐂ⱼ2.-𝐂ⱼ1

iⁱ = ones(Int64, I).-findPeripheralCells(B2)
jⁱⁿ = ones(Int64, J).-findPeripheralEdges(B2)
cellAreas2 = findCellAreas(R2, A2, B2)
linkTriangles2 = findCellLinkTriangles(R2, A2, B2)
linkTriangleAreas2 = findCellLinkTriangleAreas(R2, A2, B2)

curlᶜh = curlᶜ(R2, A2, B2, Δrⱼ)   
curlᵛh = curlᵛspokes(R2, A2, B2, Δrⱼ)   
divᶜh = divᶜ(R2, A2, B2, Δrⱼ)
divᵛh = divᵛsuppress(R2, A2, B2, Δrⱼ)
cocurlᶜh = cocurlᶜ(R2, A2, B2, Δrⱼ)   
cocurlᵛh = cocurlᵛspokes(R2, A2, B2, Δrⱼ)  
codivᶜh = codivᶜ(R2, A2, B2, Δrⱼ)
codivᵛh = codivᵛsuppress(R2, A2, B2, Δrⱼ)

L̂v, Lvreindexing = geometricLvHatReduced(R2, A2, B2)
L̂f, Lfreindexing = geometricLfHatReduced(R2, A2, B2)
Lf = geometricLf(R2, A2, B2)
L̂c, Lcreindexing = geometricLcHatReduced(R2, A2, B2)
Lc = geometricLc(R2, A2, B2)
L̂t, Ltreindexing = geometricLtHatReduced(R2, A2, B2)
H = Diagonal(cellAreas2[Lcreindexing])
E = Diagonal(linkTriangleAreas2[Lvreindexing])

𝟙ᶜ = ones(I)
𝟙ᵛ = ones(K)

# ϕpar Lv -divᵛ
ϕpar = L̂v\(-1.0.*divᵛh[Lvreindexing])
ϕpar2 = zeros(K)
ϕpar2[Lvreindexing] .= ϕpar
# ϕperp Lv -codivᵛ
ϕperp = L̂v\(-1.0.*codivᵛh[Lvreindexing])
ϕperp2 = zeros(K)
ϕperp2[Lvreindexing] .= ϕperp
# upar Lf cocurlᶜ
upar = penrosePseudoInversion(L̂f, cocurlᶜh[Lfreindexing], H)
uparCorrection = penrosePseudoInversion(Lf, ones(I), H)
upar .+= uparCorrection .* innerProd(𝟙ᶜ, H, cocurlᶜh) / innerProd(𝟙ᶜ, H, 𝟙ᶜ)
# uperp Lf curlᶜ
uperp = penrosePseudoInversion(L̂f, curlᶜh[Lfreindexing], H)
# uperpCorrection = penrosePseudoInversion(Lf, ones(I), H)
# uperp .+= uperpCorrection .* innerProd(𝟙ᶜ, H, curlᶜh) / innerProd(𝟙ᶜ, H, 𝟙ᶜ)
# ϕCapitalpar Lc -divᶜ
ϕCapitalpar = penrosePseudoInversion(L̂c, -1.0.*divᶜh[Lcreindexing], H)
ϕCapitalparCorrection = Lc\𝟙ᶜ
ϕCapitalpar .+= ϕCapitalparCorrection .* innerProd(𝟙ᶜ, H, -1.0.*divᶜh) / innerProd(𝟙ᶜ, H, 𝟙ᶜ)
# ϕCapitalperp Lc -codivᶜ
ϕCapitalperp = penrosePseudoInversion(L̂c, -1.0.*codivᶜh[Lcreindexing], H)
ϕCapitalperpCorrection = Lc\𝟙ᶜ
ϕCapitalperp .+= ϕCapitalperpCorrection .* innerProd(𝟙ᶜ, H, -1.0.*codivᶜh) / innerProd(𝟙ᶜ, H, 𝟙ᶜ)
# Upar Lt cocurlᵛ
Upar = L̂t\(cocurlᵛh[Ltreindexing])
Upar2 = zeros(K)
Upar2[Ltreindexing] .= Upar
# Uperp Lt curlᵛ
Uperp = L̂t\(curlᵛh[Ltreindexing])
Uperp2 = zeros(K)
Uperp2[Ltreindexing] .= Uperp

𝐯 = primalHH(R2, A2, B2, ϕpar2, ϕperp2, upar, uperp)
# 𝐯 .= [𝐯[j].-]
𝐱 = Δrⱼ.-𝐯
𝐕 = dualHH(R2, A2, B2, ϕCapitalpar, ϕCapitalperp, Upar2, Uperp2)
# 𝐕 .= [𝐕[j].-]
𝐗 = Δrⱼ.-𝐕

maxX = max(maximum(norm.(𝐱[jⁱⁿ.==1])), maximum(norm.(𝐗[jⁱⁿ.==1])))
#%%

fig = Figure(size=(1000, 1000), fontsize=24)
axes = Axis[]
individualLetters = string.(Char.(UInt8.(collect(97:97+26-1))))
subfigureLabels = [L"(%$l)" for l in individualLetters]

push!(axes, Axis(fig[1,1], aspect=DataAspect()))
for i=1:I 
    poly!(axes[end], cellPolygons2[i], color=(:black, 0.1), strokewidth=1, strokecolor=(:black,0.1))
end
arrowColours = [(:blue, norm(𝐱[j])/maxX) for j=1:J]
arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ2)[jⁱⁿ.==1], Vec{2,Float64}.(𝐱[jⁱⁿ.==1]), color=arrowColours[jⁱⁿ.==1], linewidth=2, lengthscale=20.0)
hidedecorations!(axes[end])
hidespines!(axes[end])
Label(fig[2,1,Bottom()], popfirst!(subfigureLabels), fontsize=24) 

push!(axes, Axis(fig[1,2], aspect=AxisAspect(1.25)))
dotColours = [(:blue, norm(𝐱[j])/maxX) for j=1:J]
scatter!(axes[end], log10.(edgeradii2[jⁱⁿ.==1]), log10.(norm.(𝐱[jⁱⁿ.==1])), color=dotColours[jⁱⁿ.==1])#, color=directions, colormap=:bam, colorrange=clims)
lines!(axes[end], log10.(edgeDummyDists2), log10.(0.02./edgeDummyDists2), color=(:black, 0.75), linestyle=:dash)
axes[end].xlabel = L"\log_{10}\left(c_j\right)"
axes[end].ylabel = L"\log_{10}\left(\breve{\mathbf{x}}_j\right)"


# push!(axes, Axis(fig[3,1], aspect=DataAspect()))
# for i=1:I 
#     poly!(axes[end], cellPolygons2[i], color=(:white, 0.0), strokewidth=1, strokecolor=(:black,0.1))
# end
# arrows!(axes[end], Point{2,Float64}.(𝐂ⱼ2[jⁱⁿ.==1]), Vec{2,Float64}.(𝐗[jⁱⁿ.==1]), color=:green) #color=directions[jⁱⁿ.==1], colormap=:bam, colorrange=clims, lengthscale=20.0, align=:head)
# hidedecorations!(axes[end])
# hidespines!(axes[end])
# Label(fig[4,1,Bottom()], popfirst!(subfigureLabels), fontsize=24) 

push!(axes, Axis(fig[3,1], aspect=DataAspect()))
for i=1:I 
    poly!(axes[end], cellPolygons2[i], color=(:black, 0.1), strokewidth=1, strokecolor=(:black,0.1))
end
# arrowColours = [(:green, norm(𝐗[j])/maximum(norm.(𝐗[jⁱⁿ.==1]))) for j=1:J]
arrowColours = [(:green, norm(𝐗[j])/maxX) for j=1:J]
arrows!(axes[end], Point{2,Float64}.(𝐂ⱼ2)[jⁱⁿ.==1], Vec{2,Float64}.(𝐗[jⁱⁿ.==1]), color=arrowColours[jⁱⁿ.==1], linewidth=2, lengthscale=20.0)
# arrows!(axes[end], Point{2,Float64}.(𝐜ⱼ)[jⁱⁿ.==1], Vec{2,Float64}.(𝐗[jⁱⁿ.==1]), color=:red, linewidth=2, lengthscale=10.0)
Label(fig[4,1,Bottom()], popfirst!(subfigureLabels), fontsize=24) 
hidedecorations!(axes[end])
hidespines!(axes[end])

push!(axes, Axis(fig[3,2], aspect=AxisAspect(1.25)))
dotColours = [(:green, norm(𝐱[j])/maxX) for j=1:J]
scatter!(axes[end], log10.(linkradii2[jⁱⁿ.==1]), log10.(norm.(𝐗[jⁱⁿ.==1])), color=dotColours[jⁱⁿ.==1])#, colormap=:bam, colorrange=clims)
lines!(axes[end], log10.(linkDummyDists2), log10.(0.02./linkDummyDists2), color=(:black, 0.75), linestyle=:dash)
axes[end].xlabel = L"\log_{10}\left(C_j\right)"
axes[end].ylabel = L"\log_{10}\left(\breve{\mathbf{X}}_j\right)"
Label(fig[4,2,Bottom()], popfirst!(subfigureLabels), fontsize=24) 

rowsize!(fig.layout, 1, Relative(0.48))
rowsize!(fig.layout, 2, Relative(0.02))
rowsize!(fig.layout, 3, Relative(0.48))
rowsize!(fig.layout, 4, Relative(0.02))

rowgap!(fig.layout, 1, Relative(-0.01))
rowgap!(fig.layout, 3, Relative(-0.01))

colsize!(fig.layout, 1, Relative(0.5))
colsize!(fig.layout, 2, Relative(0.5))
# colsize!(fig.layout, 3, Relative(0.5))
display(fig)

save(plotsdir(inputDir, "displacementHelmholtzHodge.png"), fig)
save(plotsdir(inputDir, "displacementHelmholtzHodge.pdf"), fig)


